# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Repeat frozen policies on complete, fixed-client ShareGPT conversations."""

import argparse
import csv
import random
from collections import Counter
from pathlib import Path

from run_lifecycle_calibrated_smoke import (
    ROOT,
    digest,
    evaluation_config,
    kv_config,
    read,
    run_arm,
    save,
)

ARMS = ["no-offload", "all-block", "host-no-probe", "host-probe"]
DECISION_ARMS = [
    "no-offload",
    "all-block",
    "all-block-control",
    "admission-only",
    "promotion-only",
    "joint-no-probe",
]
ADMISSION_ARMS = [
    "all-block-control",
    "promotion-only",
    "joint-no-probe",
    "joint-new-write",
]
CAPACITY_ARMS = [
    "no-offload",
    "all-block-2g",
    "selective-2g",
    "all-block-4g",
    "selective-4g",
]
CODE = [
    "scripts/run_lifecycle_fixed_workload.py",
    "scripts/run_lifecycle_calibrated_smoke.py",
    "scripts/run_trace_driven_value_policy_online_sweep.sh",
    "benchmarks/multi_turn/benchmark_serving_multi_turn.py",
    "vllm/v1/kv_offload/lifecycle_policy.py",
    "vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py",
    "vllm/v1/kv_offload/reuse_manager.py",
    "vllm/v1/kv_offload/cpu.py",
    "vllm/v1/kv_offload/abstract.py",
    "vllm/v1/kv_offload/lru_manager.py",
    "vllm/v1/kv_offload/arc_manager.py",
    "scripts/summarize_lifecycle_admission.py",
    "scripts/summarize_lifecycle_capacity.py",
]


def suite_config(suite, arm, fit, probe):
    if suite == "admission-capacity":
        if arm not in CAPACITY_ARMS:
            raise ValueError(f"Unknown admission-capacity arm: {arm}")
        if arm == "no-offload":
            return None
        selective = arm.startswith("selective")
        cfg = suite_config(
            "decision-ablation",
            "admission-only" if selective else "all-block-control",
            fit,
            probe,
        )
        extra = cfg["kv_connector_extra_config"]
        extra["cpu_bytes_to_use"] = (2 if arm.endswith("2g") else 4) * 1024**3
        extra["max_cost_events"] = 32768
        extra["lifecycle_policy"].update(
            trace_admission=True,
            new_write_store_cost=True,
        )
        return cfg
    if suite == "admission-diagnostic":
        if arm not in ADMISSION_ARMS:
            raise ValueError(f"Unknown admission-diagnostic arm: {arm}")
        base = "joint-no-probe" if arm == "joint-new-write" else arm
        cfg = suite_config("decision-ablation", base, fit, probe)
        extra = cfg["kv_connector_extra_config"]
        extra["max_cost_events"] = 32768
        extra["lifecycle_policy"].update(
            trace_admission=True, new_write_store_cost=arm == "joint-new-write"
        )
        return cfg
    if suite == "cost-probe":
        if arm not in ARMS:
            raise ValueError(f"Unknown cost-probe arm: {arm}")
        return evaluation_config(arm, fit, probe)
    if suite != "decision-ablation" or arm not in DECISION_ARMS:
        raise ValueError(f"Unknown suite/arm: {suite}/{arm}")
    if arm in ("no-offload", "all-block"):
        return evaluation_config(arm, fit, probe)
    # The four factorial cells share the same wrapper, cost priors and logging.
    # Only admission and the promotion action choice are toggled.
    policy = dict(
        fit["policy"],
        store_s_per_block=fit["store_cost_s_per_block"]["host-service"],
        admission_enabled=arm in ("admission-only", "joint-no-probe"),
        guard_enabled=False,
        cold_probe_every=0,
    )
    if policy["store_s_per_block"] is None:
        raise ValueError("Missing calibrated host-service cost")
    action = (
        "cost_aware"
        if arm in ("promotion-only", "joint-no-probe")
        else "always_restore"
    )
    return kv_config(action, policy)


def expected_requests(data):
    expected = Counter()
    ids = set()
    for index, item in enumerate(data):
        conv = str(item["id"])
        if conv in ids:
            raise ValueError(f"Duplicate conversation ID: {conv}")
        ids.add(conv)
        messages = item["messages"]
        if not messages or any(
            m["role"] != ("user" if i % 2 == 0 else "assistant")
            for i, m in enumerate(messages)
        ):
            raise ValueError(f"Non-alternating conversation: {conv}")
        for turn in range(1, len(messages) + 1, 2):
            expected[(conv, str(turn), str(index % 4))] += 1
    return expected


def audit_arm(metrics, expected, probe):
    path = Path(metrics["stats_path"])
    with path.with_name("request_summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    actual = Counter(
        (r["conversation_id"], r["input_num_turns"], r["client_id"]) for r in rows
    )
    if actual != expected:
        raise ValueError(
            f"Request/client mismatch: extra={actual - expected}, "
            f"missing={expected - actual}"
        )
    stats = read(path)["stats"]
    for key in ("offload_store_failed_count", "offload_store_completion_failed_blocks"):
        if stats.get(key, 0):
            raise ValueError(f"Failed store: {key}")
    blocks = stats.get("lcuc_cold_probe_blocks_spent", 0)
    cost = stats.get("lcuc_cold_probe_cost_spent_s", 0)
    if blocks > probe["cold_probe_budget_blocks"] or (
        cost > probe["cold_probe_budget_s"] + 1e-12
    ):
        raise ValueError("Probe budget exceeded")
    output = read(path.with_name("output.json"))
    if {str(item["id"]) for item in output} != {k[0] for k in expected}:
        raise ValueError("Not all conversations completed")
    duration_ms = max(float(r["start_time_ms"]) + float(r["latency_ms"]) for r in rows)
    duration_ms -= min(float(r["start_time_ms"]) for r in rows)
    metrics.update(
        request_set_and_client_assignment_exact=True,
        cold_probe_blocks=blocks,
        cold_probe_estimated_cost_ms=cost * 1000,
        request_span_s=duration_ms / 1000,
        requests_per_s=len(rows) / (duration_ms / 1000),
    )
    return sorted(
        (r["conversation_id"], r["input_num_turns"], r["input_num_tokens"])
        for r in rows
    )


def main(args):
    source_manifest = read(args.source / "manifest.json")
    if args.model != source_manifest["model"]:
        raise ValueError("Model must match the frozen calibration")
    fit = read(args.source / "calibration_fit.json")
    data = read(args.source / "evaluation.json")
    expected = expected_requests(data)
    orders = []
    rng = random.Random(20260917)
    for _ in range(args.repeats):
        suite_arms = {
            "decision-ablation": DECISION_ARMS,
            "admission-diagnostic": ADMISSION_ARMS,
            "cost-probe": ARMS,
            "admission-capacity": CAPACITY_ARMS,
        }
        order = suite_arms[args.suite].copy()
        rng.shuffle(order)
        if args.suite == "admission-capacity" and not orders:
            # Commission the pressure configuration before the other 14 runs.
            order.remove("all-block-2g")
            order.insert(0, "all-block-2g")
        orders.append(order)
    manifest = {
        "source": str(args.source.resolve()),
        "input_sha256": digest(args.source / "evaluation.json"),
        "fit_sha256": digest(args.source / "calibration_fit.json"),
        "gpu": args.gpu,
        "port": args.port,
        "model": args.model,
        "conversations": len(data),
        "requests_per_arm": sum(expected.values()),
        "runtime_seed": 0,
        "suite": args.suite,
        "primary_metric": "p95_ttft_ms",
        "orders": orders,
        "cold_probe_config": source_manifest["cold_probe_config"],
        "code_sha256": {p: digest(ROOT / p) for p in CODE},
        "scope": "fixed complete conversations, fixed client assignment, closed loop",
        "limitations": [
            "global arrival order and timing remain policy-dependent",
            "same previously inspected evaluation dataset, not unseen validation",
            "same seed repeated runs measure run variability, not dataset diversity",
            "host cost proxy omits asynchronous resource contention",
            "guard disabled, no long-term protection claim",
        ],
    }
    if (args.output / "manifest.json").exists():
        if read(args.output / "manifest.json") != manifest:
            raise ValueError("Resume configuration/code differs from frozen manifest")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        save(args.output / "manifest.json", manifest)
        save(args.output / "evaluation.json", data)
        save(args.output / "calibration_fit.json", fit)
    signatures = None
    results = []
    for repeat, order in enumerate(orders):
        for arm in order:
            if digest(args.output / "evaluation.json") != manifest["input_sha256"]:
                raise ValueError("Frozen input changed")
            if digest(args.output / "calibration_fit.json") != manifest["fit_sha256"]:
                raise ValueError("Frozen cost fit changed")
            for p, sha in manifest["code_sha256"].items():
                if digest(ROOT / p) != sha:
                    raise ValueError(f"Frozen code changed: {p}")
            metrics = run_arm(
                args,
                f"repeat_{repeat}/{arm}",
                args.output / "evaluation.json",
                suite_config(args.suite, arm, fit, manifest["cold_probe_config"]),
                0,
                complete_workload=True,
            )
            sig = audit_arm(metrics, expected, manifest["cold_probe_config"])
            if args.suite != "cost-probe" and metrics["cold_probe_blocks"]:
                raise ValueError("Unexpected cold probe in decision ablation")
            if args.suite == "admission-diagnostic" or (
                args.suite == "admission-capacity" and arm != "no-offload"
            ):
                from summarize_lifecycle_admission import summarize_stats

                diagnostic = summarize_stats(read(Path(metrics["stats_path"]))["stats"])
                save(
                    Path(metrics["stats_path"]).with_name("admission_summary.json"),
                    diagnostic,
                )
                if args.suite == "admission-capacity":
                    stats = read(Path(metrics["stats_path"]))["stats"]
                    if metrics["recompute_decisions"] or metrics["guard_transitions"]:
                        raise ValueError("Unexpected promotion/guard in capacity suite")
                    metrics["cpu_evicted_blocks"] = stats.get(
                        "offload_store_evicted_blocks", 0
                    )
                    if arm == "all-block-2g" and (
                        not metrics["cpu_evicted_blocks"]
                        or metrics["load_blocks"] < 100
                        or metrics["restore_decisions"] < 10
                    ):
                        raise ValueError(
                            "2 GiB requires evictions, >=100 loaded blocks, "
                            "and >=10 restore decisions"
                        )
            if signatures is None:
                signatures = sig
            elif signatures != sig:
                raise ValueError("Input token counts differ across arms")
            metrics["repeat"] = repeat
            metrics["arm"] = arm
            results.append(metrics)
            save(args.output / "evaluation_summary.json", results)
            print(
                f"AUDIT PASS {metrics['name']}: {len(sig)} identical requests",
                flush=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--port", type=int, default=8238)
    parser.add_argument("--model", default="/public/model_zoo/Qwen3-0.6B")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--suite",
        choices=[
            "cost-probe",
            "decision-ablation",
            "admission-diagnostic",
            "admission-capacity",
        ],
        default="cost-probe",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    main(args)
