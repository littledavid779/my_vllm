# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Disjoint calibration then a frozen six-arm GPU smoke (not a full evaluation)."""

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import random
import signal
import socket
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/direct_sharegpt_multiturn_20260806"


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def content_key(item):
    return hashlib.sha256(
        json.dumps(item["messages"], sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def disjoint_candidates(candidates, evaluation):
    ids = {x["id"] for x in evaluation}
    contents = {content_key(x) for x in evaluation}
    return [
        x for x in candidates if x["id"] not in ids and content_key(x) not in contents
    ]


def fit_nonnegative_line(points):
    """Least squares with nonnegative intercept/slope, including boundary fits."""
    if not points or any(
        x <= 0 or y < 0 or not math.isfinite(x + y) for x, y in points
    ):
        raise ValueError("Cost samples must be finite with x>0 and y>=0")
    n = len(points)
    sx, sy = sum(x for x, y in points), sum(y for x, y in points)
    sxx, sxy = sum(x * x for x, y in points), sum(x * y for x, y in points)
    candidates = [(sy / n, 0.0), (0.0, sxy / sxx)]
    denom = n * sxx - sx * sx
    if denom > 0:
        slope = (n * sxy - sx * sy) / denom
        intercept = (sy - slope * sx) / n
        if slope >= 0 and intercept >= 0:
            candidates.append((intercept, slope))
    return min(
        candidates, key=lambda c: sum((y - c[0] - c[1] * x) ** 2 for x, y in points)
    )


def gpu_snapshot(gpu):
    info = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-gpu=uuid,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    uuid = info.split(",")[0].strip()
    apps = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    pids = [
        int(line.split(",")[1])
        for line in apps.splitlines()
        if line.split(",")[0].strip() == uuid
    ]
    return {"time": time.time(), "gpu": gpu, "info": info, "pids": pids}


def kv_config(action="always_restore", policy=None):
    extra = {
        "cpu_bytes_to_use": 4 * 1024**3,
        "block_size": 16,
        "eviction_policy": "lru",
        "store_threshold": 0,
        "track_lifecycle_stats": True,
        "track_cost_events": True,
        "max_cost_events": 8192,
        "promotion_policy": action,
    }
    if policy is not None:
        extra["lifecycle_policy"] = policy
    return {
        "kv_connector": "OffloadingConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": extra,
    }


def run_arm(
    args,
    name,
    input_path,
    config,
    seed,
    *,
    complete_workload=False,
    correlate_requests=False,
    request_path_trace=False,
    restore_path_trace=False,
    restore_ready_mode="off",
    restore_readiness_trace=False,
    restore_host_trace=False,
):
    if restore_ready_mode not in ("off", "observe", "on"):
        raise ValueError("Invalid restore ready-first mode")
    armroot = args.output / name
    armroot.mkdir(parents=True, exist_ok=True)
    spec = {
        "input": str(input_path),
        "input_sha256": digest(input_path),
        "config": config,
        "seed": seed,
        "gpu": args.gpu,
        "restore_ready_mode": restore_ready_mode,
    }
    if restore_readiness_trace:
        spec["restore_readiness_trace"] = True
    if restore_host_trace:
        spec["restore_host_trace"] = True
    owner_map = None
    if request_path_trace:
        spec["request_path_trace"] = True
    if restore_path_trace:
        spec["restore_path_trace"] = True
    if correlate_requests:
        owner_map = {
            f"chatcmpl-{item['id']}-turn{n + 1}": f"chatcmpl-{item['id']}"
            for item in read(input_path)
            for n, message in enumerate(item["messages"])
            if message["role"] == "user"
        }
        spec.update(correlate_requests=True, request_owner_map=owner_map)
    expected_requests = 128
    if complete_workload:
        spec["complete_workload"] = True
        expected_requests = sum(
            sum(m["role"] == "user" for m in item["messages"])
            for item in read(input_path)
        )
    metrics_file = armroot / "metrics.json"
    if metrics_file.exists():
        if read(armroot / "spec.json") != spec:
            raise ValueError(f"Existing arm configuration differs: {name}")
        return read(metrics_file)
    before = gpu_snapshot(args.gpu)
    if before["pids"]:
        raise RuntimeError(f"GPU is occupied; leaving other jobs untouched: {before}")
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", args.port)) == 0:
            raise RuntimeError(f"Port {args.port} is already occupied")
    save(armroot / "spec.json", spec)
    env = os.environ.copy()
    env.update(
        GPU_ID=str(args.gpu),
        PORT_BASE=str(args.port),
        GPU_MEMORY_UTILIZATION="0.05",
        NUM_GPU_BLOCKS_OVERRIDE="1000",
        MODEL_PATH=args.model,
        MAX_MODEL_LEN="4096",
        MAX_ACTIVE_CONVERSATIONS="32",
        NUM_CLIENTS="4",
        MAX_NUM_REQUESTS="128",
        COMPLETE_WORKLOAD="1" if complete_workload else "0",
        LIMIT_MIN_TOKENS="16",
        LIMIT_MAX_TOKENS="32",
        REQUEST_RATE="0",
        CONVERSATION_SAMPLING="round_robin",
        SEED=str(seed),
        INPUT_FILE=str(input_path),
        RESULT_ROOT=str(armroot),
        SERVER_WAIT_SEC="300",
        KV_TRANSFER_CONFIG_JSON=json.dumps(config) if config else "",
        MULTITURN_CORRELATE_REQUESTS="1" if correlate_requests else "0",
        VLLM_EXPERIMENTAL_REQUEST_PATH_TRACE="1" if request_path_trace else "0",
        VLLM_EXPERIMENTAL_RESTORE_PATH_TRACE="1" if restore_path_trace else "0",
        VLLM_EXPERIMENTAL_RESTORE_READY_FIRST=restore_ready_mode,
        VLLM_EXPERIMENTAL_RESTORE_READINESS_TRACE=(
            "1" if restore_readiness_trace else "0"
        ),
        VLLM_EXPERIMENTAL_RESTORE_HOST_TRACE="1" if restore_host_trace else "0",
    )
    if correlate_requests:
        env["VLLM_EXPERIMENTAL_GPU_EVICTION_REQUEST_OWNER_MAP"] = json.dumps(owner_map)
        env["MULTITURN_BENCH_MAX_RETRIES"] = "0"
    print(f"START {name}", flush=True)
    samples = [before]
    foreign = set()
    with (armroot / "runner.log").open("w") as log:
        process = subprocess.Popen(
            [
                "bash",
                str(ROOT / "scripts/run_trace_driven_value_policy_online_sweep.sh"),
                "lru",
            ],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            started = time.monotonic()
            while process.poll() is None:
                snap = gpu_snapshot(args.gpu)
                samples.append(snap)
                for pid in snap["pids"]:
                    try:
                        if os.getpgid(pid) != process.pid:
                            foreign.add(pid)
                    except ProcessLookupError:
                        pass
                if time.monotonic() - started > 600:
                    raise TimeoutError(name)
                time.sleep(2)
            if process.returncode:
                raise RuntimeError(f"{name} failed, see {armroot}/runner.log")
        finally:
            # Only our freshly created process group; never existing GPU jobs.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=30)
            save(
                armroot / "gpu_samples.json",
                {
                    "samples": samples,
                    "foreign_pids": sorted(foreign),
                    "sampling_interval_s": 2,
                },
            )
    if foreign:
        raise RuntimeError(f"Foreign GPU processes appeared; arm invalid: {foreign}")
    paths = list(armroot.glob("*/seed_*/lru/value_policy_stats.json"))
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one stats file in {armroot}")
    path = paths[0]
    stats = read(path)["stats"]
    rows = list(csv.DictReader((path.parent / "request_summary.csv").open()))
    if len(rows) != expected_requests or any(
        float(r["output_num_tokens"]) <= 0 for r in rows
    ):
        raise ValueError(f"Incomplete request records: {name}")
    for key in (
        "offload_cost_events_dropped",
        "offload_cost_event_unmatched_transfers",
        "lcuc_pending",
        "lcuc_incomplete_promotions",
    ):
        if stats.get(key, 0):
            raise ValueError(f"Incomplete observations: {name}: {key}")
    ttft = sorted(float(r["ttft_ms"]) for r in rows)

    def quantile(q):
        pos = (len(ttft) - 1) * q
        lo = int(pos)
        return ttft[lo] + (ttft[min(lo + 1, len(ttft) - 1)] - ttft[lo]) * (pos - lo)

    events = stats.get("offload_cost_events", [])
    metrics = {
        "name": name,
        "stats_path": str(path),
        "requests": len(rows),
        "mean_ttft_ms": sum(ttft) / len(ttft),
        "p50_ttft_ms": quantile(0.5),
        "p95_ttft_ms": quantile(0.95),
        "p99_ttft_ms": quantile(0.99),
        "store_blocks": stats.get("offload_store_actual_blocks", 0),
        "load_blocks": stats.get("offload_load_completed_blocks", 0),
        "d2h_bytes": sum(
            e.get("dma_bytes", 0) for e in events if e["event_type"] == "store"
        ),
        "h2d_bytes": sum(
            e.get("dma_bytes", 0) for e in events if e["event_type"] == "promotion"
        ),
        "restore_decisions": stats.get(
            "lcuc_restore_decisions",
            stats.get("offload_promotion_restore_decision_count", 0),
        ),
        "recompute_decisions": stats.get(
            "lcuc_recompute_decisions",
            stats.get("offload_promotion_recompute_decision_count", 0),
        ),
        "guard": stats.get("lcuc_guard_state", "disabled"),
        "guard_transitions": stats.get("lcuc_guard_transitions", 0),
        "estimated_net_utility_s": stats.get("lcuc_estimated_net_utility_s"),
        "decision_time_s": stats.get("lcuc_decision_time_s", 0),
    }
    save(metrics_file, metrics)
    print(f"DONE {name}: {json.dumps(metrics)}", flush=True)
    return metrics


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=False)
    evaluation = read(DATA / "sharegpt_direct_seed0_128.json")
    # Exclude IDs and whole-conversation content from all three existing eval sets.
    excluded = sum(
        (read(DATA / f"sharegpt_direct_seed{i}_128.json") for i in range(3)), []
    )
    candidates = disjoint_candidates(
        read(DATA / "sharegpt_direct_seed1_candidates512.json"), excluded
    )
    save(args.output / "calibration_candidates.json", candidates)
    subprocess.run(
        [
            str(ROOT / ".venv/bin/python"),
            str(ROOT / "scripts/build_direct_sharegpt_workload.py"),
            str(args.output / "calibration_candidates.json"),
            str(args.output / "calibration.json"),
            "--metadata-output",
            str(args.output / "calibration.meta.json"),
            "--tokenizer",
            args.model,
            "--converter-seed",
            "1",
            "--max-conversations",
            "128",
        ],
        cwd=ROOT,
        check=True,
    )
    save(args.output / "evaluation.json", evaluation)
    calibration = read(args.output / "calibration.json")
    assert len(disjoint_candidates(calibration, excluded)) == len(calibration)
    arms = [
        "no-offload",
        "all-block",
        "admission",
        "promotion",
        "joint-no-guard",
        "full",
    ]
    if args.evaluation_suite == "cost-probe-ablation":
        arms = [
            "no-offload",
            "all-block",
            "completion-no-probe",
            "completion-probe",
            "host-no-probe",
            "host-probe",
        ]
    random.Random(20260916).shuffle(arms)
    save(
        args.output / "manifest.json",
        {
            "gpu": args.gpu,
            "model": args.model,
            "calibration_sha256": digest(args.output / "calibration.json"),
            "evaluation_sha256": digest(args.output / "evaluation.json"),
            "calibration_eval_id_overlap": 0,
            "calibration_eval_full_content_overlap": 0,
            "calibration_repeats": 2,
            "evaluation_order": arms,
            "evaluation_suite": args.evaluation_suite,
            "cold_probe_config": {
                "cold_probe_every": 8,
                "cold_probe_min_blocks": 16,
                "cold_probe_max_blocks": 64,
                "cold_probe_budget_blocks": 128,
                "cold_probe_budget_s": 0.1,
            },
            "evaluation_seed": 0,
            "store_cost_proxy": (
                f"{args.store_cost_source}; proxy, not total marginal latency"
            ),
            "restore_fit": "nonnegative affine LS on complete calibration events",
            "recompute_fit": "through-origin LS on complete calibration events",
            "limitations": [
                "single GPU/model, closed-loop, one evaluation runtime seed",
                "short run does not validate default 30-second guard",
                "GPU sampling is not a resource reservation",
            ],
            "code_sha256": {
                p: digest(ROOT / p)
                for p in [
                    "scripts/run_lifecycle_calibrated_smoke.py",
                    "vllm/v1/kv_offload/lifecycle_policy.py",
                    "vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py",
                ]
            },
        },
    )


def calibrate(args):
    manifest = read(args.output / "manifest.json")
    expected_source = manifest.get("store_cost_proxy", "")
    if args.store_cost_source == "host-service" and not expected_source.startswith(
        "host-service"
    ):
        raise ValueError(
            "Use a new prepared output directory for host-service calibration"
        )
    if args.store_cost_source == "worker-completion" and expected_source.startswith(
        "host-service"
    ):
        raise ValueError("Store cost source differs from prepared manifest")
    restore, recompute, stores = [], [], []
    for rep in range(2):
        for action in (
            ["restore", "recompute"] if rep == 0 else ["recompute", "restore"]
        ):
            m = run_arm(
                args,
                f"calibration/{rep}_{action}",
                args.output / "calibration.json",
                kv_config("always_" + action),
                rep,
            )
            events = read(m["stats_path"])["stats"]["offload_cost_events"]
            for event in events:
                if event["event_type"] == "store" and event.get("stored_blocks", 0) > 0:
                    if event.get("submit_to_complete_s") is None:
                        raise ValueError("Missing store transfer completion cost")
                    stores.append(event)
                elif event["event_type"] == "promotion":
                    field = action + "_wait_s"
                    if event.get(field) is None or event.get("recompute_incomplete"):
                        raise ValueError("Incomplete promotion cost")
                    (restore if action == "restore" else recompute).append(event)
    if min(len(restore), len(recompute), len(stores)) < 8:
        raise ValueError("Insufficient calibration events; do not invent priors")
    intercept, slope = fit_nonnegative_line(
        [(e["candidate_blocks"], e["restore_wait_s"]) for e in restore]
    )
    cc = sum(e["candidate_tokens"] * e["recompute_wait_s"] for e in recompute) / sum(
        e["candidate_tokens"] ** 2 for e in recompute
    )
    store_field = (
        "store_host_service_s"
        if args.store_cost_source == "host-service"
        else "submit_to_complete_s"
    )
    if any(e.get(store_field) is None for e in stores):
        raise ValueError(
            f"Missing {store_field}; old traces cannot calibrate this source"
        )
    cs = sum(e[store_field] for e in stores) / sum(e["stored_blocks"] for e in stores)
    policy = {
        "store_s_per_block": cs,
        "restore_s_per_block": slope,
        "restore_fixed_s": intercept,
        "recompute_s_per_token": cc,
        "max_decision_events": 8192,
    }
    from vllm.v1.kv_offload.lifecycle_policy import LifecyclePolicyConfig

    LifecyclePolicyConfig.from_dict(policy)
    save(
        args.output / "calibration_fit.json",
        {
            "policy": policy,
            "store_cost_source": args.store_cost_source,
            "store_cost_s_per_block": {
                "worker-completion": sum(e["submit_to_complete_s"] for e in stores)
                / sum(e["stored_blocks"] for e in stores),
                "host-service": (
                    sum(e["store_host_service_s"] for e in stores)
                    / sum(e["stored_blocks"] for e in stores)
                    if all(e.get("store_host_service_s") is not None for e in stores)
                    else None
                ),
            },
            "counts": {
                "restore": len(restore),
                "recompute": len(recompute),
                "store": len(stores),
            },
            "store_dma_s_per_block": sum(e.get("dma_time_s", 0) for e in stores)
            / sum(e["stored_blocks"] for e in stores),
            "restore_mean_s": sum(e["restore_wait_s"] for e in restore) / len(restore),
            "recompute_mean_s": sum(e["recompute_wait_s"] for e in recompute)
            / len(recompute),
            "model_scope": (
                "calibration only; frozen before evaluation; not lifecycle oracle"
            ),
        },
    )
    print("FROZEN COST PRIORS " + json.dumps(policy), flush=True)


def evaluation_config(arm, fit, probe_config):
    if arm == "no-offload":
        return None
    if arm == "all-block":
        return kv_config()
    policy = dict(
        fit["policy"], admission_enabled=arm != "promotion", guard_enabled=arm == "full"
    )
    if arm.startswith(("completion-", "host-")):
        source = "host-service" if arm.startswith("host-") else "worker-completion"
        cost = fit["store_cost_s_per_block"][source]
        if cost is None:
            raise ValueError("Missing calibrated store source")
        policy["store_s_per_block"] = cost
        if arm.endswith("-probe") and not arm.endswith("-no-probe"):
            policy.update(probe_config)
    return kv_config("always_restore" if arm == "admission" else "cost_aware", policy)


def evaluate(args):
    fit = read(args.output / "calibration_fit.json")
    manifest = read(args.output / "manifest.json")
    results = []
    for arm in manifest["evaluation_order"]:
        config = evaluation_config(arm, fit, manifest.get("cold_probe_config", {}))
        results.append(
            run_arm(
                args, "evaluation/" + arm, args.output / "evaluation.json", config, 0
            )
        )
        save(args.output / "evaluation_summary.json", results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--port", type=int, default=8238)
    parser.add_argument("--model", default="/public/model_zoo/Qwen3-0.6B")
    parser.add_argument(
        "--evaluation-suite",
        choices=["six-arm", "cost-probe-ablation"],
        default="six-arm",
    )
    parser.add_argument(
        "--store-cost-source",
        choices=["worker-completion", "host-service"],
        default="worker-completion",
    )
    parser.add_argument(
        "--phase", choices=["all", "prepare", "calibrate", "evaluate"], default="all"
    )
    args = parser.parse_args()
    args.output = args.output.resolve()
    for name, function in [
        ("prepare", prepare),
        ("calibrate", calibrate),
        ("evaluate", evaluate),
    ]:
        if args.phase in ("all", name):
            function(args)


if __name__ == "__main__":
    main()
