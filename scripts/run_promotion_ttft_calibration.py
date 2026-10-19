#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Collect common-endpoint labels with fixed admission and frozen actions."""

import argparse
import csv
import random
from pathlib import Path

from analyze_promotion_restore_joint import join_restore_labels
from analyze_promotion_ttft_labels import build_labels
from run_lifecycle_fixed_workload import (
    CODE,
    ROOT,
    audit_arm,
    digest,
    expected_requests,
    read,
    run_arm,
    save,
)
from run_lifecycle_ttft_confirmation import config

ORDERS = (("restore", "recompute"), ("recompute", "restore"))


def calibration_config(action, fit, probe):
    if action not in ("restore", "recompute"):
        raise ValueError("Only forced actions are supported during label collection")
    cfg = config("all-block-2g", fit, probe)
    extra = cfg["kv_connector_extra_config"]
    extra["promotion_policy"] = "always_" + action
    extra["track_promotion_ttft"] = True
    assert not extra["lifecycle_policy"]["admission_enabled"]
    assert not extra["lifecycle_policy"]["guard_enabled"]
    return cfg


def mixed_config(data, fit, probe, seed):
    cfg = calibration_config("restore", fit, probe)
    extra = cfg["kv_connector_extra_config"]
    extra["promotion_policy"] = "calibration_mixed"
    rng = random.Random(seed)
    extra["calibration_action_assignments"] = {
        f"chatcmpl-{item['id']}-turn{n + 1}": (
            "restore" if rng.random() < 0.5 else "recompute"
        )
        for item in data
        for n, message in enumerate(item["messages"])
        if message["role"] == "user"
    }
    return cfg


def main(args):
    restore_diagnostic = getattr(args, "restore_diagnostic", False)
    if restore_diagnostic and not getattr(args, "mixed_seeds", None):
        raise ValueError("--restore-diagnostic requires --mixed-seeds")
    source = read(args.source / "manifest.json")
    if args.model != source["model"]:
        raise ValueError("Model differs from source calibration")
    data = read(args.source / "calibration.json")
    heldout = read(args.source / "evaluation.json")
    if {r["id"] for r in data} & {r["id"] for r in heldout}:
        raise ValueError("Calibration/evaluation conversation overlap")
    expected = expected_requests(data)
    fit = read(args.source / "calibration_fit.json")
    probe = source["cold_probe_config"]
    files = CODE + [
        "scripts/run_promotion_ttft_calibration.py",
        "scripts/analyze_promotion_ttft_labels.py",
        "scripts/analyze_lifecycle_request_path.py",
        "scripts/analyze_lifecycle_restore_path.py",
        "scripts/analyze_promotion_restore_joint.py",
        "scripts/run_lifecycle_ttft_confirmation.py",
        "vllm/v1/kv_offload/promotion_trace.py",
        "vllm/v1/kv_offload/promotion_calibration.py",
        "vllm/v1/kv_offload/restore_trace.py",
        "vllm/v1/core/sched/request_trace.py",
        "vllm/v1/core/sched/scheduler.py",
        "vllm/v1/engine/core.py",
        "vllm/v1/core/block_pool.py",
        "vllm/v1/engine/input_processor.py",
        "vllm/v1/executor/uniproc_executor.py",
        "vllm/v1/kv_offload/worker/cpu_gpu.py",
        "vllm/v1/kv_offload/worker/worker.py",
    ]
    mixed_seeds = getattr(args, "mixed_seeds", None)
    if mixed_seeds is not None and len(set(mixed_seeds)) != len(mixed_seeds):
        raise ValueError("Duplicate mixed seeds")
    if mixed_seeds is None:
        runs = [
            (f"repeat_{i}/{a}", i, pos, a, calibration_config(a, fit, probe))
            for i, order in enumerate(ORDERS)
            for pos, a in enumerate(order)
        ]
    else:
        runs = [
            (f"mixed_{seed}", i, 0, "mixed", mixed_config(data, fit, probe, seed))
            for i, seed in enumerate(mixed_seeds)
        ]
    manifest = dict(
        suite="promotion-common-endpoint-labels",
        orders=ORDERS if mixed_seeds is None else None,
        gpu=args.gpu,
        port=args.port,
        model=args.model,
        source=str(args.source.resolve()),
        requests_per_arm=sum(expected.values()),
        input_sha256=digest(args.source / "calibration.json"),
        heldout_sha256=digest(args.source / "evaluation.json"),
        fit_sha256=digest(args.source / "calibration_fit.json"),
        code_sha256={p: digest(ROOT / p) for p in files},
        configurations={a: calibration_config(a, fit, probe) for a in ORDERS[0]},
        mixed_assignment_seeds=mixed_seeds,
        restore_diagnostic=restore_diagnostic,
        run_configurations={name: cfg for name, _, _, _, cfg in runs},
        mixed_restore_assignment_probability=0.5 if mixed_seeds else None,
        limitations=[
            "Observed-action labels, not paired counterfactual winners.",
            "All-block rule and capacity fixed; "
            "realized cache state can differ by action.",
            "Existing calibration split, no heldout tuning or performance claim.",
            "Request-path and decision traces on; full block trace off.",
            "Restore-phase trace on only in explicit diagnostic mode; "
            "do not interpret cross-mode latency differences as speedups.",
        ],
    )
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "manifest.json", manifest)
    save(args.output / "calibration.json", data)
    save(args.output / "calibration_fit.json", fit)
    metrics, signature = [], None
    for name, repeat, position, action, cfg in runs:
        for p, h in manifest["code_sha256"].items():
            if digest(ROOT / p) != h:
                raise ValueError(f"Frozen code changed: {p}")
        for p, k in (
            ("calibration.json", "input_sha256"),
            ("calibration_fit.json", "fit_sha256"),
        ):
            if digest(args.output / p) != manifest[k]:
                raise ValueError(f"Frozen input changed: {p}")
        m = run_arm(
            args,
            name,
            args.output / "calibration.json",
            cfg,
            0,
            complete_workload=True,
            correlate_requests=True,
            request_path_trace=True,
            restore_path_trace=restore_diagnostic,
        )
        current = audit_arm(m, expected, probe)
        if signature is not None and current != signature:
            raise ValueError("Input signature differs")
        signature = current
        path = Path(m["stats_path"])
        stats = read(path)["stats"]
        for k in (
            "offload_cost_events_enabled",
            "offload_admission_trace_enabled",
            "lcuc_guard_transitions",
            "lcuc_cold_probe_blocks_spent",
        ):
            if stats.get(k, 0):
                raise ValueError(f"Unexpected action or instrumentation: {k}")
        rows = list(csv.DictReader(path.with_name("request_summary.csv").open()))
        labels = build_labels(rows, stats)
        if bool(stats.get("restore_path_trace_enabled", 0)) != restore_diagnostic:
            raise ValueError("Unexpected restore trace mode")
        if restore_diagnostic:
            save(
                args.output / name / "promotion_restore_joint.json",
                join_restore_labels(rows, stats),
            )
        expected_actions = {action} if action != "mixed" else {"restore", "recompute"}
        if not labels["label_count"] or set(labels["by_action"]) != expected_actions:
            raise ValueError("No valid labels or unexpected action")
        if action == "mixed":
            assignments = cfg["kv_connector_extra_config"][
                "calibration_action_assignments"
            ]
            for e in stats["promotion_ttft_trace_events"]:
                external = e["request_id"].rsplit("-", 1)[0]
                if assignments.get(external) != e["action"]:
                    raise ValueError("Actual action differs from frozen assignment")
        owner_map = {
            "chatcmpl-" + r["external_request_id"]: "chatcmpl-" + r["conversation_id"]
            for r in rows
        }
        if read(args.output / name / "spec.json")["request_owner_map"] != owner_map:
            raise ValueError("Conversation ownership mismatch")
        save(args.output / name / "promotion_ttft_labels.json", labels)
        m.update(
            action=action,
            repeat=repeat,
            position=position,
            label_count=labels["label_count"],
            excluded=labels["excluded"],
            d2h_bytes=None,
            h2d_bytes=None,
        )
        save(args.output / name / "metrics.json", m)
        metrics.append(m)
        save(args.output / "summary.json", metrics)
        print(
            f"LABELS PASS {len(metrics)}/{len(runs)}: {action} {labels['by_action']}",
            flush=True,
        )
    print("COMPLETE: labels only; no policy fit or speedup claim.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--port", type=int, default=8238)
    parser.add_argument("--mixed-seeds", type=int, nargs="+")
    parser.add_argument("--restore-diagnostic", action="store_true")
    parser.add_argument("--model", default="/public/model_zoo/Qwen3-0.6B")
    main(parser.parse_args())
