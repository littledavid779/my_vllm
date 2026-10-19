#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Six balanced orders, three fixed 2-GiB arms, detailed tracing disabled."""

import argparse
import itertools
import random
import statistics
from pathlib import Path

from run_lifecycle_fixed_workload import (
    CODE,
    ROOT,
    audit_arm,
    digest,
    expected_requests,
    read,
    run_arm,
    save,
    suite_config,
)

ARMS = ("no-offload", "all-block-2g", "selective-2g")


def balanced_orders():
    orders = [list(p) for p in itertools.permutations(ARMS)]
    random.Random(20261003).shuffle(orders)
    return orders


def config(arm, fit, probe):
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    cfg = suite_config("admission-capacity", arm, fit, probe)
    if cfg is not None:
        extra = cfg["kv_connector_extra_config"]
        extra["track_cost_events"] = False
        extra["lifecycle_policy"]["trace_admission"] = False
    return cfg


def summarize(rows):
    by_arm = {}
    for arm in ARMS:
        group = [r for r in rows if r["arm"] == arm]
        if not group:
            continue
        by_arm[arm] = {}
        for key in (
            "p95_ttft_ms",
            "mean_ttft_ms",
            "requests_per_s",
            "store_blocks",
            "load_blocks",
        ):
            values = [r[key] for r in group]
            by_arm[arm][key] = dict(
                values=values,
                mean=statistics.mean(values),
                sample_sd=statistics.stdev(values) if len(values) > 1 else None,
            )
    differences = []
    for repeat in range(6):
        group = {r["arm"]: r for r in rows if r["repeat"] == repeat}
        if set(group) != set(ARMS):
            continue
        for base in ARMS[:2]:
            selected = group["selective-2g"]
            reference = group[base]
            differences.append(
                dict(
                    repeat=repeat,
                    reference=base,
                    selective_position=selected["position"],
                    reference_position=reference["position"],
                    p95_delta_ms=selected["p95_ttft_ms"] - reference["p95_ttft_ms"],
                    mean_delta_ms=selected["mean_ttft_ms"] - reference["mean_ttft_ms"],
                    throughput_change_pct=100
                    * (selected["requests_per_s"] / reference["requests_per_s"] - 1),
                )
            )
    return dict(
        complete=len(rows) == 18,
        runs=len(rows),
        requests=sum(r["requests"] for r in rows),
        by_arm=by_arm,
        paired_run_differences=differences,
    )


def main(args):
    source_manifest = read(args.source / "manifest.json")
    if args.model != source_manifest["model"]:
        raise ValueError("Model differs from frozen calibration")
    fit = read(args.source / "calibration_fit.json")
    data = read(args.source / "evaluation.json")
    probe = source_manifest["cold_probe_config"]
    expected = expected_requests(data)
    code = CODE + ["scripts/run_lifecycle_ttft_confirmation.py"]
    manifest = dict(
        suite="ttft-confirmation-2g",
        orders=balanced_orders(),
        gpu=args.gpu,
        port=args.port,
        model=args.model,
        runtime_seed=0,
        requests_per_arm=sum(expected.values()),
        primary_metric="p95_ttft_ms",
        source=str(args.source.resolve()),
        input_sha256=digest(args.source / "evaluation.json"),
        fit_sha256=digest(args.source / "calibration_fit.json"),
        code_sha256={p: digest(ROOT / p) for p in code},
        configurations={arm: config(arm, fit, probe) for arm in ARMS},
        limitations=[
            "Same inspected workload and runtime seed; not unseen validation.",
            "Every arm occurs twice per position; "
            "each pair has both orders three times.",
            "Balanced order reduces but cannot eliminate shared-host time variation.",
            "Cost/block traces disabled; "
            "bounded controller logs and aggregate counters remain.",
            "DMA byte/time measurements unavailable, represented as null, not zero.",
            "Always restore, guard/cold probes off; no runtime algorithm changes.",
        ],
    )
    if (args.output / "manifest.json").exists():
        if read(args.output / "manifest.json") != manifest:
            raise ValueError("Resume code/configuration differs")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        save(args.output / "manifest.json", manifest)
        save(args.output / "evaluation.json", data)
        save(args.output / "calibration_fit.json", fit)
    rows = []
    signature = None
    for repeat, order in enumerate(manifest["orders"]):
        for position, arm in enumerate(order):
            for file, sha in manifest["code_sha256"].items():
                if digest(ROOT / file) != sha:
                    raise ValueError(f"Frozen code changed: {file}")
            for file, key in (
                ("evaluation.json", "input_sha256"),
                ("calibration_fit.json", "fit_sha256"),
            ):
                if digest(args.output / file) != manifest[key]:
                    raise ValueError(f"Frozen input changed: {file}")
            metrics = run_arm(
                args,
                f"repeat_{repeat}/{arm}",
                args.output / "evaluation.json",
                manifest["configurations"][arm],
                0,
                complete_workload=True,
            )
            sig = audit_arm(metrics, expected, probe)
            if signature is None:
                signature = sig
            elif signature != sig:
                raise ValueError("Input token signature differs")
            stats = read(Path(metrics["stats_path"]))["stats"]
            for key in (
                "offload_cost_events_enabled",
                "offload_admission_trace_enabled",
                "lcuc_recompute_decisions",
                "lcuc_guard_transitions",
                "lcuc_cold_probe_blocks_spent",
                "lcuc_decision_events_dropped",
            ):
                if stats.get(key, 0):
                    raise ValueError(f"Unexpected active trace/action: {key}")
            metrics.update(
                repeat=repeat,
                arm=arm,
                position=position,
                d2h_bytes=None,
                h2d_bytes=None,
                dma_measurement_available=False,
            )
            save(args.output / metrics["name"] / "metrics.json", metrics)
            rows.append(metrics)
            save(args.output / "evaluation_summary.json", rows)
            save(args.output / "analysis_summary.json", summarize(rows))
            print(f"AUDIT PASS {len(rows)}/18: {metrics['name']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--port", type=int, default=8238)
    parser.add_argument("--model", default="/public/model_zoo/Qwen3-0.6B")
    main(parser.parse_args())
