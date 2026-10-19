#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit and summarize the admission-capacity suite (read-only)."""

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from run_lifecycle_fixed_workload import (
    CAPACITY_ARMS,
    ROOT,
    digest,
    expected_requests,
    read,
    suite_config,
)
from summarize_lifecycle_admission import summarize_stats


def describe(values):
    return {
        "values": values,
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
    }


def aggregate(rows):
    grouped = {}
    for arm in CAPACITY_ARMS:
        group = [r for r in rows if r["arm"] == arm]
        if group:
            grouped[arm] = {
                key: describe([r[key] for r in group])
                for key in (
                    "mean_ttft_ms",
                    "p95_ttft_ms",
                    "requests_per_s",
                    "store_blocks",
                    "load_blocks",
                )
            }
    contrasts = {}
    for capacity in (2, 4):
        for base in (f"all-block-{capacity}g", "no-offload"):
            differences = []
            for repeat in sorted({r["repeat"] for r in rows}):
                cells = {r["arm"]: r for r in rows if r["repeat"] == repeat}
                name = f"selective-{capacity}g"
                if name in cells and base in cells:
                    differences.append(
                        {
                            "repeat": repeat,
                            "p95_ttft_delta_ms": cells[name]["p95_ttft_ms"]
                            - cells[base]["p95_ttft_ms"],
                            "mean_ttft_delta_ms": cells[name]["mean_ttft_ms"]
                            - cells[base]["mean_ttft_ms"],
                            "throughput_change_pct": 100
                            * (
                                cells[name]["requests_per_s"]
                                / cells[base]["requests_per_s"]
                                - 1
                            ),
                        }
                    )
            contrasts[f"selective-{capacity}g_minus_{base}"] = differences
    return {"by_arm": grouped, "paired_run_contrasts": contrasts}


def audit(root, allow_partial=False):
    manifest = read(root / "manifest.json")
    if manifest["suite"] != "admission-capacity":
        raise ValueError("Not an admission-capacity suite")
    source = read(root / "evaluation.json")
    fit = read(root / "calibration_fit.json")
    if digest(root / "evaluation.json") != manifest["input_sha256"]:
        raise ValueError("Input digest mismatch")
    if digest(root / "calibration_fit.json") != manifest["fit_sha256"]:
        raise ValueError("Calibration digest mismatch")
    for path, sha in manifest["code_sha256"].items():
        if digest(ROOT / path) != sha:
            raise ValueError(f"Code differs from frozen manifest: {path}")
    expected = expected_requests(source)
    planned = {(i, arm) for i, order in enumerate(manifest["orders"]) for arm in order}
    rows = read(root / "evaluation_summary.json")
    actual = [(r["repeat"], r["arm"]) for r in rows]
    if len(set(actual)) != len(actual) or not set(actual) <= planned:
        raise ValueError("Duplicate or unplanned result")
    if not allow_partial and set(actual) != planned:
        raise ValueError("Incomplete suite; use --allow-partial for progress only")
    signature = None
    diagnostic = []
    gpu_samples = 0
    for r in rows:
        spec = read(root / r["name"] / "spec.json")
        cfg = suite_config(
            "admission-capacity", r["arm"], fit, manifest["cold_probe_config"]
        )
        if spec["config"] != cfg or spec["gpu"] != manifest["gpu"]:
            raise ValueError("Configuration mismatch")
        path = Path(r["stats_path"])
        requests = list(csv.DictReader(path.with_name("request_summary.csv").open()))
        identities = Counter(
            (x["conversation_id"], x["input_num_turns"], x["client_id"])
            for x in requests
        )
        if identities != expected:
            raise ValueError("Request identity mismatch")
        sig = sorted(
            (*k, x["input_num_tokens"])
            for x in requests
            for k in [(x["conversation_id"], x["input_num_turns"], x["client_id"])]
        )
        if signature is None:
            signature = sig
        elif signature != sig:
            raise ValueError("Request token signature mismatch")
        times = sorted(float(x["ttft_ms"]) for x in requests)
        pos = (len(times) - 1) * 0.95
        lo = int(pos)
        p95 = times[lo] + (times[min(lo + 1, len(times) - 1)] - times[lo]) * (pos - lo)
        for key, value in (
            ("p95_ttft_ms", p95),
            ("mean_ttft_ms", statistics.mean(times)),
        ):
            if not math.isclose(r[key], value, abs_tol=1e-8):
                raise ValueError(f"Metric mismatch: {key}")
        s = read(path)["stats"]
        for key in (
            "offload_store_failed_count",
            "offload_store_completion_failed_blocks",
            "offload_cost_events_dropped",
            "offload_cost_event_block_id_evictions",
            "offload_cost_event_unmatched_transfers",
            "lcuc_decision_events_dropped",
            "lcuc_pending",
            "lcuc_incomplete_promotions",
            "lcuc_guard_transitions",
            "lcuc_cold_probe_blocks_spent",
            "lcuc_recompute_decisions",
        ):
            if s.get(key, 0):
                raise ValueError(f"Invalid observations: {r['name']} {key}")
        samples = read(root / r["name"] / "gpu_samples.json")
        if samples["foreign_pids"]:
            raise ValueError("Foreign GPU process detected")
        gpu_samples += len(samples["samples"])
        if cfg is not None:
            d = summarize_stats(s)
            waits = [
                e["restore_wait_s"]
                for e in s["offload_cost_events"]
                if e.get("event_type") == "promotion" and "restore_wait_s" in e
            ]
            diagnostic.append(
                {
                    "repeat": r["repeat"],
                    "arm": r["arm"],
                    "cpu_evicted_blocks": s["offload_store_evicted_blocks"],
                    "unique_evicted_blocks_with_later_missing_suffix_demand": d[
                        "unique_evicted_blocks_with_later_missing_suffix_demand"
                    ],
                    "first_missing_gaps_with_prior_cpu_eviction": d[
                        "first_missing_gaps_with_prior_cpu_eviction"
                    ],
                    "unique_rejected_blocks_with_later_suffix_demand": d[
                        "unique_rejected_blocks_with_later_suffix_demand"
                    ],
                    "restore_wait_observations": len(waits),
                    "mean_restore_wait_ms": statistics.mean(waits) * 1000
                    if waits
                    else None,
                }
            )
    return {
        "complete": set(actual) == planned,
        "runs": len(rows),
        "requests": sum(r["requests"] for r in rows),
        "gpu_samples": gpu_samples,
        "audit_passed": True,
        **aggregate(rows),
        "diagnostic": diagnostic,
        "limitations": [
            "Same inspected workload/seed; repeated runs are not unseen validation.",
            "P95 summary averages run P95s; SD is not a confidence interval.",
            "Contrasts compare runs, not identical-state request counterfactuals.",
            "Eviction/rejection demand is association, not measured saved time.",
            "Detailed tracing is enabled; only target GPU isolation was monitored.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit(args.root, args.allow_partial), indent=2))


if __name__ == "__main__":
    main()
