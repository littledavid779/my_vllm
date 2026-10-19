# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit and summarize a frozen full-conversation decision ablation (read-only)."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from run_lifecycle_calibrated_smoke import ROOT, digest, read
from run_lifecycle_fixed_workload import audit_arm, expected_requests, suite_config


def contrasts(rows, field="mean_ttft_ms"):
    base = rows["all-block-control"][field]
    admission = rows["admission-only"][field]
    promotion = rows["promotion-only"][field]
    joint = rows["joint-no-probe"][field]
    return {
        "controller_path": base - rows["all-block"][field],
        "admission_only": admission - base,
        "promotion_only": promotion - base,
        "joint": joint - base,
        "interaction": joint - admission - promotion + base,
        "admission_given_promotion": joint - promotion,
        "promotion_given_admission": joint - admission,
    }


def mean_sd(values):
    return {
        "mean": mean(values),
        "sample_sd": stdev(values) if len(values) > 1 else None,
    }


def summarize(root, allow_partial=False):
    manifest = read(root / "manifest.json")
    if manifest["suite"] != "decision-ablation":
        raise ValueError("Expected decision-ablation suite")
    if digest(root / "evaluation.json") != manifest["input_sha256"]:
        raise ValueError("Input hash mismatch")
    if digest(root / "calibration_fit.json") != manifest["fit_sha256"]:
        raise ValueError("Cost fit hash mismatch")
    expected = expected_requests(read(root / "evaluation.json"))
    fit = read(root / "calibration_fit.json")
    planned = {(r, arm) for r, arms in enumerate(manifest["orders"]) for arm in arms}
    rows = read(root / "evaluation_summary.json")
    seen, signatures = set(), None
    by_repeat, by_arm = defaultdict(dict), defaultdict(list)
    for row in rows:
        key = row["repeat"], row["arm"]
        if key in seen or key not in planned:
            raise ValueError(f"Unexpected/duplicate result: {key}")
        seen.add(key)
        armroot = root / f"repeat_{key[0]}" / key[1]
        spec = read(armroot / "spec.json")
        if spec["config"] != suite_config(
            "decision-ablation", key[1], fit, manifest["cold_probe_config"]
        ):
            raise ValueError(f"Configuration mismatch: {key}")
        if read(armroot / "gpu_samples.json")["foreign_pids"]:
            raise ValueError(f"Foreign GPU process: {key}")
        sig = audit_arm(row, expected, manifest["cold_probe_config"])
        if signatures is None:
            signatures = sig
        if sig != signatures:
            raise ValueError(f"Input token count mismatch: {key}")
        path = Path(row["stats_path"])
        stats = read(path)["stats"]
        for k in (
            "offload_cost_events_dropped",
            "offload_cost_event_unmatched_transfers",
            "lcuc_pending",
            "lcuc_incomplete_promotions",
            "lcuc_decision_events_dropped",
            "lcuc_cold_probe_blocks_spent",
            "lcuc_guard_transitions",
        ):
            if stats.get(k, 0):
                raise ValueError(f"Unexpected state {key}: {k}")
        with path.with_name("request_summary.csv").open() as stream:
            requests = list(csv.DictReader(stream))
        if len(requests) != manifest["requests_per_arm"] or any(
            float(r["output_num_tokens"]) <= 0 for r in requests
        ):
            raise ValueError(f"Incomplete requests: {key}")
        if not math.isclose(
            row["mean_ttft_ms"], mean(float(r["ttft_ms"]) for r in requests)
        ):
            raise ValueError(f"TTFT summary mismatch: {key}")
        by_repeat[key[0]][key[1]] = row
        by_arm[key[1]].append(row)
    if not allow_partial and seen != planned:
        raise ValueError(f"Missing arms: {sorted(planned - seen)}")
    fields = (
        "mean_ttft_ms",
        "p95_ttft_ms",
        "request_span_s",
        "requests_per_s",
        "store_blocks",
        "load_blocks",
        "d2h_bytes",
        "restore_decisions",
        "recompute_decisions",
        "decision_time_s",
        "estimated_net_utility_s",
    )
    effects = {
        repeat: contrasts(data)
        for repeat, data in by_repeat.items()
        if len(data) == len(manifest["orders"][repeat])
    }
    return {
        "complete": seen == planned,
        "audited_arms": len(seen),
        "audited_requests": len(seen) * sum(expected.values()),
        "missing": sorted(planned - seen),
        "current_code_matches_manifest": all(
            digest(ROOT / p) == sha for p, sha in manifest["code_sha256"].items()
        ),
        "aggregation_note": "Mean and sample SD across runs; P95 is mean of run P95s",
        "contrast_note": "TTFT ms difference; positive means slower; not event-paired",
        "arms": {
            arm: {
                field: mean_sd([r[field] for r in data])
                for field in fields
                if all(r.get(field) is not None for r in data)
            }
            for arm, data in by_arm.items()
        },
        "contrasts_per_repeat_ms": effects,
        "contrasts_aggregate_ms": {
            name: mean_sd([e[name] for e in effects.values()])
            for name in next(iter(effects.values()), {})
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    print(json.dumps(summarize(args.root, args.allow_partial), indent=2))
