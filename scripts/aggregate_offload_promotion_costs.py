#!/usr/bin/env python3
"""Aggregate paired restore/recompute cost summaries across dataset seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any


def describe(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": mean(values) if values else None,
        "sample_stdev": stdev(values) if len(values) > 1 else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def parse_pair(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    return label, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        action="append",
        required=True,
        type=parse_pair,
        metavar="LABEL=PATH",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.pair) < 2:
        raise ValueError("at least two paired summaries are required")

    per_seed: dict[str, dict[str, Any]] = {}
    pooled_events: list[dict[str, Any]] = []
    for label, path in args.pair:
        if label in per_seed:
            raise ValueError(f"duplicate pair label: {label}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        events = payload.get("paired_events", [])
        if not isinstance(events, list) or not events:
            raise ValueError(f"{path}: paired_events must be a non-empty list")
        pairing = payload["pairing"]
        costs = payload["paired_cost_s"]
        counterfactual = payload["counterfactual_strategy_cost_s"]
        per_seed[label] = {
            "input": str(path),
            "restore_events": pairing["restore_events"],
            "recompute_events": pairing["recompute_events"],
            "paired_events": pairing["paired_events"],
            "unmatched_restore_events": pairing["unmatched_restore_events"],
            "unmatched_recompute_events": pairing["unmatched_recompute_events"],
            "restore_wait_mean_s": costs["restore_wait"]["mean"],
            "recompute_wait_mean_s": costs["recompute_wait"]["mean"],
            "recompute_minus_restore_mean_s": costs["recompute_minus_restore"]["mean"],
            "restore_faster_count": payload["decision"]["restore_faster_count"],
            "recompute_faster_count": payload["decision"]["recompute_faster_count"],
            "oracle_mean_s": counterfactual["per_event_oracle"]["mean"],
            "oracle_relative_saving_vs_always_restore": counterfactual[
                "per_event_oracle"
            ]["relative_saving_vs_always_restore"],
            "best_block_threshold": counterfactual["best_candidate_block_threshold"][
                "threshold"
            ],
        }
        pooled_events.extend({"seed": label, **event} for event in events)

    restore_waits = [float(event["restore_wait_s"]) for event in pooled_events]
    recompute_waits = [float(event["recompute_wait_s"]) for event in pooled_events]
    always_restore_total = sum(restore_waits)
    always_recompute_total = sum(recompute_waits)
    oracle_total = sum(
        min(restore, recompute)
        for restore, recompute in zip(restore_waits, recompute_waits, strict=True)
    )
    max_blocks = max(int(event["candidate_blocks"]) for event in pooled_events)
    threshold_costs = {
        threshold: sum(
            float(event["recompute_wait_s"])
            if int(event["candidate_blocks"]) <= threshold
            else float(event["restore_wait_s"])
            for event in pooled_events
        )
        for threshold in range(max_blocks + 1)
    }
    best_threshold = min(threshold_costs, key=threshold_costs.get)
    best_threshold_total = threshold_costs[best_threshold]
    event_count = len(pooled_events)

    seed_values = list(per_seed.values())
    summary = {
        "inputs": {label: data["input"] for label, data in per_seed.items()},
        "per_seed": per_seed,
        "macro_across_seed_means": {
            "restore_wait_s": describe(
                [float(data["restore_wait_mean_s"]) for data in seed_values]
            ),
            "recompute_wait_s": describe(
                [float(data["recompute_wait_mean_s"]) for data in seed_values]
            ),
            "recompute_minus_restore_s": describe(
                [float(data["recompute_minus_restore_mean_s"]) for data in seed_values]
            ),
            "oracle_relative_saving_vs_always_restore": describe(
                [
                    float(data["oracle_relative_saving_vs_always_restore"])
                    for data in seed_values
                ]
            ),
        },
        "pooled_micro": {
            "paired_events": event_count,
            "restore_events": sum(data["restore_events"] for data in seed_values),
            "recompute_events": sum(data["recompute_events"] for data in seed_values),
            "unmatched_restore_events": sum(
                data["unmatched_restore_events"] for data in seed_values
            ),
            "unmatched_recompute_events": sum(
                data["unmatched_recompute_events"] for data in seed_values
            ),
            "restore_faster_count": sum(
                data["restore_faster_count"] for data in seed_values
            ),
            "recompute_faster_count": sum(
                data["recompute_faster_count"] for data in seed_values
            ),
            "always_restore_mean_s": always_restore_total / event_count,
            "always_recompute_mean_s": always_recompute_total / event_count,
            "recompute_minus_restore_mean_s": (
                always_recompute_total - always_restore_total
            )
            / event_count,
            "per_event_oracle_mean_s": oracle_total / event_count,
            "oracle_mean_saving_vs_always_restore": (
                always_restore_total - oracle_total
            )
            / event_count,
            "oracle_relative_saving_vs_always_restore": (
                always_restore_total - oracle_total
            )
            / always_restore_total,
            "best_candidate_block_threshold": {
                "rule": "recompute when candidate_blocks <= threshold",
                "threshold": best_threshold,
                "mean_s": best_threshold_total / event_count,
                "mean_saving_vs_always_restore": (
                    always_restore_total - best_threshold_total
                )
                / event_count,
            },
        },
        "limitations": [
            "Per-seed Oracle and block threshold are fitted in-sample.",
            "Only strictly matched events are included in cost comparisons.",
            "Restore and recompute observations come from separate online runs.",
        ],
    }

    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
