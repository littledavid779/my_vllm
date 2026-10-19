#!/usr/bin/env python3
"""Compare paired restore and recompute promotion cost events.

This script is intended for two runs over the same multi-turn trace. The
benchmark passes a stable conversation ID as the ``X-Request-Id`` prefix, while
the server appends a per-run suffix. Events are therefore paired by stable
conversation prefix, candidate block count, local-prefix length, and occurrence
order. Unpaired events are reported instead of silently discarded.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def describe(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": sum(values) / len(values) if values else None,
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p99": percentile(values, 0.99),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def load_events(path: Path, action: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    stats = payload.get("stats", payload)
    events = stats.get("offload_cost_events", [])
    if not isinstance(events, list):
        raise ValueError(f"{path}: offload_cost_events must be a list")
    return [
        event
        for event in events
        if isinstance(event, dict)
        and event.get("event_type") == "promotion"
        and event.get("action") == action
    ]


def stable_request_key(request_id: str) -> str:
    prefix = request_id.removeprefix("chatcmpl-")
    stable_prefix, separator, _ = prefix.rpartition("-")
    return stable_prefix if separator else prefix


def event_signature(event: dict[str, Any]) -> tuple[str, int, int]:
    return (
        stable_request_key(str(event["request_id"])),
        int(event["candidate_blocks"]),
        int(event["num_computed_tokens"]),
    )


def block_bucket(blocks: int) -> str:
    if blocks <= 8:
        return "1-8"
    if blocks <= 16:
        return "9-16"
    if blocks <= 32:
        return "17-32"
    if blocks <= 64:
        return "33-64"
    return "65+"


def summarize_pairs(
    restore_events: list[dict[str, Any]],
    recompute_events: list[dict[str, Any]],
) -> dict[str, Any]:
    restore_by_signature: dict[tuple[str, int, int], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    recompute_by_signature: dict[tuple[str, int, int], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for event in restore_events:
        restore_by_signature[event_signature(event)].append(event)
    for event in recompute_events:
        recompute_by_signature[event_signature(event)].append(event)

    paired_events: list[dict[str, Any]] = []
    unmatched_restore: list[dict[str, Any]] = []
    unmatched_recompute: list[dict[str, Any]] = []
    signatures = set(restore_by_signature) | set(recompute_by_signature)
    for signature in sorted(signatures):
        restores = restore_by_signature.get(signature, [])
        recomputes = recompute_by_signature.get(signature, [])
        pair_count = min(len(restores), len(recomputes))
        for occurrence, (restore, recompute) in enumerate(
            zip(restores[:pair_count], recomputes[:pair_count], strict=True)
        ):
            if restore.get("restore_wait_s") is None:
                raise ValueError(f"restore event lacks restore_wait_s: {restore}")
            if recompute.get("recompute_wait_s") is None:
                raise ValueError(f"recompute event lacks recompute_wait_s: {recompute}")
            restore_wait = float(restore["restore_wait_s"])
            recompute_wait = float(recompute["recompute_wait_s"])
            delta = recompute_wait - restore_wait
            conversation_id, candidate_blocks, num_computed_tokens = signature
            paired_events.append(
                {
                    "conversation_id": conversation_id,
                    "signature_occurrence": occurrence,
                    "candidate_blocks": candidate_blocks,
                    "candidate_tokens": int(restore["candidate_tokens"]),
                    "num_computed_tokens": num_computed_tokens,
                    "restore_wait_s": restore_wait,
                    "restore_dma_time_s": restore.get("dma_time_s"),
                    "restore_submit_to_complete_s": restore.get("submit_to_complete_s"),
                    "recompute_wait_s": recompute_wait,
                    "recompute_minus_restore_s": delta,
                    "faster_action": "restore" if delta > 0 else "recompute",
                }
            )
        unmatched_restore.extend(restores[pair_count:])
        unmatched_recompute.extend(recomputes[pair_count:])

    restore_waits = [event["restore_wait_s"] for event in paired_events]
    recompute_waits = [event["recompute_wait_s"] for event in paired_events]
    deltas = [event["recompute_minus_restore_s"] for event in paired_events]
    restore_wins = sum(delta > 0 for delta in deltas)
    recompute_wins = sum(delta < 0 for delta in deltas)

    always_restore_total = sum(restore_waits)
    always_recompute_total = sum(recompute_waits)
    oracle_total = sum(
        min(event["restore_wait_s"], event["recompute_wait_s"])
        for event in paired_events
    )
    best_threshold: int | None = None
    best_threshold_total: float | None = None
    if paired_events:
        max_candidate_blocks = max(event["candidate_blocks"] for event in paired_events)
        for threshold in range(max_candidate_blocks + 1):
            total = sum(
                event["recompute_wait_s"]
                if event["candidate_blocks"] <= threshold
                else event["restore_wait_s"]
                for event in paired_events
            )
            if best_threshold_total is None or total < best_threshold_total:
                best_threshold = threshold
                best_threshold_total = total

    by_block_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in paired_events:
        by_block_bucket[block_bucket(event["candidate_blocks"])].append(event)

    return {
        "pairing": {
            "key": (
                "conversation prefix + candidate_blocks + "
                "num_computed_tokens + occurrence"
            ),
            "restore_events": len(restore_events),
            "recompute_events": len(recompute_events),
            "paired_events": len(paired_events),
            "unmatched_restore_events": len(unmatched_restore),
            "unmatched_recompute_events": len(unmatched_recompute),
            "unmatched_restore_signatures": [
                event_signature(event) for event in unmatched_restore
            ],
            "unmatched_recompute_signatures": [
                event_signature(event) for event in unmatched_recompute
            ],
        },
        "paired_cost_s": {
            "restore_wait": describe(restore_waits),
            "recompute_wait": describe(recompute_waits),
            "recompute_minus_restore": describe(deltas),
        },
        "decision": {
            "restore_faster_count": restore_wins,
            "recompute_faster_count": recompute_wins,
            "equal_count": len(deltas) - restore_wins - recompute_wins,
            "restore_faster_ratio": (restore_wins / len(deltas) if deltas else None),
            "recompute_faster_ratio": (
                recompute_wins / len(deltas) if deltas else None
            ),
        },
        "counterfactual_strategy_cost_s": {
            "note": (
                "Oracle and best block threshold are fitted in-sample and are "
                "upper bounds, not deployable-policy results."
            ),
            "always_restore": {
                "total": always_restore_total,
                "mean": (
                    always_restore_total / len(paired_events) if paired_events else None
                ),
            },
            "always_recompute": {
                "total": always_recompute_total,
                "mean": (
                    always_recompute_total / len(paired_events)
                    if paired_events
                    else None
                ),
            },
            "per_event_oracle": {
                "total": oracle_total,
                "mean": oracle_total / len(paired_events) if paired_events else None,
                "mean_saving_vs_always_restore": (
                    (always_restore_total - oracle_total) / len(paired_events)
                    if paired_events
                    else None
                ),
                "relative_saving_vs_always_restore": (
                    (always_restore_total - oracle_total) / always_restore_total
                    if always_restore_total
                    else None
                ),
                "mean_saving_vs_always_recompute": (
                    (always_recompute_total - oracle_total) / len(paired_events)
                    if paired_events
                    else None
                ),
            },
            "best_candidate_block_threshold": {
                "rule": "recompute when candidate_blocks <= threshold",
                "threshold": best_threshold,
                "total": best_threshold_total,
                "mean": (
                    best_threshold_total / len(paired_events)
                    if best_threshold_total is not None and paired_events
                    else None
                ),
                "mean_saving_vs_always_restore": (
                    (always_restore_total - best_threshold_total) / len(paired_events)
                    if best_threshold_total is not None and paired_events
                    else None
                ),
            },
        },
        "by_candidate_block_bucket": {
            bucket: {
                "count": len(events),
                "restore_wait_s": describe(
                    [event["restore_wait_s"] for event in events]
                ),
                "recompute_wait_s": describe(
                    [event["recompute_wait_s"] for event in events]
                ),
                "recompute_minus_restore_s": describe(
                    [event["recompute_minus_restore_s"] for event in events]
                ),
                "restore_faster_count": sum(
                    event["recompute_minus_restore_s"] > 0 for event in events
                ),
                "recompute_faster_count": sum(
                    event["recompute_minus_restore_s"] < 0 for event in events
                ),
            }
            for bucket, events in sorted(by_block_bucket.items())
        },
        "paired_events": paired_events,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore", required=True, type=Path)
    parser.add_argument("--recompute", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    restore_events = load_events(args.restore, "restore")
    recompute_events = load_events(args.recompute, "recompute")
    summary = {
        "restore_input": str(args.restore),
        "recompute_input": str(args.recompute),
        **summarize_pairs(restore_events, recompute_events),
    }
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
