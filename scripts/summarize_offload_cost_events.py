#!/usr/bin/env python3
"""Summarize opt-in hierarchical KV offload cost events.

The input is a ``value_policy_stats.json`` file written by the multi-turn
benchmark, or a JSON object containing the stats fields directly. The summary
uses only event-local block IDs; it does not require token content or block
hashes.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
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
        "max": max(values) if values else None,
    }


def numeric(events: list[dict[str, Any]], field: str) -> list[float]:
    return [float(event[field]) for event in events if event.get(field) is not None]


def load_stats(path: Path) -> tuple[str | None, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    stats = payload.get("stats", payload)
    if not isinstance(stats, dict):
        raise ValueError(f"{path}: expected a stats object")
    return payload.get("source"), stats


def summarize(path: Path) -> dict[str, Any]:
    source, stats = load_stats(path)
    events = stats.get("offload_cost_events", [])
    if not isinstance(events, list) or not all(
        isinstance(event, dict) for event in events
    ):
        raise ValueError(f"{path}: offload_cost_events must be a list of objects")

    promotion_events = [
        event for event in events if event.get("event_type") == "promotion"
    ]
    restore_events = [
        event for event in promotion_events if event.get("action") == "restore"
    ]
    recompute_events = [
        event for event in promotion_events if event.get("action") == "recompute"
    ]
    store_events = [event for event in events if event.get("event_type") == "store"]
    store_transfer_events = [
        event for event in store_events if event.get("action") == "store"
    ]

    stored_block_ids = {
        int(block_id)
        for event in store_events
        for block_id in event.get("stored_block_ids", [])
    }
    restored_block_ids = {
        int(block_id)
        for event in restore_events
        for block_id in event.get("block_ids", [])
    }

    restore_missing_dma = sum("dma_time_s" not in event for event in restore_events)
    restore_missing_wait = sum(
        "restore_wait_s" not in event for event in restore_events
    )
    recompute_missing_wait = sum(
        "recompute_wait_s" not in event for event in recompute_events
    )
    recompute_incomplete = sum(
        bool(event.get("recompute_incomplete")) for event in recompute_events
    )
    store_missing_dma = sum(
        "dma_time_s" not in event for event in store_transfer_events
    )
    dropped = int(stats.get("offload_cost_events_dropped", 0))
    unmatched = int(stats.get("offload_cost_event_unmatched_transfers", 0))
    promotion_oracle_blocker = None
    if not restore_events or not recompute_events:
        promotion_oracle_blocker = "requires paired restore and recompute cost traces"
    elif dropped:
        promotion_oracle_blocker = "cost events were dropped"
    elif unmatched:
        promotion_oracle_blocker = "transfer updates were unmatched"
    elif restore_missing_dma or restore_missing_wait:
        promotion_oracle_blocker = "restore events have incomplete cost fields"
    elif recompute_missing_wait or recompute_incomplete:
        promotion_oracle_blocker = "recompute events have incomplete cost fields"

    return {
        "input": str(path),
        "source": source,
        "enabled": bool(stats.get("offload_cost_events_enabled", False)),
        "events": {
            "retained": len(events),
            "dropped": dropped,
            "unmatched_transfer_updates": unmatched,
            "by_type": dict(Counter(str(event.get("event_type")) for event in events)),
            "promotion_by_action": dict(
                Counter(str(event.get("action")) for event in promotion_events)
            ),
            "store_by_action": dict(
                Counter(str(event.get("action")) for event in store_events)
            ),
        },
        "blocks": {
            "unique_stored": len(stored_block_ids),
            "unique_restored": len(restored_block_ids),
            "unique_restored_that_appear_in_retained_store_events": len(
                restored_block_ids & stored_block_ids
            ),
            "unique_stored_not_restored_in_retained_window": len(
                stored_block_ids - restored_block_ids
            ),
        },
        "timing_s": {
            "promotion_lookup_wall": describe(
                numeric(promotion_events, "lookup_wall_time_s")
            ),
            "restore_dma": describe(numeric(restore_events, "dma_time_s")),
            "restore_submit_to_complete": describe(
                numeric(restore_events, "submit_to_complete_s")
            ),
            "restore_wait": describe(numeric(restore_events, "restore_wait_s")),
            "recompute_wait": describe(numeric(recompute_events, "recompute_wait_s")),
            "store_prepare_wall": describe(
                numeric(store_events, "prepare_store_wall_time_s")
            ),
            "store_dma": describe(numeric(store_transfer_events, "dma_time_s")),
            "store_submit_to_complete": describe(
                numeric(store_transfer_events, "submit_to_complete_s")
            ),
        },
        "bytes": {
            "restore_dma_total": sum(
                int(event.get("dma_bytes", 0)) for event in restore_events
            ),
            "store_dma_total": sum(
                int(event.get("dma_bytes", 0)) for event in store_transfer_events
            ),
        },
        "completeness": {
            "restore_events": len(restore_events),
            "recompute_events": len(recompute_events),
            "restore_missing_dma": restore_missing_dma,
            "restore_missing_wait": restore_missing_wait,
            "recompute_missing_wait": recompute_missing_wait,
            "recompute_incomplete": recompute_incomplete,
            "store_transfer_missing_dma": store_missing_dma,
            "restore_cost_trace_ready": (
                bool(restore_events)
                and dropped == 0
                and unmatched == 0
                and restore_missing_dma == 0
                and restore_missing_wait == 0
            ),
            "recompute_cost_trace_ready": (
                bool(recompute_events)
                and dropped == 0
                and recompute_missing_wait == 0
                and recompute_incomplete == 0
            ),
            # One run uses one promotion policy, so an empirical Oracle normally
            # combines paired restore and recompute runs over the same trace.
            "promotion_oracle_ready": (
                bool(restore_events)
                and bool(recompute_events)
                and dropped == 0
                and unmatched == 0
                and restore_missing_dma == 0
                and restore_missing_wait == 0
                and recompute_missing_wait == 0
                and recompute_incomplete == 0
            ),
            "promotion_oracle_blocker": promotion_oracle_blocker,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = [summarize(path) for path in args.inputs]
    output: dict[str, Any] | list[dict[str, Any]]
    output = summaries[0] if len(summaries) == 1 else summaries
    rendered = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
