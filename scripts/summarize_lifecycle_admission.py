#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read-only admission/lookup associations, NOT a counterfactual speedup oracle."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def summarize_stats(stats: dict[str, Any]) -> dict[str, Any]:
    events = stats.get("offload_cost_events", [])
    if not stats.get("offload_admission_trace_enabled"):
        raise ValueError("No admission trace; old result files cannot infer rejections")
    # Recycling block IDs or losing events invalidates cross-event accounting.
    for key in ("offload_cost_events_dropped", "offload_cost_event_block_id_evictions"):
        if stats.get(key, 0):
            raise ValueError(f"Incomplete trace: {key}={stats[key]}")
    ids = [e["event_id"] for e in events]
    if ids != sorted(set(ids)):
        raise ValueError("event_id must be unique and increasing")

    counts: Counter = Counter()
    rejected: dict[int, tuple[str, int]] = {}
    ever_rejected: set[int] = set()
    demanded: set[int] = set()
    demand_pairs: set[tuple[str, int]] = set()
    gap_pairs: set[tuple[str, int, int]] = set()
    evicted: dict[int, int] = {}
    eviction_demand_pairs: set[tuple[str, int]] = set()
    eviction_gap_pairs: set[tuple[str, int, int]] = set()
    examples = []
    for e in events:
        kind = e.get("event_type")
        if kind == "store":
            for bid in e.get("evicted_block_ids", []):
                evicted[bid] = e["event_id"]
            for bid in e.get("stored_block_ids", []):
                evicted.pop(bid, None)
            continue
        if kind not in {"admission", "external_lookup"}:
            continue
        blocks = e["candidate_block_ids"]
        states = e["candidate_states"]
        if len(blocks) != len(states):
            raise ValueError("candidate IDs and states have different lengths")
        if any(s not in {"ready", "pending", "missing", "unknown"} for s in states):
            raise ValueError("Invalid block residency")
        request = e["request_id"]
        if kind == "admission":
            selected = e["selected"]
            if not 0 <= selected <= len(blocks):
                raise ValueError("Invalid selected prefix length")
            counts["admission_batches"] += 1
            counts["allocation_failures"] += int(e["allocation_failed"])
            counts["candidate_blocks"] += len(blocks)
            counts["candidate_already_allocated_blocks"] += sum(
                s in {"ready", "pending"} for s in states
            )
            counts["candidate_unknown_blocks"] += states.count("unknown")
            for bid, state in zip(blocks[selected:], states[selected:]):
                # Not selecting an existing CPU block does not delete it!
                if state == "missing":
                    rejected[bid] = (request, e["event_id"])
                    ever_rejected.add(bid)
                    counts["missing_block_rejection_occurrences"] += 1
            for bid in e["stored_block_ids"]:
                rejected.pop(bid, None)
            continue

        counts["lookup_attempts"] += 1
        counts["lookup_unknown_blocks"] += states.count("unknown")
        # Distinct later requests only; retries never multiply demand counts.
        for bid in blocks:
            if bid in rejected and rejected[bid][0] != request:
                demanded.add(bid)
                demand_pairs.add((request, bid))
        for bid, state in zip(blocks, states):
            if bid in evicted and state == "missing":
                eviction_demand_pairs.add((request, bid))
        hit = e["hit_blocks"]
        if hit is None:
            counts["lookup_retries"] += 1
            continue
        if not 0 <= hit <= len(blocks):
            raise ValueError("Invalid lookup prefix length")
        if hit == len(blocks):
            continue
        bid, state = blocks[hit], states[hit]
        key = (request, e["start_block_idx"], bid)
        if state == "missing" and bid in evicted:
            eviction_gap_pairs.add(key)
        if key in gap_pairs:
            continue
        gap_pairs.add(key)
        counts[f"first_gap_{state}"] += 1
        ready_after = states[hit + 1 :].count("ready")
        counts["ready_suffix_blocks_behind_gap"] += ready_after
        # Snapshot only: what would be adjacent if THIS gap alone became ready?
        adjacent = 0
        for s in states[hit + 1 :]:
            if s != "ready":
                break
            adjacent += 1
        if state == "missing" and bid in rejected and rejected[bid][0] != request:
            counts["first_gap_with_prior_rejection"] += 1
            counts["ready_suffix_behind_prior_rejection"] += ready_after
            counts["adjacent_ready_after_rejected_gap"] += adjacent
            if len(examples) < 20:
                examples.append(
                    {
                        "request_id": request,
                        "lookup_event_id": e["event_id"],
                        "rejection_event_id": rejected[bid][1],
                        "block_id": bid,
                        "ready_suffix_blocks": ready_after,
                        "adjacent_ready_blocks": adjacent,
                    }
                )
    return {
        "counts": dict(counts),
        "unique_rejected_missing_blocks": len(ever_rejected),
        "unique_rejected_blocks_with_later_suffix_demand": len(demanded),
        "distinct_request_rejected_block_demand_pairs": len(demand_pairs),
        "unique_evicted_blocks_with_later_missing_suffix_demand": len(
            {bid for _, bid in eviction_demand_pairs}
        ),
        "distinct_request_evicted_block_demand_pairs": len(eviction_demand_pairs),
        "first_missing_gaps_with_prior_cpu_eviction": len(eviction_gap_pairs),
        "examples": examples,
        "semantics": (
            "Observed-window associations only. Suffix demand is not an actual "
            "CPU hit; ready suffix counts are per distinct request/start/gap, "
            "not unique blocks or measured saved time. Prior rejection does not "
            "prove OFFLOAD would have retained the block or improved TTFT."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stats", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.stats.read_text())
    print(json.dumps(summarize_stats(payload.get("stats", payload)), indent=2))


if __name__ == "__main__":
    main()
