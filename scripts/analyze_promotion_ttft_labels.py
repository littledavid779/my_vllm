# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Observed-action labels only. Never manufacture a counterfactual winner."""

import math
from collections import Counter, defaultdict

from analyze_lifecycle_request_path import validate_path


def build_labels(rows, stats):
    paths = {e["request_id"]: e for e in validate_path(rows, stats).values()}
    if stats.get("promotion_ttft_trace_enabled") != 1:
        raise ValueError("Promotion TTFT trace disabled")
    events = stats.get("promotion_ttft_trace_events", [])
    if stats.get("promotion_ttft_trace_dropped", 0):
        raise ValueError("Dropped decision records")
    if len(events) != stats.get("promotion_ttft_trace_recorded") or len(
        events
    ) != stats.get("offload_promotion_decision_count"):
        raise ValueError("Decision count mismatch")
    if len({e["event_id"] for e in events}) != len(events):
        raise ValueError("Duplicate decision IDs")
    decisions = defaultdict(list)
    for e in events:
        if e["request_id"] not in paths:
            raise ValueError("Decision has no completed request record")
        if e["action"] not in ("restore", "recompute"):
            raise ValueError("Unknown decision action")
        decisions[e["request_id"]].append(e)
    labels, excluded = [], []
    for rid, es in decisions.items():
        p = paths[rid]
        reason = None
        if len(es) != 1:
            reason = "multiple_decisions"
        elif p["preemptions"]:
            reason = "preempted"
        if reason:
            excluded.append(dict(request_id=rid, reason=reason, decisions=len(es)))
            continue
        e = es[0]
        start = e["decision_recorded_s"]
        if not math.isfinite(start) or not (
            p["enqueued_s"] <= start <= p["first_output_s"]
        ):
            raise ValueError("Invalid decision/first-token timestamp")
        fields = (
            "candidate_blocks",
            "candidate_tokens",
            "gpu_hit_tokens",
            "prompt_tokens",
            "tracked_requests",
            "pending_load_requests",
            "pending_store_requests",
            "pending_load_blocks",
            "pending_store_blocks",
            "bytes_per_offloaded_block",
            "candidate_bytes",
            "pending_load_bytes",
            "pending_store_bytes",
        )
        if any(type(e.get(f)) is not int or e[f] < 0 for f in fields):
            raise ValueError("Invalid decision feature")
        if (
            min(
                e["candidate_blocks"],
                e["candidate_tokens"],
                e["bytes_per_offloaded_block"],
            )
            <= 0
        ):
            raise ValueError("Empty candidate or unknown block size")
        for b, n in (
            ("candidate_bytes", "candidate_blocks"),
            ("pending_load_bytes", "pending_load_blocks"),
            ("pending_store_bytes", "pending_store_blocks"),
        ):
            if e[b] != e[n] * e["bytes_per_offloaded_block"]:
                raise ValueError("Block/byte mismatch")
        if e["prompt_tokens"] != p["prompt_tokens"] or (
            e["gpu_hit_tokens"] != p["initial_gpu_hit_tokens"]
        ):
            raise ValueError("Decision/admission prefix mismatch")
        remaining = e["prompt_tokens"] - e["gpu_hit_tokens"]
        if e["candidate_tokens"] > remaining:
            raise ValueError("Candidate exceeds prompt")
        expected_external = e["candidate_tokens"] if e["action"] == "restore" else 0
        if p["initial_external_match_tokens"] != expected_external:
            raise ValueError("Decision did not match allocated action")
        labels.append(
            dict(
                **e,
                remaining_prefill_if_recompute=remaining,
                # Even a full-prefix hit must recompute the final prompt token.
                remaining_prefill_if_restore=max(1, remaining - e["candidate_tokens"]),
                first_output_s=p["first_output_s"],
                decision_to_first_token_ms=1000 * (p["first_output_s"] - start),
                label_endpoint="scheduler_first_output",
                label_kind="observed_action_only",
            )
        )
    return dict(
        labels=labels,
        excluded=excluded,
        decision_events=len(events),
        candidate_requests=len(decisions),
        label_count=len(labels),
        requests_without_candidate=len(paths) - len(decisions),
        by_action=dict(Counter(e["action"] for e in labels)),
        limitations=[
            "Cross-run matching does not restore identical runtime state.",
            "Pending bytes are scheduler-known outstanding work, "
            "not hardware queue depth.",
            "Clock starts when the decision is recorded, after action selection.",
        ],
    )
