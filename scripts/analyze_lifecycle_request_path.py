# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate scheduler path records and attach descriptive cross-run differences."""

import math
import re
import statistics
from collections import defaultdict

FIELDS = (
    "queue_to_first_compute_ms",
    "first_compute_to_output_ms",
    "scheduler_ttft_ms",
    "initial_gpu_hit_tokens",
    "first_compute_cached_tokens",
    "completed_prefill_tokens",
)


def validate_path(rows, stats):
    if stats.get("request_path_trace_enabled") != 1:
        raise ValueError("Request path tracing is not enabled")
    for suffix in (
        "active",
        "pending_batches",
        "dropped",
        "unsupported_requests",
        "duplicate_requests",
        "invalid_first_output",
        "finished_without_first_token",
    ):
        if stats.get("request_path_trace_" + suffix, 0):
            raise ValueError(f"Incomplete request path trace: {suffix}")
    expected = {r["external_request_id"]: r for r in rows}
    if len(expected) != len(rows) or "" in expected:
        raise ValueError("Invalid CSV correlation IDs")
    events = stats.get("request_path_trace_events", [])
    if len(events) != len(rows):
        raise ValueError("Path event count mismatch")
    result = {}
    for e in events:
        match = re.fullmatch(r"chatcmpl-(.+)-[0-9a-f]{8}", e["request_id"])
        if match is None:
            raise ValueError("Invalid internal request ID")
        external = match[1]
        if external not in expected or external in result:
            raise ValueError("Missing/duplicate path mapping")
        if any(e.get(f) is None or not math.isfinite(e[f]) or e[f] < 0 for f in FIELDS):
            raise ValueError("Invalid path field")
        if not (e["enqueued_s"] <= e["first_compute_s"] <= e["first_output_s"]):
            raise ValueError("Path timestamp order mismatch")
        if not math.isclose(
            e["scheduler_ttft_ms"],
            e["queue_to_first_compute_ms"] + e["first_compute_to_output_ms"],
            abs_tol=1e-7,
        ):
            raise ValueError("Time partition mismatch")
        if e["completed_prefill_tokens"] != e["scheduled_prefill_tokens"]:
            raise ValueError("Unaccounted prompt work at first output")
        if e["preemptions"] == 0 and (
            e["completed_prefill_tokens"] + e["first_compute_cached_tokens"]
            != e["prompt_tokens"]
        ):
            raise ValueError("Prompt compute/cache balance mismatch")
        result[external] = e
    return {
        (r["conversation_id"], int(r["input_num_turns"])): result[external]
        for external, r in expected.items()
    }


def attach_path(paired, base_rows, base_stats, selected_rows, selected_stats):
    base = validate_path(base_rows, base_stats)
    selected = validate_path(selected_rows, selected_stats)
    for row in paired:
        key = (row["conversation_id"], row["message_count"])
        b, s = base[key], selected[key]
        if b["prompt_tokens"] != s["prompt_tokens"]:
            raise ValueError("Templated prompt length differs")
        row["server_prompt_tokens"] = s["prompt_tokens"]
        row["baseline_preemptions"] = b["preemptions"]
        row["selective_preemptions"] = s["preemptions"]
        for field in FIELDS:
            row["base_" + field] = b[field]
            row["selected_" + field] = s[field]
            row["delta_" + field] = s[field] - b[field]
        row["delta_outside_scheduler_ms"] = (
            row["delta_ms"] - row["delta_scheduler_ttft_ms"]
        )
        work = row["delta_completed_prefill_tokens"]
        row["work_group"] = (
            "same_work" if work == 0 else ("more_work" if work > 0 else "less_work")
        )
        row["action_work"] = row["action"] + "/" + row["work_group"]
    output = {}
    for dimension in ("action", "action_work"):
        groups = defaultdict(list)
        for row in paired:
            groups[row[dimension]].append(row)
        output[dimension] = [
            dict(
                group=group,
                n=len(rs),
                **{
                    field: statistics.mean(r[field] for r in rs)
                    for field in (
                        "delta_ms",
                        "delta_queue_to_first_compute_ms",
                        "delta_first_compute_to_output_ms",
                        "delta_scheduler_ttft_ms",
                        "delta_outside_scheduler_ms",
                        "delta_initial_gpu_hit_tokens",
                        "delta_completed_prefill_tokens",
                    )
                },
            )
            for group, rs in sorted(groups.items())
        ]
    return output
