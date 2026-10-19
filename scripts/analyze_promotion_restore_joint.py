# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Join decision and restore clocks without double-counting DMA or feedback."""

import math
import statistics

from analyze_lifecycle_restore_path import FEEDBACK_STAGES, STAGES, analyze_restore_path
from analyze_promotion_ttft_labels import build_labels

PHASES = (
    "decision_clock_offset_ms",
    *(field for _, _, field in STAGES[1:7]),
    *(field for _, _, field in FEEDBACK_STAGES),
    "scheduler_to_first_compute_ms",
)


def join_restore_labels(rows, stats):
    labels = build_labels(rows, stats)
    if labels["excluded"]:
        raise ValueError("Joint diagnosis requires single, non-preempted decisions")
    if stats.get("restore_path_trace_retried_decisions", 0):
        raise ValueError("Retried restore decisions require episode analysis")
    restore = analyze_restore_path(stats, require_feedback=True)
    path_map = {e["request_id"]: e for e in stats["request_path_trace_events"]}
    raw_map = {e["request_id"]: e for e in stats["restore_path_trace_events"]}
    restore_map = {e["request_id"]: e for e in restore["events"]}
    expected = {e["request_id"] for e in labels["labels"] if e["action"] == "restore"}
    if set(restore_map) != expected:
        raise ValueError("Restore observations do not match promotion actions")
    joined = []
    for e in labels["labels"]:
        if e["action"] != "restore":
            continue
        rid = e["request_id"]
        p, r = path_map[rid], restore_map[rid]
        offset = 1000 * (raw_map[rid]["decision_s"] - e["decision_recorded_s"])
        if not math.isfinite(offset) or offset < 0:
            raise ValueError("Restore decision predates promotion record")
        if r["bytes"] != e["candidate_bytes"]:
            raise ValueError("Observed transfer differs from candidate bytes")
        cached = min(
            e["prompt_tokens"] - 1, e["gpu_hit_tokens"] + e["candidate_tokens"]
        )
        if p["first_compute_cached_tokens"] != cached:
            raise ValueError("Actual restored prefix differs")
        row = dict(
            e,
            **{f: r[f] for f in PHASES if f != "decision_clock_offset_ms"},
            decision_clock_offset_ms=offset,
            before_schedule_ms=1000 * (p["first_compute_s"] - e["decision_recorded_s"]),
            after_schedule_ms=1000 * (p["first_output_s"] - p["first_compute_s"]),
            dma_elapsed_ms=r["dma_elapsed_ms"],
            engine_mode=r["engine_mode"],
        )
        if not math.isclose(
            sum(row[f] for f in PHASES), row["before_schedule_ms"], abs_tol=1e-6
        ):
            raise ValueError("Joint pre-schedule partition mismatch")
        if not math.isclose(
            row["before_schedule_ms"] + row["after_schedule_ms"],
            row["decision_to_first_token_ms"],
            abs_tol=1e-6,
        ):
            raise ValueError("Joint full label partition mismatch")
        joined.append(row)
    return dict(
        count=len(joined),
        events=joined,
        phases=list(PHASES),
        means={f: statistics.mean(e[f] for e in joined) for f in PHASES},
        restore_path=restore,
        caveat="Single host/worker; DMA and parent feedback are not additive phases. "
        "All intervals end at host observations, not exact device completion.",
    )
