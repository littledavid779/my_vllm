# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Join host Future observations to restore records; visibility is interval-censored."""

from collections import Counter
from statistics import mean


def analyze_readiness(stats, joined):
    if not stats.get("restore_readiness_trace_enabled"):
        raise ValueError("Readiness trace not enabled")
    for name in (
        "unsupported_future",
        "active_overflow",
        "observation_overflow",
        "head_mismatch",
        "missing_batch",
        "dropped",
        "active",
    ):
        if stats.get("restore_readiness_trace_" + name, 0):
            raise ValueError(f"Incomplete readiness trace: {name}")
    batches = stats.get("restore_readiness_trace_events", [])
    if len(batches) != stats.get("restore_readiness_trace_restore_batches", 0):
        raise ValueError("Missing restore batches")
    by_request = {}
    for batch in batches:
        obs = batch["observations"]
        previous = -float("inf")
        was_done = False
        for point in obs:
            if not previous <= point["start_s"] <= point["end_s"]:
                raise ValueError("Non-monotonic observations")
            previous = point["end_s"]
            if was_done and point["done"] is not True:
                raise ValueError("Future went from done to pending")
            was_done = was_done or point["done"] is True
        if not was_done:
            raise ValueError("Consumed Future was never observed done")
        for rid in batch["request_ids"]:
            if rid in by_request:
                raise ValueError("Repeated restore notification")
            by_request[rid] = batch
    expected = {row["request_id"] for row in joined["events"]}
    if set(by_request) != expected:
        raise ValueError("Readiness request set differs from restore path")
    raw = {r["request_id"]: r for r in stats["restore_path_trace_events"]}
    events = []
    for row in joined["events"]:
        rid = row["request_id"]
        batch = by_request[rid]
        obs = batch["observations"]
        first_done = next(p for p in obs if p["done"] is True)
        pending = [p for p in obs if p["done"] is False]
        last_pending = pending[-1] if pending else None
        same_loop = batch["created_loop"] == batch["consumed_loop"]
        stage = first_done["stage"]
        if stage in ("enqueued", "after_enqueue"):
            stage = "new_future_done_at_enqueue"
        worker = raw[rid]["workers"][0]
        output_s = worker["executor_output_ready_s"]
        consumed_s = worker["engine_output_consumed_s"]
        # The output stamp is before task return/Future completion, not its synonym.
        if output_s > first_done["end_s"] or consumed_s > obs[-1]["end_s"]:
            raise ValueError("Output/Future observation ordering mismatch")
        timeline = batch["consumption"]
        entry = timeline[0]
        if output_s < entry["start_s"]:
            output_phase = "before_entry"
        else:
            output_phase = "after_last_boundary"
            for point in timeline:
                if output_s <= point["end_s"]:
                    output_phase = "by_" + point["stage"]
                    break
        events.append(
            dict(
                request_id=rid,
                batch_id=batch["batch_id"],
                created_and_consumed_same_loop=same_loop,
                first_done_stage=stage,
                output_stamp_phase=output_phase,
                entry_done=next(
                    (
                        p["done"]
                        for p in obs
                        if p["loop"] == batch["consumed_loop"] and p["stage"] == "entry"
                    ),
                    None,
                ),
                last_pending=last_pending,
                first_done=first_done,
                visibility_bracket_ms=(
                    1000 * (first_done["end_s"] - last_pending["start_s"])
                    if last_pending
                    else None
                ),
                output_stamp_to_first_done_observation_ms=(
                    1000 * (first_done["end_s"] - output_s)
                ),
                ready_to_engine_ms=row["ready_to_engine_ms"],
                before_schedule_ms=row["before_schedule_ms"],
            )
        )
    return dict(
        requests=len(events),
        batches=len(batches),
        first_done_stages=dict(Counter(e["first_done_stage"] for e in events)),
        output_stamp_phases=dict(Counter(e["output_stamp_phase"] for e in events)),
        same_loop_requests=sum(e["created_and_consumed_same_loop"] for e in events),
        mean_ready_to_engine_ms=mean(e["ready_to_engine_ms"] for e in events),
        events=events,
        caveat="Boundary observations only; the bracket is not exact completion time. "
        "Output stamps precede Future completion. Trace-on diagnosis only.",
    )
