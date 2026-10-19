# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Partition a host call and its overlap with an older restore output timestamp."""

import math
from collections import defaultdict
from statistics import mean


def partition_call(call, start, end):
    marks = call["marks"]
    if call["overflow"] or len(marks) < 2:
        raise ValueError("Missing or overflowed host marks")
    if marks[0]["stage"] != "executor_dispatch" or marks[-1]["stage"] != "end":
        raise ValueError("Incomplete host call")
    if not math.isfinite(start) or not math.isfinite(end) or end < start:
        raise ValueError("Invalid overlap window")
    full, cpu, overlap = defaultdict(float), defaultdict(float), defaultdict(float)
    for left, right in zip(marks, marks[1:]):
        wall = right["wall_s"] - left["wall_s"]
        thread = right["cpu_s"] - left["cpu_s"]
        if not all(math.isfinite(v) and v >= 0 for v in (wall, thread)):
            raise ValueError("Non-monotonic host clocks")
        stage = left["stage"]
        full[stage] += 1000 * wall
        cpu[stage] += 1000 * thread
        overlap[stage] += 1000 * max(
            0, min(end, right["wall_s"]) - max(start, left["wall_s"])
        )
    outside = max(0, min(end, marks[0]["wall_s"]) - start) + max(
        0, end - max(start, marks[-1]["wall_s"])
    )
    overlap["outside_scope"] = 1000 * outside
    if not math.isclose(sum(overlap.values()), 1000 * (end - start), abs_tol=1e-6):
        raise ValueError("Host overlap partition mismatch")
    return dict(full_ms=dict(full), thread_cpu_ms=dict(cpu), overlap_ms=dict(overlap))


def analyze_host(stats, readiness):
    if not stats.get("restore_host_trace_enabled"):
        raise ValueError("Host trace did not take effect")
    batches = {
        rid: b
        for b in stats["restore_readiness_trace_events"]
        for rid in b["request_ids"]
    }
    workers = {
        e["request_id"]: e["workers"][0] for e in stats["restore_path_trace_events"]
    }
    events = []
    for e in readiness["events"]:
        rid = e["request_id"]
        timeline = {p["stage"]: p for p in batches[rid]["consumption"]}
        if "after_execute" not in timeline:
            events.append(dict(request_id=rid, no_execute=True))
            continue
        after = timeline["after_execute"]
        call = after.get("host_call")
        if not call or not any(m["stage"] == "runner_setup" for m in call["marks"]):
            raise ValueError("Missing model runner host trace")
        output = workers[rid]["executor_output_ready_s"]
        end = after["end_s"]
        start = min(end, max(output, timeline["before_execute"]["start_s"]))
        parts = partition_call(call, start, end)
        events.append(
            dict(
                request_id=rid,
                no_execute=False,
                target=e["first_done_stage"] == "after_execute",
                scheduled_tokens=call["scheduled_tokens"],
                scheduled_requests=call["scheduled_requests"],
                ready_to_execute_return_ms=1000 * (end - start),
                **parts,
            )
        )
    selected = [e for e in events if e.get("target")]

    def average(field):
        stages = sorted({k for e in selected for k in e[field]})
        return {k: mean(e[field].get(k, 0) for e in selected) for k in stages}

    return dict(
        requests=len(events),
        target_requests=len(selected),
        target_mean_overlap_ms=average("overlap_ms"),
        target_mean_full_ms=average("full_ms"),
        target_mean_thread_cpu_ms=average("thread_cpu_ms"),
        events=events,
        caveat="Host wall/CPU clocks only. CPU deltas are for full segments, "
        "not prorated to the output overlap. Output stamp is before Future "
        "completion. No inference of CUDA device time or recoverable savings.",
    )
