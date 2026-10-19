# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-host, single-worker restore path audit. DMA is NOT an additive phase."""

import math
import statistics

STAGES = (
    ("enqueued_s", "decision_s", "enqueue_to_decision_ms"),
    ("decision_s", "allocated_s", "decision_to_alloc_ms"),
    ("allocated_s", "metadata_built_s", "alloc_to_metadata_ms"),
    ("metadata_built_s", "worker_received_s", "metadata_to_worker_ms"),
    ("worker_received_s", "submit_start_s", "worker_before_submit_ms"),
    ("submit_start_s", "submit_return_s", "host_submit_ms"),
    ("submit_return_s", "worker_observed_s", "submit_return_to_worker_observed_ms"),
    ("worker_observed_s", "scheduler_observed_s", "worker_to_scheduler_ms"),
    ("scheduler_observed_s", "first_compute_s", "scheduler_to_first_compute_ms"),
)


FEEDBACK_STAGES = (
    ("worker_observed_s", "executor_output_ready_s", "worker_to_output_ready_ms"),
    ("executor_output_ready_s", "engine_output_consumed_s", "ready_to_engine_ms"),
    ("engine_output_consumed_s", "scheduler_update_entry_s", "engine_to_update_ms"),
    ("scheduler_update_entry_s", "scheduler_observed_s", "update_to_connector_ms"),
)


def analyze_restore_path(stats, *, require_feedback=False):
    if stats.get("restore_path_trace_enabled") != 1:
        raise ValueError("Restore path tracing is not enabled")
    for suffix in (
        "active",
        "dropped",
        "unmatched",
        "duplicate_active",
        "duplicate_stage",
        "cancelled",
    ):
        if stats.get("restore_path_trace_" + suffix, 0):
            raise ValueError(f"Incomplete restore trace: {suffix}")
    paths = {e["request_id"]: e for e in stats["request_path_trace_events"]}
    expected = {
        rid for rid, e in paths.items() if e["initial_external_match_tokens"] > 0
    }
    result = []
    seen = set()
    for event in stats["restore_path_trace_events"]:
        rid = event["request_id"]
        if rid not in expected or rid in seen:
            raise ValueError("Unexpected or repeated restore request")
        seen.add(rid)
        if len(event["workers"]) != 1:
            raise ValueError("This diagnostic requires exactly one worker observation")
        worker = event["workers"][0]
        if worker["host"] != event["host"]:
            raise ValueError("Cross-host monotonic clocks cannot be compared")
        path = paths[rid]
        if path["preemptions"]:
            raise ValueError("Preempted requests require a multi-episode analysis")
        times = {**path, **event, **worker}
        row = dict(request_id=rid)
        for start, end, field in STAGES:
            a, b = times[start], times[end]
            if not all(math.isfinite(v) and v >= 0 for v in (a, b)) or b < a:
                raise ValueError(f"Invalid or unordered timestamps: {field}")
            row[field] = (b - a) * 1000
        feedback_keys = (
            "executor_output_ready_s",
            "engine_output_consumed_s",
            "scheduler_update_entry_s",
        )
        if require_feedback or any(k in worker for k in feedback_keys):
            if not all(k in worker for k in feedback_keys):
                raise ValueError("Incomplete restore feedback timestamps")
            if worker.get("duplicate_output_stamp"):
                raise ValueError("Duplicate restore feedback timestamps")
            if worker.get("executor") != "UniProcExecutor":
                raise ValueError("Feedback diagnostic requires UniProcExecutor")
            if worker.get("engine_mode") not in ("step", "batch_queue"):
                raise ValueError("Missing engine mode")
            for start, end, field in FEEDBACK_STAGES:
                a, b = times[start], times[end]
                if not all(math.isfinite(v) and v >= 0 for v in (a, b)) or b < a:
                    raise ValueError(f"Invalid feedback timestamps: {field}")
                row[field] = (b - a) * 1000
            if not math.isclose(
                sum(row[f] for _, _, f in FEEDBACK_STAGES),
                row["worker_to_scheduler_ms"],
                abs_tol=1e-6,
            ):
                raise ValueError("Feedback partition mismatch")
            row["engine_mode"] = worker["engine_mode"]
        dma = worker["dma_elapsed_s"]
        if dma is None or not math.isfinite(dma) or dma < 0:
            raise ValueError("Invalid DMA elapsed time")
        row["dma_elapsed_ms"] = dma * 1000
        row["bytes"] = worker["bytes"]
        if not isinstance(row["bytes"], int) or row["bytes"] <= 0:
            raise ValueError("Invalid transfer bytes")
        row["enqueue_to_first_compute_ms"] = path["queue_to_first_compute_ms"]
        if not math.isclose(
            sum(row[f] for _, _, f in STAGES),
            row["enqueue_to_first_compute_ms"],
            abs_tol=1e-6,
        ):
            raise ValueError("Restore partition does not match request path")
        result.append(row)
    if seen != expected or not seen:
        raise ValueError("Missing restore observations")
    fields = [f for _, _, f in STAGES] + [
        "dma_elapsed_ms",
        "enqueue_to_first_compute_ms",
        "bytes",
    ]
    feedback = [r for r in result if "ready_to_engine_ms" in r]
    if feedback and len(feedback) != len(result):
        raise ValueError("Mixed feedback trace versions")
    return dict(
        count=len(result),
        means={f: statistics.mean(r[f] for r in result) for f in fields},
        events=result,
        feedback_count=len(feedback),
        feedback_means={
            f: statistics.mean(r[f] for r in feedback) for _, _, f in FEEDBACK_STAGES
        }
        if feedback
        else {},
        caveat="Host phases partition waiting; CUDA elapsed overlaps them. "
        "Worker observation includes polling latency, not exact DMA completion.",
    )
