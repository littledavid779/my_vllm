# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded, opt-in host-side restore path observations (not a DMA clock)."""

import os
import socket
import time
from collections import Counter, deque

TRACE_HOST = socket.gethostname()
OUTPUT_TRACE_ENABLED = os.environ.get("VLLM_EXPERIMENTAL_RESTORE_PATH_TRACE") == "1"


def stamp_output(output, stage, **details):
    """Stamp existing restore payloads only; no callback or synchronization.

    perf_counter is comparable across processes on this host. The offline
    diagnostic rejects cross-host and multiple-worker observations.
    """
    if not OUTPUT_TRACE_ENABLED or output is None:
        return
    connector = getattr(output, "kv_connector_output", None)
    if connector is None or not connector.finished_recving:
        return
    stats = connector.kv_connector_stats
    if stats is None:
        return
    now = None
    for operations in stats.data.values():
        for operation in operations:
            timing = (
                operation.get("restore_timing")
                if isinstance(operation, dict)
                else getattr(operation, "restore_timing", None)
            )
            if timing is not None:
                if now is None:
                    now = time.perf_counter()
                if stage in timing:
                    timing["duplicate_output_stamp"] = True
                else:
                    timing[stage] = now
                    timing.update(details)


def get_traced_output(async_output):
    """Called instead of get_output only when restore tracing is enabled."""
    output = async_output.get_output()
    stamp_output(output, "executor_output_ready_s", executor="UniProcExecutor")
    return output


class RestorePathTrace:
    def __init__(self, capacity=4096, clock=time.perf_counter):
        if capacity < 1:
            raise ValueError("Trace capacity must be positive")
        self.capacity = capacity
        self.clock = clock
        self.active = {}
        self.events = deque(maxlen=capacity)
        self.counters = Counter()

    def decision(self, request_id):
        previous = self.active.get(request_id)
        if previous is not None:
            if "allocated_s" in previous:
                self.counters["duplicate_active"] += 1
                return
            # Allocation may fail after lookup. Keep the last successful attempt.
            self.counters["retried_decisions"] += 1
        elif len(self.active) >= self.capacity:
            self.counters["dropped"] += 1
            return
        self.active[request_id] = dict(
            request_id=request_id,
            host=TRACE_HOST,
            decision_s=self.clock(),
            workers=[],
        )

    def mark(self, request_id, stage):
        event = self.active.get(request_id)
        if event is None:
            self.counters["unmatched"] += 1
        elif stage in event:
            self.counters["duplicate_stage"] += 1
        else:
            event[stage] = self.clock()

    def observe(self, request_id, observation):
        event = self.active.get(request_id)
        if event is None:
            self.counters["unmatched"] += 1
        elif len(event["workers"]) >= 16:
            self.counters["dropped"] += 1
        else:
            event["workers"].append(dict(observation))

    def complete(self, request_id):
        self.mark(request_id, "scheduler_observed_s")
        event = self.active.pop(request_id, None)
        if event is not None:
            if len(self.events) == self.capacity:
                self.counters["dropped"] += 1
            self.events.append(event)

    def cancel(self, request_id):
        if self.active.pop(request_id, None) is not None:
            self.counters["cancelled"] += 1

    def snapshot(self, reset=False):
        result = dict(
            restore_path_trace_enabled=1,
            restore_path_trace_active=len(self.active),
            restore_path_trace_events=list(self.events),
            **{f"restore_path_trace_{k}": v for k, v in self.counters.items()},
        )
        if reset:
            self.events.clear()
            self.counters.clear()
        return result
