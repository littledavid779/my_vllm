# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental FIFO-ready restore notification drain; never waits on a Future."""

import time
from collections import Counter
from concurrent.futures import Future


class ReadyRestoreDrain:
    def __init__(self, mode: str):
        if mode not in ("observe", "on"):
            raise ValueError("Restore ready-first mode must be off, observe, or on")
        self.mode = mode
        self.counts: Counter[str] = Counter()

    def should_drain(self, batch_queue) -> bool:
        self.counts["checks"] += 1
        if not batch_queue:
            self.counts["empty"] += 1
            return False
        # Only the oldest result, never scan past it. UniProc uses plain Future.
        future = batch_queue[-1][0]
        if type(future) is not Future:
            self.counts["unsupported_future"] += 1
            return False
        if not future.done():
            self.counts["not_ready"] += 1
            return False
        self.counts["ready"] += 1
        if future.cancelled() or future.exception() is not None:
            # Keep failure propagation in the existing result-consumption path.
            self.counts["failed"] += 1
            return False
        output = future.result()
        connector = getattr(output, "kv_connector_output", None)
        if not connector or not connector.finished_recving:
            self.counts["no_restore"] += 1
            return False
        self.counts["eligible"] += 1
        return self.mode == "on"

    def consumed(self):
        self.counts["consumed"] += 1

    def snapshot(self, reset=False):
        result: dict[str, str | int] = {"restore_ready_first_mode": self.mode}
        result.update({"restore_ready_first_" + k: v for k, v in self.counts.items()})
        if reset:
            self.counts.clear()
        return result


class RestoreReadinessTrace:
    """Observe FIFO Futures at host boundaries; no callbacks or extra result reads.

    A done() observation brackets visibility, not the exact completion instant.
    Keep only active batches and bounded finished restore records.
    """

    def __init__(self, capacity=4096, clock=time.perf_counter):
        self.capacity = capacity
        self.clock = clock
        self.active = {}
        self.events = []
        self.counts = Counter()
        self.loop = 0
        self.serial = 0
        self.head = None
        self.timeline = []

    def _register(self, future, origin):
        if type(future) is not Future:
            self.counts["unsupported_future"] += 1
            return
        if future in self.active:
            return
        if len(self.active) >= 64:
            self.counts["active_overflow"] += 1
            return
        self.serial += 1
        self.active[future] = dict(
            batch_id=self.serial,
            origin=origin,
            created_loop=self.loop,
            submission=list(self.timeline),
            observations=[],
        )

    def begin(self, batch_queue):
        self.loop += 1
        self.timeline = []
        self.head = batch_queue[-1][0] if batch_queue else None
        if self.head is not None:
            self._register(self.head, "present_at_entry")
        self.mark("entry")

    def _observe(self, future, stage):
        start = self.clock()
        done = future.done() if type(future) is Future else None
        end = self.clock()
        row = dict(loop=self.loop, stage=stage, start_s=start, end_s=end, done=done)
        record = self.active.get(future)
        if record is not None:
            if len(record["observations"]) < 128:
                record["observations"].append(row)
            else:
                self.counts["observation_overflow"] += 1
        return row

    def mark(self, stage, **details):
        row = self._observe(self.head, stage)
        row.update(details)
        self.timeline.append(row)

    def enqueued(self, future):
        self._register(future, "enqueued")
        if self.head is None:
            self.head = future
        else:
            self._observe(future, "enqueued")
        self.mark("after_enqueue")

    def before_result(self, future):
        if self.head is not future:
            self.counts["head_mismatch"] += 1
        self.head = future
        self.mark("before_result")

    def finish(self, future, output):
        self.mark("after_result")
        record = self.active.pop(future, None)
        self.counts["consumed_batches"] += 1
        connector = getattr(output, "kv_connector_output", None)
        if connector is None or not connector.finished_recving:
            return
        self.counts["restore_batches"] += 1
        if record is None:
            self.counts["missing_batch"] += 1
            return
        if len(self.events) >= self.capacity:
            self.counts["dropped"] += 1
            return
        record.update(
            request_ids=sorted(connector.finished_recving),
            consumed_loop=self.loop,
            consumption=list(self.timeline),
        )
        self.events.append(record)

    def snapshot(self, reset=False):
        result = dict(
            restore_readiness_trace_enabled=True,
            restore_readiness_trace_events=list(self.events),
            restore_readiness_trace_active=len(self.active),
            **{"restore_readiness_trace_" + k: v for k, v in self.counts.items()},
        )
        if reset:
            self.events.clear()
            self.counts.clear()
        return result
