# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded decision snapshots, joined offline to scheduler first-token events."""

import time
from collections import deque


class PromotionDecisionTrace:
    def __init__(self, capacity=4096, clock=time.perf_counter):
        if capacity < 1:
            raise ValueError("Trace capacity must be positive")
        self.clock = clock
        self.events = deque(maxlen=capacity)
        self.sequence = 0
        self.recorded = 0
        self.dropped = 0

    def record(self, **features):
        if len(self.events) == self.events.maxlen:
            self.dropped += 1
        self.events.append(
            dict(
                **features,
                decision_recorded_s=self.clock(),
                event_id=self.sequence,
            )
        )
        self.sequence += 1
        self.recorded += 1

    def snapshot(self, reset=False):
        result = dict(
            promotion_ttft_trace_enabled=1,
            promotion_ttft_trace_recorded=self.recorded,
            promotion_ttft_trace_dropped=self.dropped,
            promotion_ttft_trace_events=list(self.events),
        )
        if reset:
            self.events.clear()
            self.recorded = self.dropped = 0
        return result
