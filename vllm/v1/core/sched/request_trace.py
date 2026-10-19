# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in bounded scheduler-to-first-token observations, not a GPU timer."""

import time
from collections import Counter, deque
from collections.abc import Callable
from typing import Any


class RequestPathTrace:
    def __init__(self, capacity: int = 4096, clock: Callable = time.perf_counter):
        if capacity < 1:
            raise ValueError("Trace capacity must be positive")
        self.capacity = capacity
        self.clock = clock
        self.active: dict[str, dict[str, Any]] = {}
        self.batches: dict[int, dict[str, int]] = {}
        self.events: deque[dict[str, Any]] = deque(maxlen=capacity)
        self.counters: Counter = Counter()

    def enqueue(self, request) -> None:
        if (
            request.resumable
            or request.pooling_params is not None
            or request.has_encoder_inputs
        ):
            self.counters["unsupported_requests"] += 1
            return
        if len(self.active) >= self.capacity:
            self.counters["dropped"] += 1
            return
        if request.request_id in self.active:
            self.counters["duplicate_requests"] += 1
            return
        self.active[request.request_id] = dict(
            request_id=request.request_id,
            enqueued_s=self.clock(),
            prompt_tokens=request.num_prompt_tokens,
            first_compute_s=None,
            initial_gpu_hit_tokens=None,
            initial_external_match_tokens=None,
            first_compute_cached_tokens=None,
            scheduled_prefill_tokens=0,
            completed_prefill_tokens=0,
            prefill_steps=0,
        )

    def admission(self, request, local_tokens: int, external_tokens: int) -> None:
        state = self.active.get(request.request_id)
        if state is not None and state["initial_gpu_hit_tokens"] is None:
            state["initial_gpu_hit_tokens"] = local_tokens
            state["initial_external_match_tokens"] = external_tokens

    def schedule(self, batch, request, tokens: int) -> None:
        state = self.active.get(request.request_id)
        if state is None:
            return
        if state["first_compute_s"] is None:
            state["first_compute_s"] = self.clock()
            state["first_compute_cached_tokens"] = request.num_computed_tokens
        # Count prompt positions only; future decode placeholders are excluded.
        start = min(request.num_computed_tokens, request.num_prompt_tokens)
        end = min(request.num_computed_tokens + tokens, request.num_prompt_tokens)
        count = max(0, end - start)
        if count:
            if id(batch) not in self.batches and len(self.batches) >= self.capacity:
                self.counters["dropped"] += 1
                return
            self.batches.setdefault(id(batch), {})[request.request_id] = count
            state["scheduled_prefill_tokens"] += count

    def take_batch(self, batch) -> dict[str, int]:
        return self.batches.pop(id(batch), {})

    def completed_step(self, request_id: str, prompt_tokens: int) -> None:
        state = self.active.get(request_id)
        if state is not None and prompt_tokens:
            state["completed_prefill_tokens"] += prompt_tokens
            state["prefill_steps"] += 1

    def first_output(self, request) -> None:
        state = self.active.pop(request.request_id, None)
        if state is None:
            return
        now = self.clock()
        first = state["first_compute_s"]
        if first is None:
            self.counters["invalid_first_output"] += 1
            return
        state.update(
            first_output_s=now,
            queue_to_first_compute_ms=(first - state["enqueued_s"]) * 1000,
            first_compute_to_output_ms=(now - first) * 1000,
            scheduler_ttft_ms=(now - state["enqueued_s"]) * 1000,
            preemptions=request.num_preemptions,
        )
        if len(self.events) == self.capacity:
            self.counters["dropped"] += 1
        self.events.append(state)

    def finish(self, request_id: str) -> None:
        if self.active.pop(request_id, None) is not None:
            self.counters["finished_without_first_token"] += 1

    def snapshot(self, reset: bool = False) -> dict[str, Any]:
        result = {
            "request_path_trace_enabled": 1,
            "request_path_trace_active": len(self.active),
            "request_path_trace_pending_batches": len(self.batches),
            "request_path_trace_events": list(self.events),
            **{f"request_path_trace_{k}": v for k, v in self.counters.items()},
        }
        if reset:
            self.events.clear()
            self.counters.clear()
        return result
