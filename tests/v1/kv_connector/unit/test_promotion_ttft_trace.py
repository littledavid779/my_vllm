# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.promotion_trace import PromotionDecisionTrace

from .test_offloading_connector import _make_promotion_scheduler


def test_bounded_decisions_and_reset():
    trace = PromotionDecisionTrace(capacity=1, clock=lambda: 2.0)
    trace.record(request_id="a", action="restore")
    trace.record(request_id="b", action="recompute")
    stats = trace.snapshot(reset=True)
    assert stats["promotion_ttft_trace_recorded"] == 2
    assert stats["promotion_ttft_trace_dropped"] == 1
    assert stats["promotion_ttft_trace_events"][0]["event_id"] == 1
    trace.record(request_id="c", action="restore")
    assert trace.events[0]["event_id"] == 2
    assert trace.events[0]["decision_recorded_s"] == 2
    assert trace.snapshot()["promotion_ttft_trace_dropped"] == 0


@pytest.mark.parametrize("action", ["restore", "recompute"])
def test_both_actions_record_without_full_cost_trace(action):
    scheduler, _ = _make_promotion_scheduler("always_" + action)
    assert scheduler.promotion_ttft_trace is None
    scheduler.promotion_ttft_trace = PromotionDecisionTrace()
    scheduler._trace_bytes_per_block = 1024
    scheduler._reqs_being_loaded["other"] = {b"1", b"2"}
    scheduler._reqs_being_stored["writer"] = {b"3"}
    request = SimpleNamespace(
        request_id="r",
        num_tokens=12,
        num_prompt_tokens=12,
        block_hashes=[BlockHash(str(i).encode()) for i in range(3)],
    )
    actual = scheduler.get_num_new_matched_tokens(request, 0)
    assert actual == ((12, True) if action == "restore" else (0, False))
    stats = scheduler.get_value_policy_stats()
    event = stats["promotion_ttft_trace_events"][0]
    assert event["action"] == action
    assert event["candidate_tokens"] == 12
    assert event["candidate_bytes"] == 1024
    assert event["pending_load_bytes"] == 2048
    assert event["pending_store_bytes"] == 1024
    assert event["prompt_tokens"] == 12
    assert stats["offload_cost_events_enabled"] == 0
    assert stats["offload_promotion_decision_count"] == 1
