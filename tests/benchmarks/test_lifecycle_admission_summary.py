# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from scripts.summarize_lifecycle_admission import summarize_stats


def trace(events):
    return {"offload_admission_trace_enabled": 1, "offload_cost_events": events}


def admission(event_id=0, selected=0, stored=(), failed=False):
    return dict(
        event_type="admission",
        event_id=event_id,
        request_id="a",
        candidate_block_ids=[1, 2, 3],
        candidate_states=["missing", "pending", "ready"],
        selected=selected,
        stored_block_ids=list(stored),
        allocation_failed=failed,
    )


def lookup(event_id=1, request_id="b", state="missing"):
    return dict(
        event_type="external_lookup",
        event_id=event_id,
        request_id=request_id,
        candidate_block_ids=[1, 2, 3],
        start_block_idx=4,
        hit_blocks=0,
        candidate_states=[state, "ready", "ready"],
    )


def test_retries_deduplicate_and_existing_unselected_is_not_rejection():
    result = summarize_stats(trace([admission(), lookup(), lookup(2)]))
    assert result["unique_rejected_missing_blocks"] == 1
    assert result["distinct_request_rejected_block_demand_pairs"] == 1
    assert result["counts"]["ready_suffix_behind_prior_rejection"] == 2
    assert result["examples"][0]["rejection_event_id"] == 0


def test_same_request_not_future_demand():
    result = summarize_stats(trace([admission(), lookup(request_id="a")]))
    assert result["unique_rejected_blocks_with_later_suffix_demand"] == 0
    assert result["counts"].get("first_gap_with_prior_rejection", 0) == 0


def test_later_reservation_breaks_rejection_association():
    result = summarize_stats(
        trace(
            [
                admission(),
                admission(1, selected=3, stored=[1]),
                lookup(2),
            ]
        )
    )
    assert result["unique_rejected_blocks_with_later_suffix_demand"] == 0


def test_failed_admission_not_policy_rejection():
    result = summarize_stats(trace([admission(selected=3, failed=True), lookup()]))
    assert result["unique_rejected_missing_blocks"] == 0
    assert result["counts"]["allocation_failures"] == 1


def test_pending_gap_is_not_missing_rejected_gap():
    result = summarize_stats(trace([admission(), lookup(state="pending")]))
    assert result["counts"]["first_gap_pending"] == 1
    assert result["counts"].get("first_gap_with_prior_rejection", 0) == 0


@pytest.mark.parametrize(
    "key",
    [
        "offload_cost_events_dropped",
        "offload_cost_event_block_id_evictions",
    ],
)
def test_loss_is_rejected(key):
    with pytest.raises(ValueError, match="Incomplete trace"):
        summarize_stats({**trace([]), key: 1})


def test_old_results_not_reinterpreted():
    with pytest.raises(ValueError, match="No admission trace"):
        summarize_stats({"offload_cost_events": []})


def test_eviction_demand_counts_missing_only_and_reservation_clears():
    evict = dict(
        event_type="store", event_id=0, evicted_block_ids=[1, 2], stored_block_ids=[]
    )
    reserve = dict(
        event_type="store", event_id=3, evicted_block_ids=[], stored_block_ids=[1]
    )
    result = summarize_stats(trace([evict, lookup(), lookup(2), reserve, lookup(4)]))
    assert result["unique_evicted_blocks_with_later_missing_suffix_demand"] == 1
    assert result["distinct_request_evicted_block_demand_pairs"] == 1
    assert result["first_missing_gaps_with_prior_cpu_eviction"] == 1
