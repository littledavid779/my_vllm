# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from dataclasses import replace

import pytest

from scripts.build_lifecycle_policy_config import main as build_config_main
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.arc_manager import ARCOffloadingManager
from vllm.v1.kv_offload.backends.cpu import CPUBackend
from vllm.v1.kv_offload.lifecycle_policy import (
    LifecycleOffloadingManager,
    LifecyclePolicyConfig,
)
from vllm.v1.kv_offload.lru_manager import LRUOffloadingManager


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


@pytest.mark.parametrize(
    "arm,disabled",
    [
        ("all-block", False),
        ("admission", False),
        ("promotion", False),
        ("full", False),
        ("full", True),
    ],
)
def test_config_generator_ablation_arms(monkeypatch, capsys, arm, disabled):
    argv = [
        "build_lifecycle_policy_config.py",
        "--arm",
        arm,
        "--cpu-bytes",
        "1073741824",
        "--block-size",
        "16",
        "--store-ms-per-block",
        "2",
        "--restore-ms-per-block",
        "2",
        "--recompute-ms-per-token",
        "1",
    ]
    if disabled:
        argv.append("--disable-guard")
    monkeypatch.setattr("sys.argv", argv)
    build_config_main()
    output = json.loads(capsys.readouterr().out)
    extra = output["kv_connector_extra_config"]
    assert extra["block_size"] == 16
    if arm == "all-block":
        assert "lifecycle_policy" not in extra
        assert extra["promotion_policy"] == "always_restore"
    else:
        config = LifecyclePolicyConfig.from_dict(extra["lifecycle_policy"])
        assert config.store_s_per_block == 0.002
        assert config.recompute_s_per_token == 0.001
        assert config.admission_enabled == (arm != "promotion")
        assert config.guard_enabled == (arm == "full" and not disabled)
        assert extra["promotion_policy"] == (
            "always_restore" if arm == "admission" else "cost_aware"
        )


def make_policy(**kwargs):
    cfg = LifecyclePolicyConfig(0.002, 0.002, 0.001)
    clock = Clock()
    policy = LifecycleOffloadingManager(
        LRUOffloadingManager(CPUBackend(block_size=16, num_blocks=16)),
        replace(cfg, **kwargs),
        16,
        clock=clock,
    )
    return policy, clock


def hashes(*values):
    return [BlockHash(str(value).encode()) for value in values]


@pytest.mark.parametrize("manager_class", [LRUOffloadingManager, ARCOffloadingManager])
@pytest.mark.parametrize("ready", [False, True])
def test_new_write_cost_excludes_allocated_blocks(manager_class, ready):
    outputs = []
    for enabled in (False, True):
        backing = manager_class(CPUBackend(block_size=16, num_blocks=16))
        backing.prepare_store(hashes(1))
        if ready:
            backing.complete_store(hashes(1))
        policy = LifecycleOffloadingManager(
            backing,
            LifecyclePolicyConfig(
                0.002,
                0.002,
                0.001,
                guard_enabled=False,
                new_write_store_cost=enabled,
                trace_admission=True,
            ),
            16,
            clock=Clock(),
        )
        output = policy.prepare_store(hashes(1, 2))
        outputs.append(output.block_hashes_to_store)
        assert policy.last_admission["candidate_states"] == [
            "ready" if ready else "pending",
            "missing",
        ]
        assert policy.last_admission["candidate_new_blocks"] == 1
        assert policy.counters["estimated_store_charge_s"] == (
            0.002 if enabled else 0.0
        )
    # p=.125: old G(2)=.0035-.004<0; new G(2)=.0035-.002>0.
    assert outputs == [[], hashes(2)]


def test_trace_does_not_change_admission_and_missing_cost_unchanged():
    results = []
    for tracing, correction in ((False, False), (True, False), (True, True)):
        policy, _ = make_policy(
            trace_admission=tracing, new_write_store_cost=correction
        )
        results.append(policy.prepare_store(hashes(1, 2)).block_hashes_to_store)
    assert results == [[], [], []]


def test_new_write_unknown_residency_is_charged(monkeypatch):
    policy, _ = make_policy(new_write_store_cost=True, trace_admission=True)
    monkeypatch.setattr(
        policy._backing, "get_block_residency", lambda hs: ["unknown" for _ in hs]
    )
    assert policy.prepare_store(hashes(1, 2)).block_hashes_to_store == []
    assert policy.last_admission["candidate_new_blocks"] == 2


@pytest.mark.parametrize("manager_class", [LRUOffloadingManager, ARCOffloadingManager])
def test_residency_snapshot_does_not_touch_replacement_state(manager_class):
    backing = manager_class(CPUBackend(block_size=16, num_blocks=2))
    backing.prepare_store(hashes(1, 2))
    backing.complete_store(hashes(1))
    assert backing.get_block_residency(hashes(2, 3, 1)) == [
        "pending",
        "missing",
        "ready",
    ]
    backing.complete_store(hashes(2))
    result = backing.prepare_store(hashes(3))
    assert result.block_hashes_evicted == hashes(1)


def test_empty_batch_clears_last_admission():
    policy, _ = make_policy(trace_admission=True)
    policy.prepare_store(hashes(1))
    assert policy.last_admission is not None
    policy.prepare_store([])
    assert policy.last_admission is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("store_s_per_block", -1),
        ("restore_fixed_s", float("nan")),
        ("demand_prior", 1.1),
        ("ewma_alpha", 0),
        ("max_tracker_size", 1.5),
        ("guard_window_s", 0),
        ("admission_enabled", "false"),
        ("recompute_s_per_token", 0),
        ("survival_discount", float("inf")),
        ("restore_s_per_block", 0),
        ("cold_probe_every", True),
        ("cold_probe_every", -1),
        ("cold_probe_budget_s", -1),
        ("cold_probe_min_blocks", 65),
    ],
)
def test_invalid_config(field, value):
    with pytest.raises(ValueError):
        make_policy(**{field: value})


def test_config_rejects_missing_and_unknown():
    with pytest.raises(ValueError, match="Missing"):
        LifecyclePolicyConfig.from_dict({})
    with pytest.raises(ValueError, match="Unknown"):
        LifecyclePolicyConfig.from_dict({"typo": 1})


def test_cold_probe_cadence_budget_and_reset():
    policy, _ = make_policy(
        store_s_per_block=0.1,
        cold_probe_every=2,
        cold_probe_min_blocks=1,
        cold_probe_max_blocks=2,
        cold_probe_budget_blocks=3,
        cold_probe_budget_s=1,
    )
    assert not policy.prepare_store(hashes(1, 2)).block_hashes_to_store
    result = policy.prepare_store(hashes(1, 2))
    assert result.block_hashes_to_store == hashes(1, 2)
    policy.complete_store(hashes(1, 2))
    event = policy.events[-1]
    assert event["reason"] == "cold_probe" and event["gain_s"] < 0
    policy.get_value_policy_stats(reset=True)
    assert policy.cold_probe_blocks_spent == 2
    assert not policy.prepare_store(hashes(3, 4)).block_hashes_to_store
    assert policy.prepare_store(hashes(3, 4)).block_hashes_to_store == hashes(3)
    for _ in range(4):
        assert not policy.prepare_store(hashes(4, 5)).block_hashes_to_store
    assert policy.cold_probe_blocks_spent == 3
    assert policy.cold_probe_cost_spent_s == pytest.approx(0.3)


def test_cold_probe_cost_budget_and_guard_are_respected():
    policy, _ = make_policy(
        store_s_per_block=0.1,
        cold_probe_every=1,
        cold_probe_min_blocks=1,
        cold_probe_max_blocks=4,
        cold_probe_budget_blocks=10,
        cold_probe_budget_s=0.25,
    )
    assert policy.prepare_store(hashes(1, 2, 3, 4)).block_hashes_to_store == hashes(
        1, 2
    )
    assert not policy.prepare_store(hashes(3, 4)).block_hashes_to_store
    assert policy.cold_probe_cost_spent_s <= 0.25
    other, _ = make_policy(
        store_s_per_block=0.1,
        cold_probe_every=1,
        cold_probe_min_blocks=1,
        cold_probe_max_blocks=4,
    )
    other.guard_state = "THROTTLE"
    for _ in range(8):
        assert not other.prepare_store(hashes(1, 2)).block_hashes_to_store
    assert other.cold_probe_blocks_spent == 0


def test_cold_probe_only_charges_new_successfully_reserved_blocks():
    policy, _ = make_policy(
        store_s_per_block=0.1,
        cold_probe_every=1,
        cold_probe_min_blocks=1,
        cold_probe_max_blocks=2,
        cold_probe_budget_s=1,
    )
    policy.prepare_store(hashes(1, 2))
    policy.complete_store(hashes(1, 2))
    policy.prepare_store(hashes(1, 2))
    assert policy.cold_probe_blocks_spent == 2
    full, _ = make_policy(
        store_s_per_block=0.1,
        cold_probe_every=1,
        cold_probe_min_blocks=17,
        cold_probe_max_blocks=17,
        cold_probe_budget_s=2,
    )
    assert full.prepare_store(hashes(*range(17))) is None
    assert full.cold_probe_blocks_spent == 0


def test_admission_observes_distinct_requests_and_real_cpu_restore():
    policy, _ = make_policy()
    blocks = hashes(1, 2)
    policy.observe_request("a", blocks)
    first = policy.prepare_store(blocks)
    assert first is not None and not first.block_hashes_to_store
    policy.observe_request("a", blocks)  # scheduler retry is not new reuse
    assert policy.history[blocks[0]][0] == 1
    policy.observe_request("b", blocks)
    second = policy.prepare_store(blocks)
    assert second is not None and second.block_hashes_to_store == blocks
    assert policy.lookup(blocks) == 0  # not ready until D2H completion
    policy.complete_store(blocks)
    assert policy.lookup(blocks) == 2
    policy.prepare_load(blocks)
    policy.complete_load(blocks)
    charge = policy.counters["estimated_store_charge_s"]
    third = policy.prepare_store(blocks)
    assert third is not None and not third.block_hashes_to_store
    assert policy.counters["estimated_store_charge_s"] == charge


def test_prefix_selection_does_not_store_isolated_hot_tail():
    policy, _ = make_policy(store_s_per_block=0.0035)
    blocks = hashes(1, 2, 3)
    policy.observe_request("a", blocks)
    policy.observe_request("b", blocks[:1])
    result = policy.prepare_store(blocks)
    assert result is not None and result.block_hashes_to_store == blocks[:1]
    other, _ = make_policy(store_s_per_block=0.0035)
    other.observe_request("a", blocks)
    other.observe_request("b", blocks[2:])
    assert other.prepare_store(blocks).block_hashes_to_store == []


def test_joint_prefix_cost_amortizes_restore_fixed_cost():
    policy, _ = make_policy(
        restore_fixed_s=0.020,
        demand_prior=1,
        survival_discount=1,
        store_s_per_block=0.001,
    )
    # One block is cheaper to recompute; a four-block bundle is worth restoring.
    assert policy.prepare_store(hashes(1)).block_hashes_to_store == []
    assert policy.prepare_store(hashes(1, 2, 3, 4)).block_hashes_to_store == hashes(
        1, 2, 3, 4
    )


def test_promotion_feedback_uses_completion_not_dma_and_learns_bucket():
    policy, clock = make_policy(min_cost_samples=1)
    assert policy.choose("a", 1, 16, 16) == "restore"
    clock.now = 0.050
    policy.complete_promotion("a")
    assert policy.estimate("restore", 1, 16) == pytest.approx(0.050)
    assert policy.choose("b", 1, 16, 16) == "recompute"
    clock.now = 0.060
    policy.complete_promotion("b", computed_tokens=8)
    assert "b" in policy.pending
    clock.now = 0.070
    policy.complete_promotion("b", computed_tokens=16)
    assert "b" not in policy.pending
    assert policy.estimate("recompute", 1, 16) == pytest.approx(0.020)
    # A different concurrency bucket falls back to explicit calibrated priors.
    assert policy.estimate("restore", 1, 16, (32, 4)) == 0.002


def test_cancelled_or_retried_decisions_do_not_train():
    policy, clock = make_policy()
    policy.choose("a", 1, 16, 16)
    clock.now = 1
    policy.choose("a", 1, 16, 16)
    assert policy.counters["restore_decisions"] == 1
    policy.complete_promotion("a", cancelled=True)
    assert not policy.costs and not policy.pending
    assert policy.counters["incomplete_promotions"] == 1


def test_guard_throttles_closes_and_recovers_with_bounded_probes():
    policy, clock = make_policy(
        guard_window_s=10,
        guard_min_samples=1,
        ewma_alpha=1,
        throttle_every=2,
        probe_every=2,
        probe_max_blocks=1,
    )
    policy._utility(-0.0011)
    clock.now = 10
    policy._guard_tick()
    assert policy.guard_state == "THROTTLE"
    policy._utility(-0.0011)
    clock.now = 20
    policy._guard_tick()
    assert policy.guard_state == "OFF"
    assert policy.prepare_store(hashes(1, 2)).block_hashes_to_store == []
    probe = policy.prepare_store(hashes(1, 2))
    assert probe.block_hashes_to_store == hashes(1)
    policy.complete_store(hashes(1))
    policy.choose("probe-hit", 1, 16, 16)
    clock.now += 0.001
    policy.complete_promotion("probe-hit")
    clock.now = 30
    policy._guard_tick()
    assert policy.guard_state == "THROTTLE"


def test_disabled_guard_and_admission_allow_all_blocks():
    policy, clock = make_policy(
        admission_enabled=False,
        guard_enabled=False,
        guard_min_samples=1,
        guard_window_s=1,
    )
    policy._utility(-100)
    clock.now = 2
    blocks = hashes(1, 2)
    assert policy.prepare_store(blocks).block_hashes_to_store == blocks
    assert policy.guard_state == "NORMAL"


def test_full_cpu_capacity_does_not_charge_failed_reservation():
    policy, _ = make_policy(admission_enabled=False)
    result = policy.prepare_store(hashes(*range(17)))
    assert result is None
    assert policy.counters["estimated_store_charge_s"] == 0


def test_replaced_promotion_target_discards_old_observation():
    policy, clock = make_policy()
    policy.choose("a", 1, 16, 16)
    clock.now = 1
    policy.choose("a", 2, 32, 32)
    assert policy.counters["incomplete_promotions"] == 1
    assert not policy.costs
    assert policy.pending["a"].started == 1


def test_reporting_reset_preserves_models_guard_and_probe_cadence():
    policy, _ = make_policy()
    policy.guard_state = "OFF"
    policy.prepare_store(hashes(1))
    policy.choose("a", 1, 16, 16)
    report = policy.get_value_policy_stats(reset=True)
    assert report["lcuc_pending"] == 1
    assert policy.guard_state == "OFF" and policy.admission_index == 1
    assert "a" in policy.pending and not policy.events


def test_history_and_event_memory_bounded():
    policy, _ = make_policy(
        max_tracker_size=2,
        max_decision_events=2,
        max_cost_buckets=2,
        min_cost_samples=1,
    )
    for i in range(10):
        policy.observe_request(str(i), hashes(i))
        policy.context = (2**i, 0)
        policy.choose(str(i), 1, 16, 16)
        policy.complete_promotion(str(i))
    assert len(policy.history) == len(policy.seen_requests) == 2
    assert len(policy.costs) == len(policy.events) == 2
    assert not policy.pending


def test_reuse_gap_decays_probability():
    policy, clock = make_policy()
    policy.observe_request("a", hashes(1))
    policy.observe_request("b", hashes(1))
    before = policy.probability(hashes(1)[0])
    clock.now = 600
    assert policy.probability(hashes(1)[0]) < before
