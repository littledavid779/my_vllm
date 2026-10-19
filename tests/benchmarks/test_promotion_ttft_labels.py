# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
from pathlib import Path

import pytest

from tests.benchmarks.test_lifecycle_request_path_analysis import rows, stats


@pytest.fixture
def mod(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    return importlib.import_module("analyze_promotion_ttft_labels")


def sample(action="restore"):
    s = stats(cached=80 if action == "restore" else 16)
    s.update(
        promotion_ttft_trace_enabled=1,
        promotion_ttft_trace_recorded=1,
        offload_promotion_decision_count=1,
        promotion_ttft_trace_events=[
            dict(
                event_id=0,
                request_id="chatcmpl-conv-turn3-abcdef12",
                action=action,
                decision_recorded_s=0.002,
                candidate_blocks=4,
                candidate_tokens=64,
                gpu_hit_tokens=16,
                prompt_tokens=100,
                tracked_requests=2,
                pending_load_requests=1,
                pending_store_requests=1,
                pending_load_blocks=2,
                pending_store_blocks=3,
                bytes_per_offloaded_block=1024,
                candidate_bytes=4096,
                pending_load_bytes=2048,
                pending_store_bytes=3072,
            )
        ],
    )
    return s


@pytest.mark.parametrize("action", ["restore", "recompute"])
def test_same_endpoint_both_actions(mod, action):
    a = mod.build_labels(rows(), sample(action))
    e = a["labels"][0]
    assert e["decision_to_first_token_ms"] == pytest.approx(13)
    assert e["remaining_prefill_if_restore"] == 20
    assert e["remaining_prefill_if_recompute"] == 84
    assert e["label_kind"] == "observed_action_only"
    assert a["by_action"] == {action: 1}


def test_multiple_decisions_excluded_not_overwritten(mod):
    s = sample()
    s["promotion_ttft_trace_events"].append(
        dict(s["promotion_ttft_trace_events"][0], event_id=1)
    )
    s["promotion_ttft_trace_recorded"] = s["offload_promotion_decision_count"] = 2
    a = mod.build_labels(rows(), s)
    assert a["label_count"] == 0
    assert a["excluded"][0]["reason"] == "multiple_decisions"


def test_calibration_configs_only_change_forced_action(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    runner = importlib.import_module("run_promotion_ttft_calibration")
    fit = dict(
        policy=dict(restore_s_per_block=0.002, recompute_s_per_token=0.001),
        store_cost_s_per_block={"host-service": 0.01},
    )
    r = runner.calibration_config("restore", fit, {})
    c = runner.calibration_config("recompute", fit, {})
    extra = r["kv_connector_extra_config"]
    assert extra["cpu_bytes_to_use"] == 2 * 1024**3
    assert extra["track_promotion_ttft"]
    assert not extra["track_cost_events"]
    assert not extra["lifecycle_policy"]["admission_enabled"]
    assert not extra["lifecycle_policy"]["guard_enabled"]
    extra["promotion_policy"] = "always_recompute"
    assert r == c
    with pytest.raises(ValueError):
        runner.calibration_config("cost_aware", fit, {})


@pytest.mark.parametrize(
    "fault", ["dropped", "count", "id", "time", "feature", "bytes", "prefix", "action"]
)
def test_corrupt_labels_rejected(mod, fault):
    s = sample()
    e = s["promotion_ttft_trace_events"][0]
    if fault == "dropped":
        s["promotion_ttft_trace_dropped"] = 1
    elif fault == "count":
        s["promotion_ttft_trace_recorded"] = 2
    elif fault == "id":
        e["request_id"] = "unknown"
    elif fault == "time":
        e["decision_recorded_s"] = 10
    elif fault == "feature":
        e["candidate_bytes"] = None
    elif fault == "bytes":
        e["candidate_bytes"] = 12
    elif fault == "prefix":
        e["gpu_hit_tokens"] = 0
    else:
        e["action"] = "recompute"
    with pytest.raises(ValueError):
        mod.build_labels(rows(), s)
