# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import csv
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    return importlib.import_module("run_lifecycle_fixed_workload")


def conversations():
    return [
        {"id": "a", "messages": [{"role": "user", "content": "x"}]},
        {
            "id": "b",
            "messages": [
                {"role": "user", "content": "x"},
                {"role": "assistant", "content": "y"},
                {"role": "user", "content": "z"},
            ],
        },
    ]


@pytest.mark.parametrize(
    "arm,admission,promotion",
    [
        ("all-block-control", False, "always_restore"),
        ("admission-only", True, "always_restore"),
        ("promotion-only", False, "cost_aware"),
        ("joint-no-probe", True, "cost_aware"),
    ],
)
def test_decision_ablation_changes_only_two_switches(runner, arm, admission, promotion):
    fit = {
        "policy": {
            "store_s_per_block": 0.5,
            "restore_s_per_block": 0.002,
            "recompute_s_per_token": 0.001,
        },
        "store_cost_s_per_block": {"host-service": 0.01},
    }
    cfg = runner.suite_config("decision-ablation", arm, fit, {"cold_probe_every": 8})
    extra = cfg["kv_connector_extra_config"]
    assert extra["promotion_policy"] == promotion
    assert extra["lifecycle_policy"] == dict(
        fit["policy"],
        store_s_per_block=0.01,
        admission_enabled=admission,
        guard_enabled=False,
        cold_probe_every=0,
    )
    assert fit["policy"]["store_s_per_block"] == 0.5


def test_decision_ablation_native_baselines_and_unknown_arm(runner):
    assert runner.suite_config("decision-ablation", "no-offload", {}, {}) is None
    cfg = runner.suite_config("decision-ablation", "all-block", {}, {})
    assert "lifecycle_policy" not in cfg["kv_connector_extra_config"]
    with pytest.raises(ValueError, match="Unknown"):
        runner.suite_config("decision-ablation", "typo", {}, {})


def test_admission_diagnostic_is_single_switch_ablation(runner):
    fit = {
        "policy": {"restore_s_per_block": 0.002, "recompute_s_per_token": 0.001},
        "store_cost_s_per_block": {"host-service": 0.01},
    }
    old = runner.suite_config("admission-diagnostic", "joint-no-probe", fit, {})
    new = runner.suite_config("admission-diagnostic", "joint-new-write", fit, {})
    old_policy = old["kv_connector_extra_config"]["lifecycle_policy"]
    assert old_policy["trace_admission"]
    assert not old_policy["guard_enabled"] and old_policy["cold_probe_every"] == 0
    old_policy["new_write_store_cost"] = True
    assert old == new
    with pytest.raises(ValueError, match="Unknown"):
        runner.suite_config("admission-diagnostic", "typo", fit, {})


def test_expected_requests_with_final_user(runner):
    assert runner.expected_requests(conversations()) == {
        ("a", "1", "0"): 1,
        ("b", "1", "1"): 1,
        ("b", "3", "1"): 1,
    }


@pytest.mark.parametrize("gib", [2, 4])
def test_capacity_suite_is_admission_only(runner, gib):
    fit = {
        "policy": {"restore_s_per_block": 0.002, "recompute_s_per_token": 0.001},
        "store_cost_s_per_block": {"host-service": 0.01},
    }
    all_cfg = runner.suite_config("admission-capacity", f"all-block-{gib}g", fit, {})
    selective = runner.suite_config("admission-capacity", f"selective-{gib}g", fit, {})
    extra = all_cfg["kv_connector_extra_config"]
    assert extra["cpu_bytes_to_use"] == gib * 1024**3
    assert extra["promotion_policy"] == "always_restore"
    policy = extra["lifecycle_policy"]
    assert policy["new_write_store_cost"] and policy["trace_admission"]
    assert not policy["guard_enabled"] and policy["cold_probe_every"] == 0
    assert not policy["admission_enabled"]
    policy["admission_enabled"] = True
    assert all_cfg == selective
    assert runner.suite_config("admission-capacity", "no-offload", fit, {}) is None
    with pytest.raises(ValueError, match="Unknown"):
        runner.suite_config("admission-capacity", "wrong", fit, {})


def test_expected_rejects_duplicate_or_invalid_roles(runner):
    data = conversations()
    with pytest.raises(ValueError, match="Duplicate"):
        runner.expected_requests(data + data[:1])
    data[0]["messages"][0]["role"] = "assistant"
    with pytest.raises(ValueError, match="Non-alternating"):
        runner.expected_requests(data)


@pytest.mark.parametrize("fault", [None, "missing", "client", "budget", "incomplete"])
def test_fixed_workload_audit(runner, tmp_path, fault):
    data = conversations()
    expected = runner.expected_requests(data)
    rows = [
        {
            "conversation_id": c,
            "input_num_turns": t,
            "client_id": client,
            "input_num_tokens": 5,
            "start_time_ms": 0,
            "latency_ms": 100,
        }
        for c, t, client in expected
    ]
    if fault == "missing":
        rows.pop()
    if fault == "client":
        rows[0]["client_id"] = "3"
    with (tmp_path / "request_summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    stats_path = tmp_path / "value_policy_stats.json"
    stats_path.write_text(
        json.dumps(
            {"stats": {"lcuc_cold_probe_blocks_spent": 129 if fault == "budget" else 0}}
        )
    )
    (tmp_path / "output.json").write_text(
        json.dumps(data[:1] if fault == "incomplete" else data)
    )
    metrics = {"stats_path": str(stats_path)}
    probe = {"cold_probe_budget_blocks": 128, "cold_probe_budget_s": 0.1}
    if fault is None:
        assert len(runner.audit_arm(metrics, expected, probe)) == 3
        assert metrics["requests_per_s"] == 30
    else:
        with pytest.raises(ValueError):
            runner.audit_arm(metrics, expected, probe)
