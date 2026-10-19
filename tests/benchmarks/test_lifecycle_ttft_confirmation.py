# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
from collections import Counter
from pathlib import Path

import pytest


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    return importlib.import_module("run_lifecycle_ttft_confirmation")


def test_orders_balance_positions_and_pairs(runner):
    orders = runner.balanced_orders()
    assert len({tuple(x) for x in orders}) == 6
    for position in range(3):
        assert Counter(x[position] for x in orders) == dict.fromkeys(runner.ARMS, 2)
    for a in runner.ARMS:
        for b in runner.ARMS:
            if a != b:
                assert sum(x.index(a) < x.index(b) for x in orders) == 3


def test_confirmation_disables_detailed_traces_not_feedback(runner):
    fit = dict(
        policy=dict(restore_s_per_block=0.002, recompute_s_per_token=0.001),
        store_cost_s_per_block={"host-service": 0.01},
    )
    a = runner.config("all-block-2g", fit, {})
    b = runner.config("selective-2g", fit, {})
    extra = a["kv_connector_extra_config"]
    assert extra["cpu_bytes_to_use"] == 2 * 1024**3
    assert not extra["track_cost_events"]
    assert not extra["lifecycle_policy"]["trace_admission"]
    assert extra["promotion_policy"] == "always_restore"
    extra["lifecycle_policy"]["admission_enabled"] = True
    assert a == b
    assert runner.config("no-offload", fit, {}) is None


def test_partial_summary_does_not_invent_pairs(runner):
    result = runner.summarize([])
    assert result["paired_run_differences"] == []
    assert not result["complete"]
