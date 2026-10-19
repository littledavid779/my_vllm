# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def mod(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    return importlib.import_module("fit_promotion_ttft")


def records(mod):
    return [
        dict(
            conversation=str(i),
            split=mod.group_split(str(i)),
            action=a,
            decision_to_first_token_ms=float(i % 20 + 1),
            **{f: i % 10 for f in mod.FEATURES},
        )
        for i in range(100)
        for a in mod.ACTIONS
    ]


def test_split_deterministic_and_shared_by_actions(mod):
    rows = records(mod)
    partitions = {
        s: {e["conversation"] for e in rows if e["split"] == s} for s in mod.SPLITS
    }
    assert all(partitions.values())
    assert not partitions["train"] & partitions["validate"]
    assert not partitions["train"] & partitions["calibrate"]
    assert not partitions["calibrate"] & partitions["validate"]
    assert mod.group_split("abc") == mod.group_split("abc")


def test_validation_labels_do_not_change_fit_or_margin(mod):
    rows = records(mod)
    a = mod.analyze(rows)
    for e in rows:
        if e["split"] == "validate":
            e["decision_to_first_token_ms"] += 1000
    b = mod.analyze(rows)
    for action in mod.ACTIONS:
        for kind in ("constant", "ridge"):
            x, y = (r["actions"][action]["models"][kind] for r in (a, b))
            assert x["model"] == y["model"]
            assert x["empirical_p90_margin_ms"] == y["empirical_p90_margin_ms"]
            assert x["validation"]["mae_ms"] != y["validation"]["mae_ms"]
    assert not b["deployment_approved"]


def test_constant_feature_and_missing_action_support(mod):
    rows = records(mod)
    for e in rows:
        e["pending_load_blocks"] = int(e["action"] == "restore")
    c = mod.coverage(rows)
    assert c["jointly_supported_events"] == 0
    assert c["pending_load_nonzero"]["recompute"] == 0
    m = mod.fit(rows[:5], "ridge")
    assert np.isfinite(mod.predict(m, rows)).all()
    with pytest.raises(ValueError):
        mod.fit([], "ridge")


def test_mixed_config_reproducible_and_fixed_admission(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    runner = importlib.import_module("run_promotion_ttft_calibration")
    data = [dict(id=str(i), messages=[dict(role="user")]) for i in range(100)]
    fit = dict(
        policy=dict(restore_s_per_block=0.002, recompute_s_per_token=0.001),
        store_cost_s_per_block={"host-service": 0.01},
    )
    c = runner.mixed_config(data, fit, {}, 17)
    assert c == runner.mixed_config(data, fit, {}, 17)
    assert c != runner.mixed_config(data, fit, {}, 29)
    extra = c["kv_connector_extra_config"]
    assert set(extra["calibration_action_assignments"].values()) == {
        "restore",
        "recompute",
    }
    assert len(extra["calibration_action_assignments"]) == 100
    assert not extra["lifecycle_policy"]["admission_enabled"]
    assert not extra["lifecycle_policy"]["guard_enabled"]
