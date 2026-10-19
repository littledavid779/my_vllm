# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
from copy import deepcopy
from pathlib import Path

import pytest


@pytest.fixture
def mod(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    return importlib.import_module("diagnose_promotion_ttft_errors")


def sample():
    e = dict(
        arm="run",
        request_id="r",
        decision_recorded_s=1,
        decision_to_first_token_ms=20,
        action="restore",
        prompt_tokens=100,
        gpu_hit_tokens=16,
        candidate_tokens=64,
    )
    p = dict(
        first_compute_s=1.005,
        first_output_s=1.020,
        first_compute_cached_tokens=80,
        completed_prefill_tokens=20,
        prefill_steps=1,
    )
    return e, p


def test_time_partition(mod):
    e, p = sample()
    r = mod.attach_paths([e], {("run", "r"): p})[0]
    assert r["before_schedule_ms"] == pytest.approx(5)
    assert r["after_schedule_ms"] == pytest.approx(15)


@pytest.mark.parametrize("fault", ["time", "nan", "sum", "cache"])
def test_invalid_partition_rejected(mod, fault):
    e, p = sample()
    if fault == "time":
        p["first_compute_s"] = 0.5
    elif fault == "nan":
        p["first_compute_s"] = float("nan")
    elif fault == "sum":
        e["decision_to_first_token_ms"] = 50
    else:
        p["first_compute_cached_tokens"] = 16
    with pytest.raises(ValueError):
        mod.attach_paths([e], {("run", "r"): p})


def records():
    f = importlib.import_module("fit_promotion_ttft")
    return [
        dict(
            conversation=str(i),
            split=f.group_split(str(i)),
            action=a,
            decision_to_first_token_ms=float(i % 20 + 3),
            before_schedule_ms=1.0,
            after_schedule_ms=float(i % 20 + 2),
            **{k: i % 10 for k in f.FEATURES},
        )
        for i in range(100)
        for a in f.ACTIONS
    ]


def test_signed_error_partition_and_no_validation_refitting(mod):
    f = importlib.import_module("fit_promotion_ttft")
    rows = records()
    frozen = f.analyze(rows)
    before = deepcopy(frozen)
    output = mod.diagnose(rows, frozen)
    for e in output:
        assert e["signed_error_ms"] == pytest.approx(
            e["before_error_ms"] + e["after_error_ms"] + e["clipping_adjustment_ms"]
        )
    for e in rows:
        if e["split"] == "validate":
            e["after_schedule_ms"] += 100
            e["decision_to_first_token_ms"] += 100
    changed = mod.diagnose(rows, frozen)
    assert frozen == before
    for e, c in zip(output, changed):
        assert e["predicted_ms"] == c["predicted_ms"]
        assert (
            e["diagnostic_after_prediction_ms"] == c["diagnostic_after_prediction_ms"]
        )
        if e["split"] == "validate":
            assert c["after_error_ms"] == pytest.approx(e["after_error_ms"] - 100)


def test_changed_frozen_coefficients_rejected(mod):
    f = importlib.import_module("fit_promotion_ttft")
    rows = records()
    frozen = f.analyze(rows)
    frozen["actions"]["restore"]["models"]["ridge"]["model"]["intercept"] += 1
    with pytest.raises(ValueError, match="Frozen model"):
        mod.diagnose(rows, frozen)
