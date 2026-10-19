# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
from pathlib import Path

import pytest


def test_factorial_contrasts_and_uncertainty(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    summary = importlib.import_module("summarize_lifecycle_decision_ablation")
    values = {
        "all-block": 9,
        "all-block-control": 10,
        "admission-only": 12,
        "promotion-only": 13,
        "joint-no-probe": 18,
    }
    rows = {k: {"mean_ttft_ms": v} for k, v in values.items()}
    assert summary.contrasts(rows) == {
        "controller_path": 1,
        "admission_only": 2,
        "promotion_only": 3,
        "joint": 8,
        "interaction": 3,
        "admission_given_promotion": 5,
        "promotion_given_admission": 6,
    }
    assert summary.mean_sd([1])["sample_sd"] is None
    assert summary.mean_sd([1, 2, 3]) == {"mean": 2, "sample_sd": pytest.approx(1)}
