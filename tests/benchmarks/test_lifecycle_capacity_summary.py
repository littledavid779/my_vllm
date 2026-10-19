# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
from pathlib import Path

import pytest


def test_capacity_contrasts_only_match_repeat(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    module = importlib.import_module("summarize_lifecycle_capacity")

    def row(repeat, arm, p95):
        return dict(
            repeat=repeat,
            arm=arm,
            mean_ttft_ms=20,
            p95_ttft_ms=p95,
            requests_per_s=10,
            store_blocks=10,
            load_blocks=5,
        )

    result = module.aggregate(
        [
            row(0, "no-offload", 50),
            row(0, "all-block-2g", 45),
            row(0, "selective-2g", 48),
            row(1, "selective-2g", 46),
        ]
    )
    contrasts = result["paired_run_contrasts"]
    assert contrasts["selective-2g_minus_all-block-2g"] == [
        dict(
            repeat=0,
            p95_ttft_delta_ms=3,
            mean_ttft_delta_ms=0,
            throughput_change_pct=0,
        )
    ]
    assert contrasts["selective-2g_minus_no-offload"][0]["p95_ttft_delta_ms"] == -2
    assert result["by_arm"]["selective-2g"]["p95_ttft_ms"]["mean"] == 47
    assert result["by_arm"]["selective-2g"]["p95_ttft_ms"][
        "sample_sd"
    ] == pytest.approx(2**0.5)
