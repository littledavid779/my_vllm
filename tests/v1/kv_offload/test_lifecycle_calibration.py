# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from scripts.run_lifecycle_calibrated_smoke import (
    disjoint_candidates,
    evaluation_config,
    fit_nonnegative_line,
)


@pytest.mark.parametrize(
    "arm,cost,probe",
    [
        ("completion-no-probe", 0.5, False),
        ("completion-probe", 0.5, True),
        ("host-no-probe", 0.01, False),
        ("host-probe", 0.01, True),
    ],
)
def test_cost_probe_ablation_is_independent(arm, cost, probe):
    fit = {
        "policy": {
            "store_s_per_block": 0.01,
            "restore_s_per_block": 0.002,
            "recompute_s_per_token": 0.001,
        },
        "store_cost_s_per_block": {"host-service": 0.01, "worker-completion": 0.5},
    }
    cfg = evaluation_config(arm, fit, {"cold_probe_every": 8})[
        "kv_connector_extra_config"
    ]
    assert cfg["promotion_policy"] == "cost_aware"
    policy = cfg["lifecycle_policy"]
    assert policy["store_s_per_block"] == cost
    assert policy.get("cold_probe_every", 0) == (8 if probe else 0)
    assert policy["admission_enabled"] and not policy["guard_enabled"]
    assert fit["policy"]["store_s_per_block"] == 0.01


def test_affine_cost_fit():
    a, b = fit_nonnegative_line([(1, 3), (2, 5), (3, 7)])
    assert a == pytest.approx(1)
    assert b == pytest.approx(2)


def test_cost_fit_boundaries():
    assert fit_nonnegative_line([(1, 3), (2, 2), (3, 1)]) == (2, 0)
    a, b = fit_nonnegative_line([(1, 1), (2, 3), (3, 5)])
    assert a == 0
    assert b > 0
    assert fit_nonnegative_line([(2, 4), (2, 4)])[0] >= 0


@pytest.mark.parametrize("points", [[], [(0, 1)], [(1, -1)], [(1, float("nan"))]])
def test_cost_fit_rejects_bad_data(points):
    with pytest.raises(ValueError):
        fit_nonnegative_line(points)


def test_disjoint_by_id_and_content():
    source = [
        {"id": "a", "messages": [1]},
        {"id": "b", "messages": [2]},
        {"id": "c", "messages": [3]},
    ]
    excluded = [{"id": "a", "messages": [9]}, {"id": "x", "messages": [2]}]
    assert disjoint_candidates(source, excluded) == source[2:]
