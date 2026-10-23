"""Mechanism controls must change reserved locations without buying more tokens."""

import ast
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from apply_query_control_patch import HELPER, MODES  # noqa: E402

scope = {"torch": torch}
exec(compile(ast.parse(HELPER), "query-controls", "exec"), scope)
select = scope["select_query_control"]
query_span = scope["query_control_span"]


@pytest.mark.parametrize("mode", MODES)
def test_budget_unique_and_scores_unchanged(mode):
    scores = torch.arange(2048, 0, -1).float()
    original = scores.clone()
    indices, forced = select(scores, 204, mode, 19)
    assert indices.numel() == indices.unique().numel() == 204
    assert torch.equal(scores, original)
    assert forced == {"topk": 0, "query": 19}.get(mode, 64)


def test_mandatory_and_remaining_best_positions():
    scores = torch.arange(2048, 0, -1).float()
    for mode, region in (
        ("query", set(range(2029, 2048))),
        ("tail64", set(range(1984, 2048))),
        ("pre_tail64", set(range(1920, 1984))),
    ):
        indices, _ = select(scores, 204, mode, 19)
        assert set(indices.tolist()) == region | set(range(204 - len(region)))


def test_random_reproducible_and_excludes_query_tail():
    scores = torch.arange(2048, 0, -1).float()
    a, _ = select(scores, 204, "random64", 19)
    b, _ = select(scores, 204, "random64", 19)
    c, _ = select(scores, 204, "random64", 19, seed=99)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert bool((a[-64:] < 1984).all())
    mandatory = set(a[-64:].tolist())
    remaining = [i for i in range(2048) if i not in mandatory][:140]
    assert set(a.tolist()) == mandatory | set(remaining)


@pytest.mark.parametrize("mode", MODES)
def test_short_preparation_equals_original(mode):
    scores = torch.arange(37).float()
    indices, forced = select(scores, 3, mode, 0)
    assert forced == 0
    assert torch.equal(indices, torch.topk(scores, 3).indices)


def test_zero_mode_and_all_positions_and_small_budget():
    scores = torch.arange(1024).float()
    indices, _ = select(scores, 102, "topk", 19)
    assert torch.equal(indices, torch.topk(scores, 102).indices)
    for mode in MODES:
        indices, _ = select(scores, 1024, mode, 19)
        assert indices.unique().numel() == 1024
    for mode in ("tail64", "pre_tail64", "random64"):
        indices, forced = select(scores, 7, mode, 0)
        assert indices.unique().numel() == forced == 7
    with pytest.raises(ValueError):
        select(scores, 7, "query", 19)
    with pytest.raises(ValueError):
        select(scores, 102, "invalid", 19)


def test_prompt_markers_and_last_question():
    assert query_span([1, 2, 3]) == 0  # disjoint synthetic warmup
    assert query_span([4, 14582, 25, 7, 151644, 77091, 9]) == 6
    assert query_span([14582, 25, 7, 14582, 25, 8, 151644, 77091, 9]) == 6
    with pytest.raises(ValueError):
        query_span([14582, 25, 7, 9])
