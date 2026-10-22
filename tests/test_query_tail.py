"""Verify total budget, unique positions, full tail, and an unchanged control."""

import ast
import sys
from pathlib import Path

import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
from apply_query_tail_patch import HELPER  # noqa: E402

scope = {"torch": torch}
exec(compile(ast.parse(HELPER), "query-tail-selector", "exec"), scope)
select = scope["select_with_query_tail"]


def test_zero_tail_is_exact_original_topk():
    torch.manual_seed(22)
    for length in (17, 1024, 8044):
        values = torch.randn(length)
        budget = max(1, length // 10)
        indices, forced = select(values, budget, 0)
        assert forced == 0
        assert torch.equal(indices, torch.topk(values, budget).indices)


def test_tail_reserves_budget_and_keeps_best_remaining_positions():
    # Make the tail the least-important positions to ensure it is forced.
    values = torch.arange(2048, 0, -1).float()
    original = values.clone()
    indices, forced = select(values, 204, 64)
    assert forced == 64
    assert indices.numel() == indices.unique().numel() == 204
    assert set(indices.tolist()) == set(range(140)) | set(range(1984, 2048))
    assert torch.equal(values, original)


def test_small_budget_and_short_preparation_never_exceed_budget():
    indices, forced = select(torch.arange(1024).float(), 7, 64)
    assert forced == 7
    assert indices.tolist() == list(range(1017, 1024))
    scores = torch.arange(37, 0, -1).float()
    indices, forced = select(scores, 3, 64)
    assert forced == 0
    assert indices.tolist() == [0, 1, 2]


def test_all_positions_budget_and_invalid_configuration():
    indices, forced = select(torch.randn(1024), 1024, 64)
    assert forced == 64
    assert indices.unique().numel() == 1024
    with pytest.raises(ValueError):
        select(torch.ones(10), 0, 64)
    with pytest.raises(ValueError):
        select(torch.ones(10), 1, 65)
