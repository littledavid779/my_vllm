"""Reject factor leakage, incomplete question spans, and invalid audit evidence."""

import ast
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from apply_precheck_kv_patch import RELATIVE, commit_precheck  # noqa: E402
from apply_query_control_patch import HELPER as PREVIOUS_HELPER  # noqa: E402
from apply_query_template_patch import (  # noqa: E402
    HELPER,
    MODES,
    TEMPLATE,
    template_controls,
)
from run_query_template import audit  # noqa: E402

SCOPE = {"torch": torch}
exec(compile(PREVIOUS_HELPER + HELPER, "test-query-template", "exec"), SCOPE)


def test_span_partition_preserves_question_and_all_template_tokens():
    q = [14582, 25, 2160, 3757, 304, 279, 50802, 30, 220]
    tail = [7, 8, 9] + q + TEMPLATE
    assert SCOPE["query_template_spans"](tail) == (9, 9)
    assert sum(SCOPE["query_template_spans"](tail)) == SCOPE["query_control_span"](tail)
    assert SCOPE["query_template_spans"]([7, 8, 9]) == (0, 0)


@pytest.mark.parametrize(
    "suffix", [TEMPLATE[:-1], TEMPLATE + [7], TEMPLATE[1:], [151645] + TEMPLATE]
)
def test_reject_ambiguous_or_incomplete_template(suffix):
    with pytest.raises(ValueError):
        SCOPE["query_template_spans"]([14582, 25, 2160] + suffix)


@pytest.mark.parametrize("mode", MODES)
def test_unselected_region_cannot_leak_through_topk(mode):
    scores = torch.arange(2048).float()
    before = scores.clone()
    selected, forced = SCOPE["select_query_template"](scores, 204, mode, 9, 9)
    values = selected.tolist()
    assert len(values) == len(set(values)) == 204
    q, t = set(range(2030, 2039)), set(range(2039, 2048))
    if mode == "topk":
        assert q | t <= set(values) and forced == 0
    else:
        assert set(values) & q == (q if mode in ("question", "both") else set())
        assert set(values) & t == (t if mode in ("template", "both") else set())
    assert torch.equal(before, scores)


def test_both_matches_previous_query_control_and_short_requests_unchanged():
    scores = torch.randperm(2048, generator=torch.Generator().manual_seed(19)).float()
    current, _ = SCOPE["select_query_template"](scores, 204, "both", 9, 9)
    previous, _ = SCOPE["select_query_control"](scores, 204, "query", 18)
    assert set(current.tolist()) == set(previous.tolist())
    for mode in MODES:
        selected, count = SCOPE["select_query_template"](scores[:37], 3, mode, 0, 0)
        assert set(selected.tolist()) == set(scores[:37].topk(3).indices.tolist())
        assert count == 0


@pytest.mark.parametrize(
    "budget,mode,q,t",
    [
        (2, "both", 9, 9),
        (2040, "none", 9, 9),
        (204, "question", 0, 9),
        (204, "unknown", 9, 9),
    ],
)
def test_reject_infeasible_budget_and_partial_spans(budget, mode, q, t):
    with pytest.raises(ValueError):
        SCOPE["select_query_template"](torch.zeros(2048), budget, mode, q, t)


def test_patch_compiles_against_pinned_source():
    project = Path(__file__).resolve().parents[1]
    source = (project / "upstream/lmcache" / RELATIVE).read_text()
    ast.parse(template_controls(commit_precheck(source)))


def test_bit_audit_detects_leakage_duplicates_and_wrong_boundaries(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps({"status": "completed", "config": {"query_template": "question"}})
    )
    entry = dict(
        mode="question",
        total=2048,
        selected=204,
        unique=204,
        min_index=1826,
        max_index=2038,
        forced=9,
        question_tokens=9,
        template_tokens=9,
        question_bits=511,
        template_bits=0,
        prompt_sha="abc",
        selected_sha="def",
    )
    coverage = {
        "prompts": [
            dict(
                session_id="one",
                version=1,
                length=2048,
                question_tokens=9,
                template_tokens=9,
                prompt_sha="abc",
            )
        ]
    }
    log = tmp_path / "cacheblend10.log"

    def write(item):
        log.write_text("QUERY_TEMPLATE " + json.dumps(item) + "\n")

    write(entry)
    audit(tmp_path, "question", coverage)
    for patch in (
        {"template_bits": 1},
        {"question_bits": 510},
        {"unique": 203},
        {"prompt_sha": "missing"},
        {"question_tokens": 10, "question_bits": 1023, "forced": 10},
    ):
        write({**entry, **patch})
        with pytest.raises(ValueError):
            audit(tmp_path, "question", coverage)
    write(entry)
    log.write_text(log.read_text() * 2)
    with pytest.raises(ValueError, match="duplicated"):
        audit(tmp_path, "question", coverage)
