"""Budget, position, and reference alignment regressions for the fair screen."""

import ast
import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_query_fair import summary  # noqa: E402
from apply_precheck_kv_patch import RELATIVE, commit_precheck  # noqa: E402
from apply_query_fair_patch import ALL_HELPERS, SEPARATOR, fair_controls  # noqa: E402
from run_query_fair import audit  # noqa: E402

SCOPE = {"torch": torch}
exec(compile(ALL_HELPERS, "test-query-fair", "exec"), SCOPE)


def test_boundary_equal_depth_and_short_chunk_redistribution():
    # Two short chunks, plus two query tokens. Exhaust first, continue second.
    selected = SCOPE["boundary_positions"](20, 10, 2, [5, 8])
    assert selected == list(range(5, 13)) + [18, 19]
    assert SCOPE["boundary_positions"](20, 6, 2, [5, 12]) == [5, 6, 12, 13, 18, 19]
    assert SCOPE["boundary_positions"](20, 8, 2, [15, 17]) == [
        0,
        1,
        2,
        15,
        16,
        17,
        18,
        19,
    ]


@pytest.mark.parametrize(
    "budget,query,starts",
    [(1, 2, [5]), (21, 2, [5]), (6, 2, [8, 5]), (6, 2, [5, 5]), (6, 2, [])],
)
def test_boundary_rejects_incomplete_or_invalid_selection(budget, query, starts):
    with pytest.raises(ValueError):
        SCOPE["boundary_positions"](20, budget, query, starts)


def test_separator_positions_are_delimiters_not_content_offsets():
    ids = [1] * 30 + SEPARATOR + [2] * 20 + SEPARATOR + [3] * 100
    assert SCOPE["fair_boundaries"](ids) == [30, 60]


@pytest.mark.parametrize("mode", ["query", "tail64", "boundary"])
def test_complete_query_same_budget_no_score_mutation(mode):
    scores = torch.arange(2048, dtype=torch.float32)
    before = scores.clone()
    positions, _ = SCOPE["select_query_fair"](scores, 204, mode, 19, [200, 240])
    assert positions.numel() == positions.unique().numel() == 204
    assert set(range(2029, 2048)) <= set(positions.tolist())
    assert torch.equal(before, scores)
    short, forced = SCOPE["select_query_fair"](scores[:37], 3, mode, 0, [])
    assert set(short.tolist()) == {34, 35, 36} and forced == 0


def test_generated_patch_compiles_against_locked_source():
    project = Path(__file__).resolve().parents[1]
    source = (project / "upstream/lmcache" / RELATIVE).read_text()
    modified = fair_controls(commit_precheck(source))
    ast.parse(modified)
    assert modified.count("selected_sha =") == 1
    assert "self.query_fair_starts" in modified


def test_reference_correctness_uses_same_repeat():
    rows = [
        dict(
            repeat=i,
            session_id="one",
            version=1,
            source_id="source",
            task_correct=True,
            client_ttft_ms=10.0,
        )
        for i in (0, 1)
    ]
    refs = {(i, "one", 1): {"task_correct": i == 0} for i in (0, 1)}
    result = summary(rows, refs)
    assert result["reference_correct_observations"] == 1
    assert result["correct_when_full_wrong"] == 1


def test_audit_checks_exact_boundary_positions_not_only_count(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps({"status": "completed", "config": {"query_control": "boundary"}})
    )
    selection_sha = hashlib.sha256(b"correct positions").hexdigest()
    prompt_sha = "a" * 64
    first = (
        "QUERY_CONTROL mode=boundary total=2048 selected=204 forced=204 "
        "query=19 qcovered=19 tailcovered=19 signature=abcdef12\n"
    )
    second = (
        f"QUERY_FAIR mode=boundary promptsha={prompt_sha} selectedsha={selection_sha}\n"
    )
    coverage = {
        "prompts": [
            {
                "version": 1,
                "length": 2048,
                "query_length": 19,
                "signature": "abcdef12",
                "prompt_sha": prompt_sha,
                "boundary_selection_sha": selection_sha,
            }
        ]
    }
    log = tmp_path / "cacheblend10.log"
    log.write_text(first + second)
    audit(tmp_path, "boundary", coverage)
    log.write_text(first + second.replace(selection_sha, "b" * 64))
    with pytest.raises(ValueError, match="boundary positions"):
        audit(tmp_path, "boundary", coverage)
    log.write_text(first + second + second)
    with pytest.raises(ValueError, match="duplicated"):
        audit(tmp_path, "boundary", coverage)
