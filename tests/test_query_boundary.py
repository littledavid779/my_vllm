"""Check native query exclusion, contextual cache keys, and budget equivalence."""

import ast
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from apply_query_boundary_patch import HELPER, MODES, plans  # noqa: E402
from build_query_boundary_input import ROOM_LABELS, paraphrase  # noqa: E402
from run_query_boundary import audit  # noqa: E402

SCOPE = {}
exec(compile(HELPER, "test-boundary", "exec"), SCOPE)
TAIL = [14582, 25, 7, 8] + SCOPE["TEMPLATE"]


@pytest.mark.parametrize("mode", MODES)
def test_segment_boundaries_and_contextual_question_keys(mode):
    tokens = [11, 12, 13] + TAIL
    chunks = SCOPE["split_query_segment"](tokens, 100, mode)
    assert chunks[0] == (100, 103, [11, 12, 13])
    if mode == "fresh":
        assert len(chunks) == 1
    else:
        assert chunks[1] == (103, 100 + len(tokens), tokens)
        alternate = SCOPE["split_query_segment"]([31, 32, 33] + TAIL, 100, mode)
        assert chunks[1][2] != alternate[1][2]
    assert tokens == [11, 12, 13] + TAIL


def test_truncated_context_retrieves_the_same_context_key():
    tokens = [11] * 1024 + TAIL
    stored = SCOPE["split_query_segment"](tokens, 0, "fresh")
    retrieved = SCOPE["split_query_segment"](tokens[:1024], 0, "fresh")
    assert stored == retrieved
    for mode in MODES:
        assert SCOPE["split_query_segment"]([1, 2, 3], 9, mode) == [(9, 12, [1, 2, 3])]


def test_decode_suffix_does_not_make_query_cacheable():
    tokens = [11] * 1024 + TAIL + [5] * 64
    assert SCOPE["query_offset"](tokens) == 1024
    assert SCOPE["split_query_segment"](tokens, 0, "fresh") == [(0, 1024, [11] * 1024)]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("ratio", [0.1, 1.0])
def test_selection_matches_independent_rank_oracle(mode, ratio):
    total, span = 2048, 18
    n = total - span if mode == "fresh" else total
    scores = torch.randperm(n, generator=torch.Generator().manual_seed(55)).float()
    before = scores.clone()
    info = {
        "mode": mode,
        "total": total,
        "query_start": total - span,
        "query_tokens": span,
    }
    positions, forced = SCOPE["select_boundary"](scores, ratio, info)
    budget = int(total * ratio)
    mandatory = set(range(total - span, total)) if mode == "query" else set()
    target = budget - (span if mode == "fresh" else 0)
    remaining = sorted(
        (i for i in range(n) if i not in mandatory),
        key=lambda i: float(scores[i]),
        reverse=True,
    )[: target - len(mandatory)]
    assert set(positions.tolist()) == mandatory | set(remaining)
    assert len(positions) == len(set(positions.tolist())) == target
    assert forced == len(mandatory)
    assert torch.equal(scores, before)


def test_refuse_blending_native_query_and_infeasible_budgets():
    info = {"mode": "fresh", "total": 2048, "query_start": 2030, "query_tokens": 18}
    with pytest.raises(ValueError):
        SCOPE["select_boundary"](torch.zeros(2048), 0.1, info)
    with pytest.raises(ValueError):
        SCOPE["select_boundary"](torch.zeros(2030), 0.001, info)


def test_patch_compiles_and_native_count_is_observed():
    transformed = plans()
    for _, text in transformed.values():
        ast.parse(text)
    adapter = transformed["lmcache/integration/vllm/vllm_v1_adapter.py"][1]
    assert "am.num_actual_tokens" in adapter
    assert "boundary_info=info" in adapter


def test_prompt_variants_preserve_question_entities():
    assert (
        paraphrase("Is Mary in the garden? ", "qa9")
        == "Is it true that Mary is in the garden?"
    )
    assert paraphrase("Where is the milk? ", "qa2") == "Which room is the milk in?"
    assert (
        paraphrase("Where was the milk before the kitchen? ", "qa3")
        == "Before the kitchen, where was the milk?"
    )
    assert len(ROOM_LABELS) == len(set(ROOM_LABELS.values())) == 6


def test_audit_rejects_query_loading_or_missing_native_compute(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps({"status": "completed", "config": {"query_boundary": "fresh"}})
    )
    prompt = {
        "prompt_sha": "s",
        "total": 2048,
        "query_start": 2030,
        "query_tokens": 18,
        "session_id": "x",
        "version": 1,
    }
    good = {
        **prompt,
        "mode": "fresh",
        "hit_tokens": 2030,
        "native_prompt_tokens": 18,
        "actual_native_tokens": 18,
        "selected": 186,
        "forced": 0,
        "unique": 186,
        "blend_tokens": 2030,
        "min_index": 1,
        "max_index": 2020,
        "question_selected": 0,
    }
    rows = [{"arm": "cacheblend10", "session_id": "x", "version": 1}]
    (tmp_path / "results.json").write_text(json.dumps(rows))

    def write(entry):
        (tmp_path / "cacheblend10.log").write_text(
            "BOUNDARY_LOAD "
            + json.dumps(entry)
            + "\nBOUNDARY_SELECT "
            + json.dumps(entry)
            + "\n"
        )

    write(good)
    audit(tmp_path, "fresh", {"prompts": [prompt]})
    for field, value in (
        ("hit_tokens", 2048),
        ("actual_native_tokens", 1),
        ("question_selected", 1),
        ("selected", 204),
    ):
        write({**good, field: value})
        with pytest.raises(ValueError):
            audit(tmp_path, "fresh", {"prompts": [prompt]})
