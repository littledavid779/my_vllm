"""Mask planning must not change the token indices later used for KV writes."""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
from apply_sparse_index_patch import copied_indices  # noqa: E402


def method(source):
    cls = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.ClassDef) and n.name == "LMCFlashInferSparseMetadata"
    )
    function = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "update_from_top_indices"
    )
    env = {"torch": torch}
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]), "upstream_method", "exec"
        ),
        env,
    )
    return env["update_from_top_indices"]


def test_copy_preserves_mask_but_stops_index_corruption():
    original = (
        PROJECT / "upstream/lmcache/lmcache/v1/compute/attention/metadata.py"
    ).read_text()
    results = []
    for source in (original, copied_indices(original)):
        plans = []
        metadata = SimpleNamespace(
            sparse_blk_row_size=32,
            sparse_blk_col_size=32,
            block_col_sizes=torch.tensor([32, 32, 32, 32]),
            num_kv_heads=8,
            num_qo_heads=32,
            head_dim=128,
            q_data_dtype=torch.bfloat16,
            is_causal=True,
            wrapper=SimpleNamespace(
                plan=lambda *args, captured=plans, **kwargs: captured.append(args)
            ),
        )
        indices = torch.arange(64, dtype=torch.int64)
        method(source)(metadata, indices)
        results.append((indices, plans[0][0], metadata.is_causal))
    assert not torch.equal(results[0][0], torch.arange(64))
    assert torch.equal(results[1][0], torch.arange(64))
    assert torch.equal(results[0][1], results[1][1])
    assert results[0][2] is True  # Raw backend still takes the ordinary causal path.
    assert results[1][2] is False  # The corrected run uses the planned sparse mask.
