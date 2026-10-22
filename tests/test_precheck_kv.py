"""The paged-cache exporter must see KV actually computed before token selection."""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
from apply_precheck_kv_patch import commit_precheck  # noqa: E402


def method(source):
    cls = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.ClassDef) and n.name == "LMCBlender"
    )
    func = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "process_qkv"
    )
    env = {
        "torch": torch,
        "Optional": Optional,
        "LMCAttnMetadata": object,
        "logger": SimpleNamespace(debug=lambda _: None),
    }
    exec(
        compile(ast.Module(body=[func], type_ignores=[]), "blender_method", "exec"), env
    )
    return env["process_qkv"]


def test_precheck_export_matches_computed_kv_including_gaps():
    source = (
        PROJECT / "upstream/lmcache/lmcache/v1/compute/blend/blender.py"
    ).read_text()
    fresh_k, fresh_v = torch.full((4, 2), 3.0), torch.full((4, 2), 5.0)
    for code, writes_current in ((source, False), (commit_precheck(source), True)):
        old_k, old_v = torch.zeros_like(fresh_k), torch.zeros_like(fresh_v)
        layer = SimpleNamespace(
            self_attn=SimpleNamespace(rotary_emb=lambda positions, q, k: (q, k))
        )
        blender = SimpleNamespace(
            gpu_connector=SimpleNamespace(get_kv=lambda _, k=old_k, v=old_v: (k, v)),
            metadata=SimpleNamespace(positions=None, imp_indices=None),
            layerwise_model=SimpleNamespace(
                vllm_model=SimpleNamespace(model=SimpleNamespace(layers=[layer]))
            ),
            common_metadata=SimpleNamespace(check_layers=[1]),
        )
        result = method(code)(
            blender,
            torch.ones_like(fresh_k),
            fresh_k,
            fresh_v,
            torch.ones_like(fresh_k),
            0,
            None,
            object(),
        )
        assert torch.equal(result[1], fresh_k)
        assert torch.equal(result[2], fresh_v)
        assert torch.equal(old_k, fresh_k) is writes_current
        assert torch.equal(old_v, fresh_v) is writes_current
        if writes_current:
            assert result[1].data_ptr() == old_k.data_ptr()
            assert result[2].data_ptr() == old_v.data_ptr()
