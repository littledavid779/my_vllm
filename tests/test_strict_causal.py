"""Validate position preservation and exact allowed-key sets before model runs."""

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
from apply_strict_causal_patch import strict_metadata  # noqa: E402


def test_strict_mask_positions_and_composition(monkeypatch):
    def pack_cpu(bits, bitorder):
        return torch.from_numpy(np.packbits(bits.numpy(), bitorder=bitorder))

    # This is the CPU logic test; the separate GPU validator exercises real packbits.
    parent, quant = ModuleType("flashinfer"), ModuleType("flashinfer.quantization")
    quant.packbits = pack_cpu
    parent.quantization = quant
    monkeypatch.setitem(sys.modules, "flashinfer", parent)
    monkeypatch.setitem(sys.modules, "flashinfer.quantization", quant)
    source = (
        PROJECT / "upstream/lmcache/lmcache/v1/compute/attention/metadata.py"
    ).read_text()
    cls = next(
        n
        for n in ast.parse(strict_metadata(source)).body
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
            ast.Module(body=[function], type_ignores=[]), "strict_metadata", "exec"
        ),
        env,
    )
    update = env["update_from_top_indices"]
    meta = SimpleNamespace(strict_query_positions=None, seq_len=67, is_causal=True)
    original = torch.tensor([0, 1, 31, 32, 66])
    before = original.clone()
    update(meta, original)
    assert torch.equal(original, before)
    mask = np.unpackbits(meta.strict_packed_mask.numpy(), bitorder="little")[: 5 * 67]
    mask = mask.reshape(5, 67)
    for row, position in enumerate(original):
        assert np.array_equal(np.flatnonzero(mask[row]), np.arange(int(position) + 1))
    update(meta, torch.tensor([1, 3]))
    assert meta.strict_query_positions.tolist() == [1, 32]
    mask = np.unpackbits(meta.strict_packed_mask.numpy(), bitorder="little")[: 2 * 67]
    assert [int(row.sum()) for row in mask.reshape(2, 67)] == [2, 33]
