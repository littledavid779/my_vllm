"""GPU checks against independent float32 dense attention and future perturbation."""

import argparse
import json
import math
import os
from pathlib import Path

from run_smoke import gpu_state


def reference(q, k, v, positions):
    import torch

    grouped_k = k.float().repeat_interleave(q.shape[1] // k.shape[1], dim=1)
    grouped_v = v.float().repeat_interleave(q.shape[1] // v.shape[1], dim=1)
    scores = torch.einsum("qhd,khd->hqk", q.float(), grouped_k) / math.sqrt(q.shape[-1])
    allowed = torch.arange(k.shape[0], device=q.device)[None, :] <= positions[:, None]
    scores.masked_fill_(~allowed[None, :, :], -torch.inf)
    return torch.einsum("hqk,khd->qhd", torch.softmax(scores, -1), grouped_v)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("need a new output")
    state = gpu_state(args.gpu)
    if state["pids"] or state["memory_mib"] > 100:
        raise RuntimeError("GPU is busy")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["CUDA_HOME"] = "/usr/local/cuda"
    os.environ["MAX_JOBS"] = "4"
    os.environ["FLASHINFER_WORKSPACE_DIR"] = str(
        Path(__file__).resolve().parents[1] / ".cache/flashinfer"
    )
    import numpy as np
    import torch
    from lmcache.v1.compute.attention.flash_infer_sparse import (
        LMCFlashInferSparseBackend,
    )
    from lmcache.v1.compute.attention.metadata import LMCFlashInferSparseMetadata

    torch.manual_seed(22026)
    backend = object.__new__(LMCFlashInferSparseBackend)
    backend.k_scale = backend.v_scale = 1.0
    backend.sm_scale = 1 / math.sqrt(128)
    backend.window_left, backend.logits_soft_cap = -1, 0.0
    rows = []
    for length, selected in (
        (7, [0, 3, 6]),
        (67, [0, 1, 31, 32, 66]),
        (129, [0, 17, 64, 128]),
    ):
        q = torch.randn(length, 32, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(length, 8, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        positions = torch.tensor(selected, device="cuda")
        meta = LMCFlashInferSparseMetadata(
            wrapper=None,
            seq_len=length,
            num_qo_heads=32,
            num_kv_heads=8,
            head_dim=128,
            block_col_sizes=torch.empty(0, device="cuda"),
        )
        full = backend.forward_contiguous(q, k, v, torch.empty_like(q), meta)
        before = positions.clone()
        meta.update_from_top_indices(positions)
        assert torch.equal(positions, before)
        decoded = np.unpackbits(
            meta.strict_packed_mask.cpu().numpy(), bitorder="little"
        )
        expected = np.arange(length)[None, :] <= np.array(selected)[:, None]
        assert np.array_equal(
            decoded[: expected.size].reshape(expected.shape), expected
        )
        partial = backend.forward_contiguous(
            q[positions], k, v, torch.empty_like(q[positions]), meta
        )
        ref = reference(q[positions], k, v, positions)
        torch.testing.assert_close(partial.float(), ref, atol=0.02, rtol=0.02)
        torch.testing.assert_close(
            partial.float(), full[positions].float(), atol=0.02, rtol=0.02
        )
        # None of the chosen past queries may read any of the future keys/values.
        past = torch.tensor([0, 2], device="cuda")
        meta.strict_query_positions = None
        meta.update_from_top_indices(past)
        a = backend.forward_contiguous(q[past], k, v, torch.empty_like(q[past]), meta)
        altered_k, altered_v = k.clone(), v.clone()
        altered_k[3:] += 100
        altered_v[3:] -= 100
        b = backend.forward_contiguous(
            q[past], altered_k, altered_v, torch.empty_like(q[past]), meta
        )
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        rows.append(
            {
                "length": length,
                "selected": selected,
                "max_abs_vs_float32": (partial.float() - ref).abs().max().item(),
                "max_abs_vs_full_rows": (partial - full[positions]).abs().max().item(),
                "future_perturbation_max_abs": (a - b).abs().max().item(),
            }
        )
    torch.cuda.synchronize()
    args.output.write_text(
        json.dumps({"status": "passed", "gpu": args.gpu, "cases": rows}, indent=2)
    )
    print(args.output.read_text())


if __name__ == "__main__":
    main()
