"""CPU-only audit of selected-query mask/index semantics in pinned upstream."""

import argparse
import hashlib
import json
from pathlib import Path


class PlanRecorder:
    def plan(self, *args, **kwargs):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a new output")
    import torch
    from lmcache.v1.compute.attention import metadata

    indices = torch.arange(64, dtype=torch.int64)
    before = indices.clone()
    value = metadata.LMCFlashInferSparseMetadata(
        wrapper=PlanRecorder(),
        seq_len=128,
        num_qo_heads=32,
        num_kv_heads=8,
        head_dim=128,
        block_col_sizes=torch.tensor([32, 32, 32, 32]),
    )
    value.update_from_top_indices(indices)
    changed = torch.where(before != indices)[0].tolist()
    selected_positions = [0, 3]
    packed_positions = [3, 4]  # FlashAttention bottom-right alignment: Lk - Lq + row.
    result = {
        "metadata_sha256": hashlib.sha256(
            Path(metadata.__file__).read_bytes()
        ).hexdigest(),
        "flashattn_example": {
            "key_length": 5,
            "selected_query_positions": selected_positions,
            "packed_causal_positions": packed_positions,
            "extra_future_edges": sum(
                p - q for p, q in zip(packed_positions, selected_positions, strict=True)
            ),
            "scope": "Interface/source audit; not a GPU accuracy attribution.",
        },
        "sparse_metadata_argument_mutation": {
            "is_causal_after_selection": value.is_causal,
            "changed_offsets": changed,
            "before": [before[i].item() for i in changed],
            "after": [indices[i].item() for i in changed],
            "scope": "Actual CPU call into the installed upstream metadata method.",
        },
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
