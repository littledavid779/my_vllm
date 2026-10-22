"""Opt-in exact selected-query causal mask; keep upstream selector unchanged."""

import argparse
import ast
import difflib
import hashlib
import json
import os
import sysconfig
import tempfile
import textwrap
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
META = Path("lmcache/v1/compute/attention/metadata.py")
BACKEND = Path("lmcache/v1/compute/attention/flash_infer_sparse.py")


def replace_method(source, cls_name, method_name, replacement):
    cls = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.ClassDef) and n.name == cls_name
    )
    method = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name
    )
    lines = source.splitlines(keepends=True)
    lines[method.lineno - 1 : method.end_lineno] = [
        textwrap.indent(textwrap.dedent(replacement).strip() + "\n", "    ")
    ]
    return "".join(lines)


def strict_metadata(source):
    anchor = (
        "    q_data_dtype: torch.dtype = torch.bfloat16  "
        "# TODO(Jiayi): remove hardcode\n"
    )
    if source.count(anchor) != 1:
        raise ValueError("unexpected pinned metadata fields")
    source = source.replace(
        anchor,
        anchor + "    strict_packed_mask: torch.Tensor | None = None\n"
        "    strict_query_positions: torch.Tensor | None = None\n",
    )
    return replace_method(
        source,
        "LMCFlashInferSparseMetadata",
        "update_from_top_indices",
        """
def update_from_top_indices(self, top_indices: torch.Tensor):
    from flashinfer.quantization import packbits

    # Compose positions if there is more than one check layer; never mutate input.
    if self.strict_query_positions is None:
        positions = top_indices.clone()
    else:
        positions = self.strict_query_positions[top_indices].clone()
    self.strict_query_positions = positions
    key_positions = torch.arange(self.seq_len, device=positions.device)
    mask = key_positions.unsqueeze(0) <= positions.unsqueeze(1)
    self.strict_packed_mask = packbits(mask.flatten(), bitorder="little")
    self.is_causal = False
""",
    )


def strict_backend(source):
    return replace_method(
        source,
        "LMCFlashInferSparseBackend",
        "forward_contiguous",
        """
def forward_contiguous(self, query, key, value, output, attn_metadata, **kwargs):
    assert isinstance(attn_metadata, LMCFlashInferSparseMetadata)
    packed = attn_metadata.strict_packed_mask
    result = flashinfer.prefill.single_prefill_with_kv_cache(
        q=query, k=key, v=value,
        scale_q=None, scale_k=self.k_scale, scale_v=self.v_scale,
        o_dtype=query.dtype, custom_mask=None, packed_custom_mask=packed,
        causal=packed is None, kv_layout="NHD", pos_encoding_mode="NONE",
        use_fp16_qk_reduction=False, sm_scale=self.sm_scale,
        window_left=self.window_left, logits_soft_cap=self.logits_soft_cap,
        rope_scale=None, rope_theta=None, backend="auto", return_lse=False,
    )
    output.copy_(result)
    return output
""",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore", action="store_true")
    parser.add_argument("--record-only", action="store_true")
    args = parser.parse_args()
    pending = []
    for relative, transform, name in (
        (META, strict_metadata, "lmcache-strict-metadata.patch"),
        (BACKEND, strict_backend, "lmcache-strict-attention.patch"),
    ):
        original = (PROJECT / "upstream/lmcache" / relative).read_text()
        modified = transform(original)
        compile(modified, str(relative), "exec")
        target = Path(sysconfig.get_paths()["purelib"]) / relative
        if not target.resolve().is_relative_to(PROJECT / ".venv"):
            raise ValueError("refusing to patch outside the dedicated environment")
        if target.read_text() not in (original, modified):
            raise ValueError("restore other diagnostics before enabling strict mask")
        (PROJECT / "patches" / name).write_text(
            "".join(
                difflib.unified_diff(
                    original.splitlines(keepends=True),
                    modified.splitlines(keepends=True),
                    fromfile="a/" + str(relative),
                    tofile="b/" + str(relative),
                )
            )
        )
        pending.append((target, original if args.restore else modified))
    if not args.record_only:
        for target, desired in pending:
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                stream.write(desired.encode())
                temporary = Path(stream.name)
            temporary.chmod(target.stat().st_mode & 0o777)
            os.replace(temporary, target)
    print(
        json.dumps(
            {
                "restored": args.restore,
                "record_only": args.record_only,
                "sha256": {
                    str(p): hashlib.sha256(s.encode()).hexdigest() for p, s in pending
                },
            }
        )
    )


if __name__ == "__main__":
    main()
