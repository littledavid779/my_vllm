"""Split cached question keys without changing prompt tokens; allow native queries."""

import argparse
import difflib
import hashlib
import json
import os
import sysconfig
import tempfile
from pathlib import Path

from apply_precheck_kv_patch import commit_precheck
from apply_runtime_patch import patched_lmcache

PROJECT = Path(__file__).resolve().parents[1]
MODULE = "lmcache/v1/query_boundary.py"
BLENDER = "lmcache/v1/compute/blend/blender.py"
DATABASE = "lmcache/v1/token_database.py"
ADAPTER = "lmcache/integration/vllm/vllm_v1_adapter.py"
MODES = ("topk", "query", "fresh")

HELPER = '''"""Question boundary primitives that preserve token positions."""
import hashlib
import json
import os

TEMPLATE = [151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271]


def query_offset(tokens):
    values = tokens.tolist() if hasattr(tokens, "tolist") else list(tokens)
    offset = max(0, len(values) - 128)
    tail = values[offset:]
    starts = [i for i in range(len(tail) - 1) if tail[i:i + 2] == [14582, 25]]
    if not starts:
        return None
    start = starts[-1]
    ends = [i for i in range(start + 2, len(tail)) if tail[i] == 151645]
    if not ends or tail[ends[0]:ends[0] + len(TEMPLATE)] != TEMPLATE:
        return None
    return offset + start


def split_query_segment(tokens, start, mode):
    if mode not in ("topk", "query", "fresh"):
        raise ValueError("unknown boundary mode")
    q = query_offset(tokens)
    if q is None:
        return [(start, start + len(tokens), tokens)]
    if q <= 0:
        raise ValueError("context before the query is required")
    chunks = [(start, start + q, tokens[:q])]
    if mode != "fresh":
        # Salt the question key with the entire unchanged C segment. Otherwise
        # identical questions from different sessions alias one global cache key.
        chunks.append((start + q, start + len(tokens), tokens))
    return chunks


def boundary_info(tokens, hit):
    mode = os.environ.get("LMCACHE_QUERY_BOUNDARY", "topk")
    q = query_offset(tokens) if len(tokens) >= 1024 else None
    n = len(tokens)
    return {"mode": mode, "total": n, "query_start": q,
            "query_tokens": n - q if q is not None else 0,
            "hit_tokens": hit, "native_prompt_tokens": n - min(hit, n - 1),
            "prompt_sha": hashlib.sha256(json.dumps(list(tokens)).encode()).hexdigest()}


def select_boundary(diff, ratio, info):
    import torch
    n = diff.numel()
    mode, total, span = info["mode"], info["total"], info["query_tokens"]
    budget = max(1, int(total * ratio))
    forced = 0
    if mode == "fresh" and span:
        if n != info["query_start"]:
            raise ValueError("native question was included in blend computation")
        budget -= span
    if not 1 <= budget <= n:
        raise ValueError("invalid context repair budget")
    if mode == "query" and span:
        if n != total or span > budget:
            raise ValueError("invalid mandatory question region")
        forced = span
        extra = torch.topk(diff[:n - span], budget - span).indices
        selected = torch.cat((extra, torch.arange(n - span, n, device=diff.device)))
    else:
        selected = torch.topk(diff, budget).indices
    selected = torch.sort(selected).values
    return selected, forced
'''


def database_transform(source):
    old = """                if start_idx >= num_falses:
                    if make_key:
                        yield (
                            start_idx,
                            end_idx,
                            self._make_key_by_hash(
                                self._hash_tokens(token_chunk), request_configs
                            ),
                        )
                    else:
                        yield start_idx, end_idx, self._hash_tokens(token_chunk)
"""
    new = """                for begin, finish, key_tokens in split_query_segment(
                    token_chunk, start_idx,
                    os.environ.get("LMCACHE_QUERY_BOUNDARY", "topk"),
                ):
                    if begin >= num_falses:
                        h = self._hash_tokens(key_tokens)
                        yield (begin, finish,
                               self._make_key_by_hash(h, request_configs)
                               if make_key else h)
"""
    if source.count(old) != 1:
        raise ValueError("segment enumeration changed")
    source = source.replace(
        "from lmcache.logging import init_logger",
        "from lmcache.v1.query_boundary import split_query_segment\n"
        "from lmcache.logging import init_logger",
    )
    return source.replace(old, new)


def adapter_transform(source):
    source = source.replace(
        "from lmcache.logging import init_logger",
        "import json\nfrom lmcache.v1.query_boundary import boundary_info\n"
        "from lmcache.logging import init_logger",
    )
    anchor = """                    self.blender.blend(
                        tokens[:lmcache_cached_tokens],"""
    replacement = """                    info = boundary_info(
                        tokens, lmcache_cached_tokens
                    )
                    am = (next(iter(attn_metadata.values()))
                          if isinstance(attn_metadata, dict) else attn_metadata)
                    info["actual_native_tokens"] = int(am.num_actual_tokens)
                    if info["query_tokens"] and len(metadata.requests) == 1:
                        if info["actual_native_tokens"] != info["native_prompt_tokens"]:
                            raise ValueError("native question scheduling differs")
                    logger.info("BOUNDARY_LOAD %s", json.dumps(info))
                    self.blender.blend(
                        tokens[:lmcache_cached_tokens],"""
    if source.count(anchor) != 1:
        raise ValueError("blend entry changed")
    source = source.replace(anchor, replacement)
    anchor = (
        "                        slot_mapping="
        "slot_mapping[:lmcache_cached_tokens],\n                    )"
    )
    if source.count(anchor) != 1:
        raise ValueError("blend arguments changed")
    return source.replace(
        anchor,
        """                        slot_mapping=slot_mapping[:lmcache_cached_tokens],
                        boundary_info=info,
                    )""",
    )


def blender_transform(source):
    source = source.replace(
        "from typing import Optional, Union",
        "import json\n"
        "from lmcache.v1.query_boundary import boundary_info, select_boundary\n"
        "from typing import Optional, Union",
    )
    old = """            top_indices = torch.topk(diff_k, k=topk_num).indices
            top_indices, _ = torch.sort(top_indices)
"""
    new = """            top_indices, forced = select_boundary(
                diff_k, self.common_metadata.recomp_ratios[0], self.boundary_info
            )
            topk_num = len(top_indices)
            positions = top_indices.tolist()
            begin = self.boundary_info["query_start"]
            evidence = dict(self.boundary_info)
            evidence.update(selected=topk_num, forced=forced,
                            blend_tokens=total_len, unique=len(set(positions)),
                            min_index=min(positions), max_index=max(positions),
                            question_selected=sum(p >= begin for p in positions)
                            if begin is not None else 0)
            logger.info("BOUNDARY_SELECT %s", json.dumps(evidence))
"""
    if source.count(old) != 1:
        raise ValueError("selector changed")
    source = source.replace(old, new)
    anchor = (
        "        layerwise_model_executor = self.layerwise_model.compute_layer(tokens)"
    )
    if source.count(anchor) != 1:
        raise ValueError("blend capture changed")
    return source.replace(
        anchor,
        """        self.boundary_info = kwargs.pop("boundary_info", None)
        if self.boundary_info is None:
            self.boundary_info = boundary_info(tokens.tolist(), len(tokens))
"""
        + anchor,
    )


def plans():
    transforms = {
        BLENDER: (commit_precheck, blender_transform),
        ADAPTER: (patched_lmcache, adapter_transform),
        DATABASE: (lambda s: s, database_transform),
    }
    result = {}
    for path, (base, change) in transforms.items():
        text = base((PROJECT / "upstream/lmcache" / path).read_text())
        result[path] = (text, change(text))
    result[MODULE] = (None, HELPER)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore", action="store_true")
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    site = Path(sysconfig.get_paths()["purelib"])
    pending, differences = [], []
    for relative, (base, changed) in plans().items():
        compile(changed, relative, "exec")
        differences.extend(
            difflib.unified_diff(
                (base or "").splitlines(True),
                changed.splitlines(True),
                fromfile="a/" + relative,
                tofile="b/" + relative,
            )
        )
        target = site / relative
        if not target.resolve().is_relative_to(PROJECT / ".venv"):
            raise ValueError("outside dedicated environment")
        current = target.read_text() if target.exists() else None
        if not args.record and current not in (base, changed):
            raise ValueError(f"unexpected runtime state: {relative}")
        pending.append((target, base if args.restore else changed))
    patch = PROJECT / "patches/lmcache-query-boundary.patch"
    if args.record:
        patch.write_text("".join(differences))
        return
    for target, desired in pending:
        if desired is None:
            if target.exists():
                target.unlink()
            continue
        mode = target.stat().st_mode & 0o777 if target.exists() else 0o644
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
            stream.write(desired.encode())
            temporary = Path(stream.name)
        temporary.chmod(mode)
        os.replace(temporary, target)
    print(
        json.dumps(
            {
                "restored": args.restore,
                "files": {
                    str(p): hashlib.sha256(t.encode()).hexdigest() if t else None
                    for p, t in pending
                },
            }
        )
    )


if __name__ == "__main__":
    main()
