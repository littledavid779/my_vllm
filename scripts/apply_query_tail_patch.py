"""Reserve a fixed tail inside the original top-k budget; opt-in integration."""

import argparse
import difflib
import hashlib
import json
import os
import sysconfig
import tempfile
from pathlib import Path

from apply_precheck_kv_patch import RELATIVE, commit_precheck

PROJECT = Path(__file__).resolve().parents[1]

HELPER = '''def select_with_query_tail(diff_k, topk_num, tail_tokens):
    """Return exactly topk_num unique positions, without changing scores."""
    if tail_tokens not in (0, 64):
        raise ValueError("only frozen 0/64-token query-tail policies are supported")
    if not 1 <= topk_num <= diff_k.numel():
        raise ValueError("invalid selection budget")
    total_len = diff_k.numel()
    forced = min(tail_tokens, topk_num) if total_len >= 1024 else 0
    if not forced:
        return torch.topk(diff_k, k=topk_num).indices, 0
    tail = torch.arange(total_len - forced, total_len, device=diff_k.device)
    other_count = topk_num - forced
    if not other_count:
        return tail, forced
    outside = torch.topk(diff_k[:total_len - forced], k=other_count).indices
    return torch.cat((outside, tail)), forced


'''


def reserve_query_tail(source):
    if source.count("class LMCBlender:") != 1:
        raise ValueError("blender class changed")
    source = source.replace(
        "from typing import Optional, Union",
        "import os\nfrom typing import Optional, Union",
    )
    source = source.replace("class LMCBlender:", HELPER + "class LMCBlender:")
    old = """            top_indices = torch.topk(diff_k, k=topk_num).indices
            top_indices, _ = torch.sort(top_indices)
"""
    new = """            tail_tokens = int(
                os.environ.get("LMCACHE_QUERY_TAIL_TOKENS", "0")
            )
            top_indices, forced = select_with_query_tail(
                diff_k, topk_num, tail_tokens
            )
            top_indices, _ = torch.sort(top_indices)
            # Identical instrumentation in both arms, included in client TTFT.
            tail_scope = min(64, total_len)
            covered = int((top_indices >= total_len - tail_scope).sum().item())
            logger.info(
                "QUERY_TAIL total=%d selected=%d forced=%d covered=%d",
                total_len, top_indices.numel(), forced, covered,
            )
"""
    if source.count(old) != 1:
        raise ValueError("upstream selector changed")
    return source.replace(old, new)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()
    original = (PROJECT / "upstream/lmcache" / RELATIVE).read_text()
    base = commit_precheck(original)
    modified = reserve_query_tail(base)
    target = Path(sysconfig.get_paths()["purelib"]) / RELATIVE
    if not target.resolve().is_relative_to(PROJECT / ".venv"):
        raise ValueError("refusing to patch outside dedicated environment")
    if target.read_text() not in (base, modified):
        raise ValueError("apply common pre-check KV patch first; unknown blender")
    (PROJECT / "patches/lmcache-query-tail.patch").write_text(
        "".join(
            difflib.unified_diff(
                base.splitlines(keepends=True),
                modified.splitlines(keepends=True),
                fromfile="a/" + str(RELATIVE),
                tofile="b/" + str(RELATIVE),
            )
        )
    )
    desired = base if args.restore else modified
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
        stream.write(desired.encode())
        temporary = Path(stream.name)
    temporary.chmod(target.stat().st_mode & 0o777)
    os.replace(temporary, target)
    print(
        json.dumps(
            {
                "restored": args.restore,
                "sha256": hashlib.sha256(desired.encode()).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()
