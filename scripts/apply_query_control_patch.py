"""Opt-in fixed-budget controls for the query-refresh mechanism experiment."""

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
MODES = ("topk", "query", "tail64", "pre_tail64", "random64")

# These token markers are verified against the locked Qwen3 tokenizer and every
# measured prompt before launch. They are prompt syntax, not answer annotations.
HELPER = '''def query_control_span(tail):
    starts = [i for i in range(len(tail) - 1)
              if tail[i:i + 2] == [14582, 25]]
    if not starts:
        return 0
    start = starts[-1]
    if not any(tail[i:i + 2] == [151644, 77091]
               for i in range(start + 2, len(tail) - 1)):
        raise ValueError("query marker without assistant template")
    return len(tail) - start


def select_query_control(diff_k, topk_num, mode, query_length, seed=20261023):
    """Reserve positions within one budget, then take the best remaining K diffs."""
    total_len = diff_k.numel()
    if mode not in ("topk", "query", "tail64", "pre_tail64", "random64"):
        raise ValueError("unknown query control")
    if not 1 <= topk_num <= total_len or not 0 <= query_length <= 64:
        raise ValueError("invalid budget or query span")
    if total_len < 1024 or mode == "topk":
        return torch.topk(diff_k, topk_num).indices, 0
    if mode == "query":
        if query_length > topk_num:
            raise ValueError("cannot fit entire query in the fixed budget")
        mandatory = torch.arange(total_len - query_length, total_len,
                                 device=diff_k.device)
    elif mode == "tail64":
        count = min(64, topk_num)
        mandatory = torch.arange(total_len - count, total_len,
                                 device=diff_k.device)
    elif mode == "pre_tail64":
        count = min(64, topk_num)
        mandatory = torch.arange(total_len - 64 - count, total_len - 64,
                                 device=diff_k.device)
    else:
        # CPU generator gives the same reserved set across devices and repeats.
        generator = torch.Generator(device="cpu").manual_seed(seed + total_len)
        mandatory = torch.randperm(total_len - 64, generator=generator)[
            :min(64, topk_num)].to(diff_k.device)
    count = mandatory.numel()
    if count == 0:
        return torch.topk(diff_k, topk_num).indices, 0
    remaining = topk_num - count
    if remaining == 0:
        return mandatory, count
    # Mask a clone, leaving scores and mandatory indices untouched.
    candidates = diff_k.clone()
    candidates[mandatory] = -torch.inf
    extra = torch.topk(candidates, remaining).indices
    return torch.cat((extra, mandatory)), count


'''


def query_controls(source):
    if source.count("class LMCBlender:") != 1:
        raise ValueError("blender class changed")
    source = source.replace(
        "from typing import Optional, Union",
        "import hashlib\nimport json\nimport os\nfrom typing import Optional, Union",
    )
    source = source.replace("class LMCBlender:", HELPER + "class LMCBlender:")
    capture = (
        "        layerwise_model_executor = self.layerwise_model.compute_layer(tokens)"
    )
    if source.count(capture) != 1:
        raise ValueError("blending entry changed")
    source = source.replace(
        capture,
        """        # All arms pay identical prompt-boundary parsing and audit costs.
        tail = (tokens[-64:].tolist() if isinstance(tokens, torch.Tensor)
                else tokens[-64:])
        self.query_control_length = (query_control_span(tail)
                                     if len(tokens) >= 1024 else 0)
        self.query_control_signature = hashlib.sha256(
            json.dumps(tail).encode()
        ).hexdigest()[:16]
"""
        + capture,
    )
    old = """            top_indices = torch.topk(diff_k, k=topk_num).indices
            top_indices, _ = torch.sort(top_indices)
"""
    new = """            mode = os.environ.get("LMCACHE_QUERY_CONTROL", "topk")
            query_length = self.query_control_length
            top_indices, forced = select_query_control(
                diff_k, topk_num, mode, query_length
            )
            top_indices, _ = torch.sort(top_indices)
            # Identical synchronization in all arms; included in HTTP TTFT.
            coverage = torch.stack((
                (top_indices >= total_len - query_length).sum(),
                (top_indices >= total_len - min(64, total_len)).sum(),
            )).tolist()
            logger.info(
                "QUERY_CONTROL mode=%s total=%d selected=%d forced=%d "
                "query=%d qcovered=%d tailcovered=%d signature=%s",
                mode, total_len, top_indices.numel(), forced, query_length,
                coverage[0], coverage[1], self.query_control_signature,
            )
"""
    if source.count(old) != 1:
        raise ValueError("selector changed")
    return source.replace(old, new)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()
    base = commit_precheck((PROJECT / "upstream/lmcache" / RELATIVE).read_text())
    modified = query_controls(base)
    target = Path(sysconfig.get_paths()["purelib"]) / RELATIVE
    if not target.resolve().is_relative_to(PROJECT / ".venv"):
        raise ValueError("refusing to patch outside the dedicated environment")
    if target.read_text() not in (base, modified):
        raise ValueError("apply pre-check correction first; unexpected blender")
    (PROJECT / "patches/lmcache-query-controls.patch").write_text(
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
