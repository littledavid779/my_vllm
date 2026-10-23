"""Separate question and template repair with a fixed-budget factorial control."""

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
MODES = ("topk", "none", "question", "template", "both")
TEMPLATE = [151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271]

HELPER = '''def query_template_spans(tail):
    starts = [i for i in range(len(tail) - 1)
              if tail[i:i + 2] == [14582, 25]]
    if not starts:
        return 0, 0
    start = starts[-1]
    ends = [i for i in range(start + 2, len(tail)) if tail[i] == 151645]
    if len(ends) != 1:
        raise ValueError("question needs exactly one closing user marker")
    end = ends[0]
    expected = [151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271]
    if tail[end:] != expected:
        raise ValueError("unexpected Qwen3 no-thinking template")
    return end - start, len(tail) - end


def select_query_template(diff_k, budget, mode, question_length, template_length):
    """Top-k anchor or strict 2x2 inclusion/exclusion within a single budget."""
    total = diff_k.numel()
    if mode not in ("topk", "none", "question", "template", "both"):
        raise ValueError("unknown question/template mode")
    span = question_length + template_length
    if (not 1 <= budget <= total or min(question_length, template_length) < 0
            or span > min(64, total)):
        raise ValueError("invalid budget or spans")
    if total < 1024 or not span or mode == "topk":
        return torch.topk(diff_k, budget).indices, 0
    if min(question_length, template_length) == 0:
        raise ValueError("measured prompt needs both regions")
    q_start, t_start = total - span, total - template_length
    mandatory = []
    if mode in ("question", "both"):
        mandatory.extend(range(q_start, t_start))
    if mode in ("template", "both"):
        mandatory.extend(range(t_start, total))
    remaining = budget - len(mandatory)
    if remaining < 0 or remaining > q_start:
        raise ValueError("cannot satisfy exclusions within budget")
    reserved = torch.tensor(mandatory, device=diff_k.device, dtype=torch.long)
    if remaining == 0:
        return reserved, len(mandatory)
    # Mask the full pair of regions, including whichever factor is off.
    # Off-region positions cannot leak back in through the residual top-k.
    candidates = diff_k.clone()
    candidates[q_start:] = -torch.inf
    extra = torch.topk(candidates, remaining).indices
    return torch.cat((extra, reserved)), len(mandatory)


def query_template_evidence(positions, total, qlen, tlen):
    q_start, t_start = total - qlen - tlen, total - tlen
    qbits = sum(1 << (p - q_start) for p in positions if q_start <= p < t_start)
    tbits = sum(1 << (p - t_start) for p in positions if t_start <= p < total)
    return {"unique": len(set(positions)), "min_index": min(positions),
            "max_index": max(positions), "question_bits": qbits,
            "template_bits": tbits}


'''


def template_controls(source):
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
        """        # All modes pay the same prompt parsing and audit cost.
        token_list = (tokens.tolist() if isinstance(tokens, torch.Tensor)
                      else list(tokens))
        self.qt_lengths = (query_template_spans(token_list[-64:])
                           if len(token_list) >= 1024 else (0, 0))
        self.qt_prompt_sha = hashlib.sha256(
            json.dumps(token_list).encode()
        ).hexdigest()
"""
        + capture,
    )
    old = """            top_indices = torch.topk(diff_k, k=topk_num).indices
            top_indices, _ = torch.sort(top_indices)
"""
    new = """            mode = os.environ.get("LMCACHE_QUERY_TEMPLATE", "topk")
            qlen, tlen = self.qt_lengths
            top_indices, forced = select_query_template(
                diff_k, topk_num, mode, qlen, tlen
            )
            top_indices, _ = torch.sort(top_indices)
            selected = top_indices.tolist()
            evidence = query_template_evidence(selected, total_len, qlen, tlen)
            evidence.update(mode=mode, total=total_len, selected=len(selected),
                            forced=forced, question_tokens=qlen,
                            template_tokens=tlen, prompt_sha=self.qt_prompt_sha,
                            selected_sha=hashlib.sha256(
                                json.dumps(selected).encode()).hexdigest())
            logger.info("QUERY_TEMPLATE %s", json.dumps(evidence))
"""
    if source.count(old) != 1:
        raise ValueError("selector changed")
    return source.replace(old, new)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()
    base = commit_precheck((PROJECT / "upstream/lmcache" / RELATIVE).read_text())
    modified = template_controls(base)
    target = Path(sysconfig.get_paths()["purelib"]) / RELATIVE
    if not target.resolve().is_relative_to(PROJECT / ".venv"):
        raise ValueError("refusing to patch outside dedicated environment")
    if target.read_text() not in (base, modified):
        raise ValueError("apply pre-check correction first; unexpected blender")
    (PROJECT / "patches/lmcache-query-template.patch").write_text(
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
