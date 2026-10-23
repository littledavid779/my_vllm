"""Shared query refresh, with a fixed-budget EPIC-inspired boundary control.

This is a selection-rule adaptation, not an EPIC implementation: all arms still
pay the LMCache layer-1 K-difference probe and use its cache construction path.
"""

import argparse
import difflib
import hashlib
import json
import os
import sysconfig
import tempfile
from pathlib import Path

from apply_precheck_kv_patch import RELATIVE, commit_precheck
from apply_query_control_patch import HELPER as QUERY_HELPER
from apply_query_control_patch import query_controls

PROJECT = Path(__file__).resolve().parents[1]
MODES = ("query", "tail64", "boundary")
SEPARATOR = [671, 84648, 87354, 62, 22, 64, 24, 66, 671, 671]

HELPER = '''def fair_boundaries(tokens):
    separator = [671, 84648, 87354, 62, 22, 64, 24, 66, 671, 671]
    return [i for i in range(len(tokens) - len(separator) + 1)
            if tokens[i:i + len(separator)] == separator]


def boundary_positions(total, budget, query_length, starts):
    """Equal-depth prefixes of nonfirst chunks, plus the entire query.

    Include the delimiter at each boundary. Water filling transfers unused
    allocation from the short edited chunk to the longer suffix. No gold labels
    or supporting-fact locations are consulted.
    """
    if not 0 <= query_length <= budget <= total or budget < 1:
        raise ValueError("invalid boundary budget")
    query_start = total - query_length
    if starts != sorted(set(starts)) or any(s < 0 or s >= query_start for s in starts):
        raise ValueError("invalid chunk boundaries")
    selected = list(range(query_start, total))
    if not starts:
        raise ValueError("boundary selection needs nonfirst chunks")
    ends = starts[1:] + [query_start]
    offset = 0
    while len(selected) < budget:
        added = 0
        for start, end in zip(starts, ends):
            if start + offset < end and len(selected) < budget:
                selected.append(start + offset)
                added += 1
        if not added:
            # Late edits may leave fewer than 10% tokens after the first chunk.
            # Fill from the exact prefix to keep the comparison's total budget.
            # Since budget <= total, these positions fit before starts[0].
            selected.extend(range(budget - len(selected)))
            break
        offset += 1
    return sorted(selected)


def select_query_fair(diff_k, topk_num, mode, query_length, starts):
    if mode not in ("query", "tail64", "boundary"):
        raise ValueError("unknown fair control")
    if diff_k.numel() < 1024:
        return torch.topk(diff_k, topk_num).indices, 0
    if mode != "boundary":
        return select_query_control(diff_k, topk_num, mode, query_length)
    positions = boundary_positions(diff_k.numel(), topk_num, query_length, starts)
    return torch.tensor(positions, device=diff_k.device, dtype=torch.long), topk_num


'''

ALL_HELPERS = QUERY_HELPER + HELPER


def fair_controls(source):
    source = query_controls(source)
    source = source.replace("class LMCBlender:", HELPER + "class LMCBlender:")
    old = """        tail = (tokens[-64:].tolist() if isinstance(tokens, torch.Tensor)
                else tokens[-64:])"""
    new = """        # Identical parsing cost in every fair-comparison arm.
        token_list = (tokens.tolist() if isinstance(tokens, torch.Tensor)
                      else list(tokens))
        tail = token_list[-64:]
        self.query_fair_starts = fair_boundaries(token_list)
        self.query_fair_prompt_sha = hashlib.sha256(
            json.dumps(token_list).encode()
        ).hexdigest()"""
    if source.count(old) != 1:
        raise ValueError("query-control capture changed")
    source = source.replace(old, new)
    old = """            top_indices, forced = select_query_control(
                diff_k, topk_num, mode, query_length
            )"""
    new = """            top_indices, forced = select_query_fair(
                diff_k, topk_num, mode, query_length, self.query_fair_starts
            )"""
    if source.count(old) != 1:
        raise ValueError("query-control selector changed")
    source = source.replace(old, new)
    needle = "            k, v = k[top_indices], v[top_indices]"
    if source.count(needle) != 1:
        raise ValueError("selected KV write changed")
    return source.replace(
        needle,
        """            selected_sha = hashlib.sha256(
                json.dumps(top_indices.tolist()).encode()
            ).hexdigest()
            logger.info("QUERY_FAIR mode=%s promptsha=%s selectedsha=%s",
                        mode, self.query_fair_prompt_sha, selected_sha)
"""
        + needle,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()
    base = commit_precheck((PROJECT / "upstream/lmcache" / RELATIVE).read_text())
    modified = fair_controls(base)
    target = Path(sysconfig.get_paths()["purelib"]) / RELATIVE
    if not target.resolve().is_relative_to(PROJECT / ".venv"):
        raise ValueError("refusing to patch outside dedicated environment")
    if target.read_text() not in (base, modified):
        raise ValueError("apply pre-check correction first; unexpected blender")
    (PROJECT / "patches/lmcache-query-fair.patch").write_text(
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
