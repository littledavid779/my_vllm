"""Commit already-computed pre-check KV into the buffer exported to vLLM."""

import argparse
import difflib
import hashlib
import json
import os
import sysconfig
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
RELATIVE = Path("lmcache/v1/compute/blend/blender.py")


def commit_precheck(source):
    old = """        else:
            return q, k, v, residual, attn_output, attn_metadata
"""
    if source.count(old) != 1:
        raise ValueError("upstream no-selection KV path changed")
    return source.replace(
        old,
        """        else:
            # These layers are already fully computed. Export their current KV,
            # including separator gaps, rather than the retrieved stale buffer.
            old_k.copy_(k)
            old_v.copy_(v)
            return q, old_k, old_v, residual, attn_output, attn_metadata
""",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()
    original = (PROJECT / "upstream/lmcache" / RELATIVE).read_text()
    modified = commit_precheck(original)
    target = Path(sysconfig.get_paths()["purelib"]) / RELATIVE
    if not target.resolve().is_relative_to(PROJECT / ".venv"):
        raise ValueError("refusing to patch outside the dedicated environment")
    if target.read_text() not in (original, modified):
        raise ValueError("installed blender is not a known state")
    (PROJECT / "patches/lmcache-precheck-kv-write.patch").write_text(
        "".join(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                modified.splitlines(keepends=True),
                fromfile="a/" + str(RELATIVE),
                tofile="b/" + str(RELATIVE),
            )
        )
    )
    desired = original if args.restore else modified
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
