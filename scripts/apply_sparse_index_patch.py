"""Optional diagnostic: activate the planned sparse mask and preserve token IDs."""

import argparse
import difflib
import hashlib
import json
import os
import sysconfig
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
SPARSE_RELATIVE = Path("lmcache/v1/compute/attention/metadata.py")
BACKEND_RELATIVE = Path("lmcache/v1/compute/attention/flash_infer_sparse.py")


def copied_indices(text):
    old = """        top_indices_slice = top_indices[
            self.sparse_blk_row_size - 1 :: self.sparse_blk_row_size
        ]
"""
    if text.count(old) != 1:
        raise ValueError("upstream sparse metadata source changed")
    activation = "        # self.is_causal = False\n"
    if text.count(activation) != 1:
        raise ValueError("upstream sparse activation changed")
    return text.replace(old, old[:-2] + "].clone()\n").replace(
        activation, "        self.is_causal = False\n"
    )


def workspace_argument(text):
    old_run = (
        "            output = attn_metadata.wrapper.run(query, key, value, output)\n"
    )
    old_wrapper = "        wrapper = HackBSAWrapper(self.workspace_buffer)\n"
    if text.count(old_run) != 1 or text.count(old_wrapper) != 1:
        raise ValueError("upstream sparse wrapper call changed")
    # FlashInfer 0.3.1's public wrapper handles grouped heads and workspace size.
    # The pinned custom wrapper bypasses both; use the installed public API.
    return text.replace(
        old_wrapper,
        "        wrapper = VariableBlockSparseAttentionWrapper"
        "(self.workspace_buffer)\n",
    ).replace(
        old_run,
        "            sparse_output = attn_metadata.wrapper.run(\n"
        "                query.transpose(0, 1),\n"
        "                key.transpose(0, 1),\n"
        "                value.transpose(0, 1),\n"
        "            )\n"
        "            output.copy_(sparse_output.transpose(0, 1))\n",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore", action="store_true")
    parser.add_argument("--record-only", action="store_true")
    args = parser.parse_args()
    pending = []
    for relative, transform, patch_name in (
        (SPARSE_RELATIVE, copied_indices, "lmcache-sparse-correctness.patch"),
        (BACKEND_RELATIVE, workspace_argument, "lmcache-sparse-public-api.patch"),
    ):
        original = (PROJECT / "upstream/lmcache" / relative).read_text()
        modified = transform(original)
        compile(modified, str(relative), "exec")
        target = Path(sysconfig.get_paths()["purelib"]) / relative
        if not target.resolve().is_relative_to(PROJECT / ".venv"):
            raise ValueError("refusing to modify outside the dedicated environment")
        if target.read_text() not in (original, modified):
            raise ValueError("installed sparse source is not a known state")
        patch = PROJECT / "patches" / patch_name
        patch.write_text(
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
    if args.record_only:
        print("Recorded optional sparse correctness patch; runtime unchanged")
        return
    for target, desired in pending:
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
            stream.write(desired.encode())
            temporary = Path(stream.name)
        temporary.chmod(target.stat().st_mode & 0o777)
        os.replace(temporary, target)
    result = {
        "restored": args.restore,
        "files_sha256": {
            str(t): hashlib.sha256(d.encode()).hexdigest() for t, d in pending
        },
    }
    (PROJECT / "state/sparse_index_patch.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
