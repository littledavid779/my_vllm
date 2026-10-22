"""Record/check the original vLLM checkout without touching its index or branch."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def snapshot(parent):
    def git(*args):
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={parent}", "-C", str(parent), *args]
        )

    return {
        "path": str(parent),
        "head": git("rev-parse", "HEAD").decode().strip(),
        "branch": git("symbolic-ref", "--short", "HEAD").decode().strip(),
        "index_sha256": hashlib.sha256(
            (parent / ".git/index").read_bytes()
        ).hexdigest(),
        "staged_diff_sha256": hashlib.sha256(
            git("diff", "--cached", "--binary")
        ).hexdigest(),
        "working_diff_sha256": hashlib.sha256(git("diff", "--binary")).hexdigest(),
        "tracked_status": git("status", "--porcelain", "--untracked-files=no").decode(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("record", "check"))
    args = parser.parse_args()
    target = PROJECT / "state/parent_before.json"
    if args.action == "record":
        if target.exists():
            raise RuntimeError("refusing to overwrite original checkout snapshot")
        parent = PROJECT.parents[1]
        current = snapshot(parent)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(current, indent=2))
    else:
        before = json.loads(target.read_text())
        # The independent experiment may move to another disk. Keep guarding
        # the original checkout recorded in the snapshot, not a new ancestor.
        parent = Path(before["path"])
        if not parent.is_absolute():
            raise RuntimeError("recorded parent path must be absolute")
        current = snapshot(parent)
        if current != before:
            changed = [key for key in current if current[key] != before[key]]
            raise RuntimeError(f"parent checkout changed: {changed}")
    print(json.dumps({"action": args.action, "parent": str(parent), "ok": True}))


if __name__ == "__main__":
    main()
