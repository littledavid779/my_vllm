"""Fail closed if either upstream checkout or baseline algorithm differs."""

import hashlib
import json
import subprocess
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def main():
    locked = json.loads((PROJECT / "sources.lock.json").read_text())
    output = {}
    for name, item in locked["upstream"].items():
        path = PROJECT / item["path"]
        head = (
            subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"])
            .decode()
            .strip()
        )
        status = subprocess.check_output(
            ["git", "-C", str(path), "status", "--porcelain"]
        )
        if head != item["commit"] or status:
            raise ValueError(f"upstream {name} is changed or not pinned")
        for file, expected in item["files_sha256"].items():
            actual = hashlib.sha256((path / file).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"upstream source changed: {name}/{file}")
        output[name] = head
    print(json.dumps({"verified": output}))


if __name__ == "__main__":
    main()
