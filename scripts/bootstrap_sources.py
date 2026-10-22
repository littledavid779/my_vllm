"""Populate ignored upstream checkouts from the committed source lock."""

import json
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def main():
    lock = json.loads((PROJECT / "sources.lock.json").read_text())
    for name, item in lock["upstream"].items():
        target = PROJECT / item["path"]
        if not target.resolve().is_relative_to(PROJECT / "upstream"):
            raise RuntimeError("upstream destination escapes project")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--branch",
                    item["tag"],
                    item["repository"],
                    str(target),
                ],
                check=True,
            )
        print(f"checking {name}: {item['commit']}", flush=True)
    subprocess.run(
        [sys.executable, str(PROJECT / "scripts/verify_sources.py")], check=True
    )


if __name__ == "__main__":
    main()
