"""Take a deterministic source subset, without inspecting model results."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-family", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("need a new output")
    data = json.loads(args.input.read_text())
    selected = set()
    for family in sorted({s["family"] for s in data["sessions"]}):
        ids = sorted(
            {s["source_id"] for s in data["sessions"] if s["family"] == family}
        )
        selected.update(ids[: args.per_family])
    data["sessions"] = [s for s in data["sessions"] if s["source_id"] in selected]
    data["backend_probe"] = {
        "selection": "first sorted four source IDs per family, no response filtering",
        "source_ids": sorted(selected),
        "parent_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
    }
    data["protocol"]["independent_sources"] = len(selected)
    args.output.write_text(json.dumps(data))
    args.output.with_suffix(".lock.json").write_text(
        json.dumps(
            {
                "input_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "probe": data["backend_probe"],
                "builder_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
    )
    print(json.dumps({"sources": len(selected), "chains": len(data["sessions"])}))


if __name__ == "__main__":
    main()
