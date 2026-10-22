"""Select long-suffix sources outside the fixed pilot, without response filtering."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("need a new output")
    data = json.loads(args.input.read_text())
    pilot = json.loads(args.pilot.read_text())
    excluded = {s["source_id"] for s in pilot["sessions"]}
    data["sessions"] = [
        s
        for s in data["sessions"]
        if s["source_id"] not in excluded and s["placement"] == "early8k"
    ]
    assert len(data["sessions"]) == 96
    assert len({s["source_id"] for s in data["sessions"]}) == 96
    data["protocol"] = {
        "phase": "strict causal calibration extension",
        "sources": 96,
        "selection": "all early8k sources outside the fixed 12-source pilot",
        "response_filtering": False,
        "edits": 3,
        "parent_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "pilot_sha256": hashlib.sha256(args.pilot.read_bytes()).hexdigest(),
        "caveat": "Sources used in round2 release screening; no tuning here.",
    }
    args.output.write_text(json.dumps(data))
    args.output.with_suffix(".lock.json").write_text(
        json.dumps(
            {
                "input_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "protocol": data["protocol"],
                "builder_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
    )
    print(json.dumps({"sources": 96, "chains": 96, "edits_per_arm": 288}))


if __name__ == "__main__":
    main()
