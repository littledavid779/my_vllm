"""Reuse the archived public-source diagnostic, adding official chunk separators."""

import argparse
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    data = json.loads(args.source.read_text())
    tokenizer = AutoTokenizer.from_pretrained(data["model"], local_files_only=True)
    # This reproduces SegmentTokenDatabase's separator encoding convention.
    separator = tokenizer.encode("# #")[1:]
    if not separator:
        raise ValueError("empty upstream separator")
    warm = [
        tokenizer.encode(
            f"Warmup archive {name}. Its marker is {name}.\n" * 80,
            add_special_tokens=False,
        )
        for name in ("amber", "violet", "silver")
    ]
    data["warmup_seed_ids"] = warm[0] + separator + warm[1] + separator + warm[2]
    data["warmup_edit_ids"] = warm[2] + separator + warm[0] + separator + warm[1]
    for session in data["sessions"]:
        begin, end = session["editable_token_span"]
        for version in session["versions"]:
            ids = version["token_ids"]
            for segment in (ids[:begin], ids[begin:end], ids[end:]):
                if any(
                    segment[i : i + len(separator)] == separator
                    for i in range(len(segment) - len(separator) + 1)
                ):
                    raise ValueError("separator already occurs inside source segment")
            version["token_ids"] = (
                ids[:begin] + separator + ids[begin:end] + separator + ids[end:]
            )
        session["prompt_tokens"] += 2 * len(separator)
    data["pilot_provenance"] = {
        "source_path": str(args.source.resolve()),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "adaptation": "Insert chunk separators at A/B and B/C; gold unchanged.",
        "separator_ids": separator,
        "scope": "Small middle-edit diagnostic; not the original paper benchmark.",
    }
    destination = PROJECT / "data/pilot.json"
    destination.parent.mkdir(exist_ok=True)
    if destination.exists():
        raise ValueError("refusing to overwrite pilot input")
    destination.write_text(json.dumps(data))
    locked = {
        **data["pilot_provenance"],
        "input_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "model": data["model"],
        "sessions": len(data["sessions"]),
        "source_manifest": data["source_manifest"],
    }
    (PROJECT / "data.lock.json").write_text(json.dumps(locked, indent=2))
    print(json.dumps({"sessions": len(data["sessions"]), "input": str(destination)}))


if __name__ == "__main__":
    main()
