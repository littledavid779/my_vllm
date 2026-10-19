#!/usr/bin/env python3
"""Filter deterministic ShareGPT candidates into a model-safe workload.

The candidate file is produced by ``convert_sharegpt_to_openai.py``. This
second stage measures every user-turn prompt with the model chat template,
reserves output-token capacity, and selects the first valid conversations in
the deterministic candidate order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
from transformers import AutoTokenizer


def describe(values: list[int]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot describe an empty collection")
    return {
        "mean": mean(values),
        "p50": median(values),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "min": min(values),
        "max": max(values),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_lengths(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    lengths = []
    for index, message in enumerate(messages):
        if message["role"] != "user":
            continue
        prompt = messages[: index + 1]
        token_ids = tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
        )
        lengths.append(len(token_ids))
    if not lengths:
        raise ValueError("conversation contains no user turn")
    return lengths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--metadata-output", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--converter-seed", required=True, type=int)
    parser.add_argument("--source-file", type=Path)
    parser.add_argument("--source-url")
    parser.add_argument("--max-conversations", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--reserved-output-tokens", type=int, default=32)
    parser.add_argument("--converter-max-items", type=int, default=512)
    parser.add_argument("--converter-min-messages", type=int, default=6)
    parser.add_argument("--converter-max-messages", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
    if not isinstance(candidates, list):
        raise ValueError("candidate file must contain a JSON list")
    if args.max_conversations < 1:
        raise ValueError("max-conversations must be >= 1")
    max_prompt_tokens = args.max_model_len - args.reserved_output_tokens
    if max_prompt_tokens < 1:
        raise ValueError("reserved output tokens leave no prompt capacity")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    valid: list[tuple[dict[str, Any], int, int, int]] = []
    rejected = 0
    for item in candidates:
        lengths = prompt_lengths(tokenizer, item["messages"])
        if max(lengths) > max_prompt_tokens:
            rejected += 1
            continue
        user_turns = sum(message["role"] == "user" for message in item["messages"])
        valid.append((item, max(lengths), lengths[-1], user_turns))

    selected = valid[: args.max_conversations]
    if len(selected) < args.max_conversations:
        raise ValueError(
            f"only {len(selected)} valid conversations; "
            f"requested {args.max_conversations}"
        )

    output_payload = [item for item, _, _, _ in selected]
    max_lengths = [max_length for _, max_length, _, _ in selected]
    final_lengths = [final_length for _, _, final_length, _ in selected]
    user_turns = [turns for _, _, _, turns in selected]
    metadata = {
        "source_file": str(args.source_file) if args.source_file else None,
        "source_url": args.source_url,
        "source_sha256": sha256(args.source_file) if args.source_file else None,
        "tokenizer": args.tokenizer,
        "converter": "benchmarks/multi_turn/convert_sharegpt_to_openai.py",
        "workload_builder": "scripts/build_direct_sharegpt_workload.py",
        "converter_seed": args.converter_seed,
        "converter_max_items": args.converter_max_items,
        "converter_min_messages": args.converter_min_messages,
        "converter_max_messages": args.converter_max_messages,
        "converter_content_filter": (
            "every retained message contains at least one non-ASCII character "
            "(existing converter behavior)"
        ),
        "burstgpt_matching": False,
        "arrival_timestamps_replayed": False,
        "candidate_count": len(candidates),
        "valid_candidate_count": len(valid),
        f"rejected_over_{max_prompt_tokens}": rejected,
        "selection": (
            f"first {args.max_conversations} valid conversations in the "
            f"deterministic seed-{args.converter_seed} candidate order"
        ),
        "num_conversations": len(selected),
        "max_prompt_tokens_limit": max_prompt_tokens,
        "reserved_output_tokens": args.reserved_output_tokens,
        "max_prompt_tokens": describe(max_lengths),
        "final_prompt_tokens": describe(final_lengths),
        "user_turns": describe(user_turns),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.metadata_output.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    print(f"wrote {args.metadata_output}")


if __name__ == "__main__":
    main()
