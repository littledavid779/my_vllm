#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


NUMERIC_FIELDS = (
    "ttft_ms",
    "tpot_ms",
    "latency_ms",
    "e2e_ms",
    "input_num_tokens",
    "input_num_turns",
    "output_num_tokens",
    "approx_cached_percent",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two benchmark request_summary.csv files."
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument(
        "--join-key",
        choices=("conversation_turn", "request_id"),
        default="conversation_turn",
    )
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args()


def load_rows(path: str) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value else 0.0


def row_key(row: dict[str, str], join_key: str) -> tuple[str, ...]:
    if join_key == "request_id":
        return (row.get("request_id", ""),)
    return (
        row.get("conversation_id", ""),
        row.get("input_num_turns", ""),
        row.get("input_num_tokens", ""),
    )


def input_bucket(tokens: float) -> str:
    if tokens < 128:
        return "tokens_lt_128"
    if tokens < 512:
        return "tokens_128_511"
    if tokens < 1024:
        return "tokens_512_1023"
    return "tokens_ge_1024"


def turn_bucket(turns: float) -> str:
    if turns <= 1:
        return "turn_1"
    if turns <= 3:
        return "turn_2_3"
    return "turn_ge_4"


def cached_bucket(cached_percent: float) -> str:
    if cached_percent <= 0:
        return "cached_0"
    if cached_percent < 50:
        return "cached_0_50"
    if cached_percent < 90:
        return "cached_50_90"
    return "cached_ge_90"


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = round((len(sorted_values) - 1) * pct)
    return sorted_values[idx]


def compare_rows(
    baseline_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
    join_key: str,
) -> list[dict[str, Any]]:
    baseline_by_key = {row_key(row, join_key): row for row in baseline_rows}
    candidate_by_key = {row_key(row, join_key): row for row in candidate_rows}
    rows = []
    for key in sorted(baseline_by_key.keys() & candidate_by_key.keys()):
        base = baseline_by_key[key]
        cand = candidate_by_key[key]
        row: dict[str, Any] = {
            "key": "|".join(key),
            "conversation_id": cand.get("conversation_id", base.get("conversation_id", "")),
            "input_num_turns": to_float(cand, "input_num_turns"),
            "input_num_tokens": to_float(cand, "input_num_tokens"),
            "approx_cached_percent": to_float(cand, "approx_cached_percent"),
        }
        for field in NUMERIC_FIELDS:
            row[f"baseline_{field}"] = to_float(base, field)
            row[f"candidate_{field}"] = to_float(cand, field)
            row[f"delta_{field}"] = row[f"candidate_{field}"] - row[f"baseline_{field}"]
        row["input_bucket"] = input_bucket(row["input_num_tokens"])
        row["turn_bucket"] = turn_bucket(row["input_num_turns"])
        row["cached_bucket"] = cached_bucket(row["approx_cached_percent"])
        rows.append(row)
    return rows


def aggregate(rows: list[dict[str, Any]], bucket_field: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[bucket_field])].append(row)
    out = []
    for bucket, bucket_rows in sorted(groups.items()):
        deltas = [float(row["delta_ttft_ms"]) for row in bucket_rows]
        out.append({
            "bucket": bucket,
            "count": len(bucket_rows),
            "avg_delta_ttft_ms": mean(deltas) if deltas else 0.0,
            "p90_delta_ttft_ms": percentile(deltas, 0.90),
            "worse_count": sum(delta > 0 for delta in deltas),
            "better_count": sum(delta < 0 for delta in deltas),
        })
    return out


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    if not rows:
        return "(no rows)"
    lines = ["| " + " | ".join(fields) + " |"]
    lines.append("| " + " | ".join("---" for _ in fields) + " |")
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines)


def write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "key",
        "conversation_id",
        "input_num_turns",
        "input_num_tokens",
        "approx_cached_percent",
        "input_bucket",
        "turn_bucket",
        "cached_bucket",
        "baseline_ttft_ms",
        "candidate_ttft_ms",
        "delta_ttft_ms",
        "baseline_e2e_ms",
        "candidate_e2e_ms",
        "delta_e2e_ms",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_markdown(path: str, args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    deltas = [float(row["delta_ttft_ms"]) for row in rows]
    summary = [{
        "matched_requests": len(rows),
        "avg_delta_ttft_ms": mean(deltas) if deltas else 0.0,
        "p50_delta_ttft_ms": percentile(deltas, 0.50),
        "p90_delta_ttft_ms": percentile(deltas, 0.90),
        "worse_count": sum(delta > 0 for delta in deltas),
        "better_count": sum(delta < 0 for delta in deltas),
    }]
    top_worse = sorted(rows, key=lambda row: row["delta_ttft_ms"], reverse=True)[: args.top_n]
    top_better = sorted(rows, key=lambda row: row["delta_ttft_ms"])[: args.top_n]
    request_fields = [
        "conversation_id",
        "input_num_turns",
        "input_num_tokens",
        "approx_cached_percent",
        "baseline_ttft_ms",
        "candidate_ttft_ms",
        "delta_ttft_ms",
    ]
    bucket_fields = ["bucket", "count", "avg_delta_ttft_ms", "p90_delta_ttft_ms", "worse_count", "better_count"]
    content = "\n\n".join([
        "# Request Summary Diff",
        f"baseline: `{args.baseline_name}`",
        f"candidate: `{args.candidate_name}`",
        "## Overall",
        markdown_table(summary, list(summary[0].keys())),
        "## By Input Tokens",
        markdown_table(aggregate(rows, "input_bucket"), bucket_fields),
        "## By Turns",
        markdown_table(aggregate(rows, "turn_bucket"), bucket_fields),
        "## By Cached Percent",
        markdown_table(aggregate(rows, "cached_bucket"), bucket_fields),
        "## Top Candidate Slower",
        markdown_table(top_worse, request_fields),
        "## Top Candidate Faster",
        markdown_table(top_better, request_fields),
        "",
    ])
    out_path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = compare_rows(load_rows(args.baseline), load_rows(args.candidate), args.join_key)
    if not rows:
        raise SystemExit("No matched requests found.")
    write_csv(args.output_csv, rows)
    write_markdown(args.output_md, args, rows)
    print(f"matched {len(rows)} requests")
    print(f"wrote {args.output_csv}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
