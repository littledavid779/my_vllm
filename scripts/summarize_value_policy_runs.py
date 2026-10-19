#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

NUMERIC_FIELDS = [
    "request_count",
    "avg_ttft_ms",
    "p50_ttft_ms",
    "p90_ttft_ms",
    "avg_e2e_ms",
    "requests_per_e2e_s",
    "restore_ratio",
    "avg_restore_bytes",
    "avg_local_cached_tokens",
    "avg_shared_prefix_local_hit_tokens",
    "shared_restore_per_query",
    "shared_remove_per_query",
    "avg_effective_prefix_hit_blocks",
    "avg_effective_prefix_hit_tokens",
    "prefix_cache_zero_hit_ratio",
    "evicted_blocks",
    "avg_evicted_prefix_index",
    "avg_evicted_chain_downstream_blocks",
    "avg_evicted_apc_hit_count",
    "avg_seconds_since_last_apc_hit_on_eviction",
    "reorder_count",
    "reorder_skipped_low_hit_count",
    "reorder_skipped_low_margin_count",
    "min_reorder_score_margin",
    "reorder_time_s",
    "offload_store_threshold",
    "offload_count_store_attempts",
    "offload_tracker_size",
    "offload_lookup_query_count",
    "offload_lookup_observed_blocks",
    "offload_lookup_hit_query_count",
    "offload_lookup_hit_blocks",
    "offload_load_request_count",
    "offload_load_requested_blocks",
    "offload_load_completed_blocks",
    "offload_store_request_count",
    "offload_store_requested_blocks",
    "offload_store_eligible_blocks",
    "offload_store_dropped_blocks",
    "offload_store_actual_blocks",
    "offload_store_already_present_blocks",
    "offload_store_evicted_blocks",
    "offload_store_failed_count",
    "offload_store_completed_blocks",
    "offload_store_completion_failed_blocks",
    "offload_lifecycle_stats_enabled",
    "offload_lifecycle_tracked_blocks",
    "offload_lifecycle_tracker_evictions",
    "offload_lifecycle_ever_stored_blocks",
    "offload_lifecycle_resident_blocks",
    "offload_lifecycle_never_looked_up_after_store_blocks",
    "offload_lifecycle_looked_up_after_store_blocks",
    "offload_lifecycle_cpu_hit_after_store_blocks",
    "offload_lifecycle_looked_up_after_eviction_blocks",
    "offload_lifecycle_evicted_before_first_lookup_blocks",
    "offload_lifecycle_loaded_blocks",
]

FIELDNAMES = [
    "row_type",
    "policy",
    "run_id",
    "seed",
    "run_dir",
    "variant",
    *NUMERIC_FIELDS,
]

MARKDOWN_FIELDS = [
    "policy",
    "run_id",
    "request_count",
    "avg_ttft_ms",
    "p90_ttft_ms",
    "avg_effective_prefix_hit_blocks",
    "prefix_cache_zero_hit_ratio",
    "evicted_blocks",
    "avg_evicted_prefix_index",
    "avg_evicted_chain_downstream_blocks",
    "avg_evicted_apc_hit_count",
    "reorder_count",
    "reorder_skipped_low_hit_count",
    "reorder_skipped_low_margin_count",
    "min_reorder_score_margin",
    "reorder_time_s",
    "offload_store_threshold",
    "offload_count_store_attempts",
    "offload_lookup_hit_blocks",
    "offload_load_requested_blocks",
    "offload_load_completed_blocks",
    "offload_store_requested_blocks",
    "offload_store_eligible_blocks",
    "offload_store_dropped_blocks",
    "offload_store_actual_blocks",
    "offload_store_already_present_blocks",
    "offload_store_evicted_blocks",
    "offload_store_failed_count",
    "offload_store_completed_blocks",
    "offload_store_completion_failed_blocks",
    "offload_lifecycle_stats_enabled",
    "offload_lifecycle_tracked_blocks",
    "offload_lifecycle_tracker_evictions",
    "offload_lifecycle_ever_stored_blocks",
    "offload_lifecycle_resident_blocks",
    "offload_lifecycle_never_looked_up_after_store_blocks",
    "offload_lifecycle_looked_up_after_store_blocks",
    "offload_lifecycle_cpu_hit_after_store_blocks",
    "offload_lifecycle_looked_up_after_eviction_blocks",
    "offload_lifecycle_evicted_before_first_lookup_blocks",
    "offload_lifecycle_loaded_blocks",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize value-policy benchmark runs into CSV and Markdown."
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        help="Run directory containing request_summary.csv. Can be repeated.",
    )
    parser.add_argument(
        "--root",
        action="append",
        default=[],
        help="Recursively scan for run directories under this root.",
    )
    parser.add_argument(
        "--output-csv",
        default="policy_compare_summary.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--output-md",
        default="policy_compare_summary.md",
        help="Output Markdown path.",
    )
    parser.add_argument(
        "--min-restore-bytes",
        type=int,
        default=1,
        help="Restore byte threshold used when computing restore_ratio.",
    )
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def to_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value else 0.0


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = round((len(sorted_values) - 1) * pct)
    return sorted_values[idx]


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def discover_run_dirs(args: argparse.Namespace) -> list[Path]:
    seen: set[Path] = set()
    run_dirs: list[Path] = []
    for item in args.run_dir:
        path = Path(item)
        if path not in seen:
            seen.add(path)
            run_dirs.append(path)
    for item in args.root:
        root = Path(item)
        for request_summary in root.rglob("request_summary.csv"):
            path = request_summary.parent
            if path not in seen:
                seen.add(path)
                run_dirs.append(path)
    return sorted(run_dirs)


def infer_seed(run_dir: Path) -> str:
    for part in reversed(run_dir.parts):
        match = re.search(r"seed[_-]?(\d+)", part)
        if match:
            return match.group(1)
    return ""


def value_stats_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    stats = payload.get("stats", payload)
    return stats if isinstance(stats, dict) else {}


def summarize_run(run_dir: Path, min_restore_bytes: int) -> dict[str, Any]:
    request_rows = load_csv(run_dir / "request_summary.csv")
    analysis = load_json(run_dir / "analysis_summary.json")
    policy_config = load_json(run_dir / "gpu_eviction_policy.json")
    value_stats = value_stats_from_payload(
        load_json(run_dir / "value_policy_stats.json")
    )

    ttfts = [to_float(row, "ttft_ms") for row in request_rows]
    e2es = [to_float(row, "e2e_ms") for row in request_rows]
    restore_rows = [
        row
        for row in request_rows
        if to_float(row, "restore_bytes") >= min_restore_bytes
    ]
    total_e2e_s = sum(e2es) / 1000.0
    request_count = len(request_rows)
    shared_blocks = analysis.get("shared_blocks", {})

    policy = (
        policy_config.get("policy")
        or value_stats.get("policy")
        or (request_rows[0].get("run_id", "") if request_rows else "")
        or run_dir.name
    )
    variant = request_rows[0].get("variant", "") if request_rows else ""

    row: dict[str, Any] = {
        "row_type": "run",
        "policy": policy,
        "run_id": run_dir.name,
        "seed": infer_seed(run_dir),
        "run_dir": str(run_dir),
        "variant": variant,
        "request_count": request_count,
        "avg_ttft_ms": mean(ttfts) if ttfts else 0.0,
        "p50_ttft_ms": percentile(ttfts, 0.50),
        "p90_ttft_ms": percentile(ttfts, 0.90),
        "avg_e2e_ms": mean(e2es) if e2es else 0.0,
        "requests_per_e2e_s": safe_div(request_count, total_e2e_s),
        "restore_ratio": safe_div(len(restore_rows), request_count),
        "avg_restore_bytes": (
            mean(to_float(row, "restore_bytes") for row in request_rows)
            if request_rows
            else 0.0
        ),
        "avg_local_cached_tokens": (
            mean(to_float(row, "num_local_cached_tokens") for row in request_rows)
            if request_rows
            else 0.0
        ),
        "avg_shared_prefix_local_hit_tokens": (
            mean(
                to_float(row, "shared_prefix_local_hit_tokens")
                for row in request_rows
            )
            if request_rows
            else 0.0
        ),
        "shared_restore_per_query": shared_blocks.get(
            "shared_restore_per_query", 0.0
        ),
        "shared_remove_per_query": shared_blocks.get("shared_remove_per_query", 0.0),
    }
    for field in NUMERIC_FIELDS:
        if field in row:
            continue
        row[field] = value_stats.get(field, 0.0)
    return row


def aggregate_by_policy(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["policy"])].append(row)

    aggregates = []
    for policy, policy_rows in sorted(grouped.items()):
        aggregate: dict[str, Any] = {
            "row_type": "aggregate",
            "policy": policy,
            "run_id": f"n={len(policy_rows)}",
            "seed": "",
            "run_dir": "",
            "variant": policy_rows[0].get("variant", ""),
        }
        for field in NUMERIC_FIELDS:
            aggregate[field] = mean(float(row.get(field, 0.0) or 0.0)
                                    for row in policy_rows)
        aggregates.append(aggregate)
    return aggregates


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDNAMES})


def markdown_table(rows: list[dict[str, Any]]) -> str:
    header = "| " + " | ".join(MARKDOWN_FIELDS) + " |"
    sep = "| " + " | ".join("---" for _ in MARKDOWN_FIELDS) + " |"
    lines = [header, sep]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(format_value(row.get(field, "")) for field in MARKDOWN_FIELDS)
            + " |"
        )
    return "\n".join(lines)


def write_markdown(
    path: Path,
    aggregate_rows: list[dict[str, Any]],
    run_rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n\n".join(
        [
            "# Value Policy Run Summary",
            "## Aggregate By Policy",
            markdown_table(aggregate_rows),
            "## Per Run",
            markdown_table(run_rows),
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dirs = discover_run_dirs(args)
    if not run_dirs:
        raise SystemExit("No run directories found.")

    run_rows = [
        summarize_run(run_dir, args.min_restore_bytes)
        for run_dir in run_dirs
        if (run_dir / "request_summary.csv").exists()
    ]
    aggregate_rows = aggregate_by_policy(run_rows)
    rows = aggregate_rows + run_rows
    write_csv(Path(args.output_csv), rows)
    write_markdown(Path(args.output_md), aggregate_rows, run_rows)
    print(f"wrote {args.output_csv}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
