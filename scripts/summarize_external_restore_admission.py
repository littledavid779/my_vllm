# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize repeated external KV admission experiments."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any


def _median(rows: list[dict[str, Any]], name: str) -> float | None:
    values = [row[name] for row in rows if row[name] is not None]
    return median(values) if values else None


def _show(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def summarize(root: Path) -> list[dict[str, Any]]:
    rows = []
    for report_path in sorted(root.rglob("report.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        deltas = report.get("deltas", {})
        configuration = report.get("configuration", {})
        rows.append(
            {
                "arm": report_path.parent.name,
                "repetition": report_path.parent.parent.name,
                "status": report.get("status", "MISSING"),
                "stores": deltas.get("pre_final_store_completed_blocks"),
                "target_stores": deltas.get("target_store_actual_blocks"),
                "polluter_stores": deltas.get("polluter_store_actual_blocks"),
                "polluter_dropped": deltas.get("polluter_store_dropped_blocks"),
                "loads": deltas.get("final_load_completed_blocks"),
                "promotion_policy": configuration.get("promotion_policy"),
                "promotion_candidate_blocks": deltas.get(
                    "final_promotion_candidate_blocks"
                ),
                "restore_decisions": deltas.get("final_promotion_restore_decisions"),
                "recompute_decisions": deltas.get(
                    "final_promotion_recompute_decisions"
                ),
                "d2h_mib": (deltas.get("pre_final_gpu_to_cpu_bytes", 0) / 1024 / 1024),
                "d2h_ms": deltas.get("pre_final_gpu_to_cpu_time_s", 0) * 1000,
                "h2d_mib": (deltas.get("final_cpu_to_gpu_bytes", 0) / 1024 / 1024),
                "h2d_ms": deltas.get("final_cpu_to_gpu_time_s", 0) * 1000,
                "useful_store_ratio": deltas.get("loaded_blocks_per_completed_store"),
                "recomputed_blocks": deltas.get("estimated_recomputed_target_blocks"),
                "final_latency_ms": deltas.get("final_target_latency_ms"),
                "latency_sum_ms": deltas.get("request_latency_sum_ms"),
            }
        )
    return rows


def render_markdown(rows: list[dict[str, Any]]) -> str:
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_arm[row["arm"]].append(row)

    lines = [
        "# External restore admission comparison",
        "",
        "| arm | pass | n | stores | P stores | P dropped | loads | promotion | "
        "R/C decisions | D2H MiB | useful-store | final A median ms [min, max] | "
        "request sum median ms |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for arm, group in sorted(by_arm.items()):
        final_latencies = [
            row["final_latency_ms"]
            for row in group
            if row["final_latency_ms"] is not None
        ]
        final_range = (
            f"{_show(_median(group, 'final_latency_ms'))} "
            f"[{_show(min(final_latencies))}, {_show(max(final_latencies))}]"
            if final_latencies
            else "n/a"
        )
        useful_ratio = _median(group, "useful_store_ratio")
        lines.append(
            f"| {arm} | "
            f"{sum(row['status'] == 'PASS' for row in group)}/{len(group)} | "
            f"{len(group)} | {_show(_median(group, 'stores'), 0)} | "
            f"{_show(_median(group, 'polluter_stores'), 0)} | "
            f"{_show(_median(group, 'polluter_dropped'), 0)} | "
            f"{_show(_median(group, 'loads'), 0)} | "
            f"{group[0]['promotion_policy'] or 'none'} | "
            f"{_show(_median(group, 'restore_decisions'), 0)}/"
            f"{_show(_median(group, 'recompute_decisions'), 0)} | "
            f"{_show(_median(group, 'd2h_mib'))} | "
            f"{_show(useful_ratio * 100 if useful_ratio is not None else None)}% | "
            f"{final_range} | {_show(_median(group, 'latency_sum_ms'))} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = summarize(args.root)
    (args.root / "summary.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    markdown = render_markdown(rows)
    (args.root / "summary.md").write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
