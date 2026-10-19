# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize a forced restore/recompute prefix-length sweep."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


def _numbers(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [
        float(value)
        for row in rows
        if isinstance((value := row.get(key)), (int, float))
        and not isinstance(value, bool)
    ]


def _median(rows: list[dict[str, Any]], key: str) -> float | None:
    values = _numbers(rows, key)
    return median(values) if values else None


def _format(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def load_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("report.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        config = report.get("configuration", {})
        deltas = report.get("deltas", {})
        target = report.get("prompts", {}).get("target", {})
        policy = config.get("promotion_policy")
        requested = config.get("target_prompt_tokens")
        if not isinstance(policy, str) or not isinstance(requested, int):
            continue
        h2d_bytes = deltas.get("final_cpu_to_gpu_bytes", 0) or 0
        rows.append(
            {
                "report": str(path.relative_to(root)),
                "repetition": path.parent.parent.name,
                "policy": policy,
                "status": report.get("status", "MISSING"),
                "requested_tokens": requested,
                "actual_tokens": target.get("tokenized_count"),
                "candidate_blocks": deltas.get("final_promotion_candidate_blocks"),
                "final_latency_ms": deltas.get("final_target_latency_ms"),
                "h2d_mib": h2d_bytes / 1024 / 1024,
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["requested_tokens"]].append(row)

    summaries = []
    for requested, group in sorted(grouped.items()):
        restore = [row for row in group if row["policy"] == "always_restore"]
        recompute = [row for row in group if row["policy"] == "always_recompute"]
        pairs: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in group:
            pairs[row["repetition"]][row["policy"]] = row
        deltas = []
        for pair in pairs.values():
            restore_row = pair.get("always_restore")
            recompute_row = pair.get("always_recompute")
            if restore_row is None or recompute_row is None:
                continue
            restore_ms = restore_row.get("final_latency_ms")
            recompute_ms = recompute_row.get("final_latency_ms")
            if isinstance(restore_ms, (int, float)) and isinstance(
                recompute_ms, (int, float)
            ):
                deltas.append(float(recompute_ms - restore_ms))

        restore_latencies = _numbers(restore, "final_latency_ms")
        recompute_latencies = _numbers(recompute, "final_latency_ms")
        summaries.append(
            {
                "requested_tokens": requested,
                "actual_tokens": _median(group, "actual_tokens"),
                "candidate_blocks": _median(group, "candidate_blocks"),
                "restore_pass": sum(row["status"] == "PASS" for row in restore),
                "restore_runs": len(restore),
                "recompute_pass": sum(row["status"] == "PASS" for row in recompute),
                "recompute_runs": len(recompute),
                "restore_latency_median_ms": (
                    median(restore_latencies) if restore_latencies else None
                ),
                "restore_latency_min_ms": (
                    min(restore_latencies) if restore_latencies else None
                ),
                "restore_latency_max_ms": (
                    max(restore_latencies) if restore_latencies else None
                ),
                "recompute_latency_median_ms": (
                    median(recompute_latencies) if recompute_latencies else None
                ),
                "recompute_latency_min_ms": (
                    min(recompute_latencies) if recompute_latencies else None
                ),
                "recompute_latency_max_ms": (
                    max(recompute_latencies) if recompute_latencies else None
                ),
                "paired_delta_mean_ms": mean(deltas) if deltas else None,
                "paired_delta_median_ms": median(deltas) if deltas else None,
                "restore_wins": sum(delta > 0 for delta in deltas),
                "recompute_wins": sum(delta < 0 for delta in deltas),
                "ties": sum(delta == 0 for delta in deltas),
                "pairs": len(deltas),
                "restore_h2d_mib": _median(restore, "h2d_mib"),
                "recompute_h2d_mib": _median(recompute, "h2d_mib"),
            }
        )
    return summaries


def _latency(row: dict[str, Any], prefix: str) -> str:
    return (
        f"{_format(row[f'{prefix}_latency_median_ms'])} "
        f"[{_format(row[f'{prefix}_latency_min_ms'])}, "
        f"{_format(row[f'{prefix}_latency_max_ms'])}]"
    )


def render_markdown(summaries: list[dict[str, Any]]) -> str:
    lines = [
        "# External restore/recompute prefix-length sweep",
        "",
        "Paired delta = recompute latency - restore latency; positive means "
        "restore is faster.",
        "",
        "| requested | actual | blocks | pass R/C | restore median [min,max] ms "
        "| recompute median [min,max] ms | paired delta mean/median ms "
        "| wins R/C/tie | H2D MiB R/C |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['requested_tokens']} | "
            f"{_format(row['actual_tokens'], 0)} | "
            f"{_format(row['candidate_blocks'], 0)} | "
            f"{row['restore_pass']}/{row['restore_runs']} / "
            f"{row['recompute_pass']}/{row['recompute_runs']} | "
            f"{_latency(row, 'restore')} | {_latency(row, 'recompute')} | "
            f"{_format(row['paired_delta_mean_ms'])} / "
            f"{_format(row['paired_delta_median_ms'])} | "
            f"{row['restore_wins']}/{row['recompute_wins']}/{row['ties']} | "
            f"{_format(row['restore_h2d_mib'])} / "
            f"{_format(row['recompute_h2d_mib'])} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    rows = load_rows(args.root)
    summaries = summarize(rows)
    (args.root / "summary.json").write_text(
        json.dumps({"rows": rows, "lengths": summaries}, indent=2),
        encoding="utf-8",
    )
    markdown = render_markdown(summaries)
    (args.root / "summary.md").write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    return 0 if summaries else 1


if __name__ == "__main__":
    raise SystemExit(main())
