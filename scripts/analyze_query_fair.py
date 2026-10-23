"""Analyze new-source, shared-query controls with fresh native references."""

import argparse
import json
import statistics
from pathlib import Path

from analyze_query_controls import compare
from analyze_round2 import percentile
from apply_query_fair_patch import MODES
from run_http_pilot import score


def key(row):
    return row["repeat"], row["session_id"], row["version"]


def summary(rows, refs):
    eligible = [r for r in rows if refs[key(r)]["task_correct"]]
    errors = [r for r in eligible if not r["task_correct"]]
    return {
        "observations": len(rows),
        "unique_edit_events": len({(r["session_id"], r["version"]) for r in rows}),
        "sources": len({r["source_id"] for r in rows}),
        "correct": sum(r["task_correct"] for r in rows),
        "accuracy": statistics.mean(r["task_correct"] for r in rows),
        "reference_correct_observations": len(eligible),
        "cache_induced_errors": len(errors),
        "error_sources": len({r["source_id"] for r in errors}),
        "correct_when_full_wrong": sum(
            r["task_correct"] and not refs[key(r)]["task_correct"] for r in rows
        ),
        "mean_ttft_ms": statistics.mean(r["client_ttft_ms"] for r in rows),
        "p95_ttft_ms": percentile([r["client_ttft_ms"] for r in rows], 0.95),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    root = args.run
    progress = json.loads((root / "progress.json").read_text())
    if progress["status"] != "completed" or progress["restore_errors"]:
        raise ValueError("measurements and restoration must finish first")
    frozen = json.loads((root / "frozen_config.json").read_text())
    groups = {mode: [] for mode in ("full", "prefix", *MODES)}
    expected = set()
    for repeat, order in enumerate(frozen["orders"]):
        native = None
        for mode in order:
            path = root / f"rep{repeat}_{mode}"
            manifest = json.loads((path / "manifest.json").read_text())
            config = manifest["config"]
            if (
                manifest["status"] != "completed"
                or manifest["input_sha256"] != frozen["input_sha256"]["query_fair_v1"]
                or config["query_control"] != mode
                or not config["query_fair"]
                or not config["strict_causal"]
                or not config["precheck_kv_write"]
                or config["ratios"] != {"cacheblend10": 0.1}
            ):
                raise ValueError("stage differs from frozen configuration")
            audit = json.loads((path / "selection_audit.json").read_text())
            if audit["full_input_audit_calls"] != 288 or (
                mode == "boundary" and not audit["boundary_positions_verified"]
            ):
                raise ValueError("missing selection audit")
            data = json.loads((path / "input.json").read_text())
            gold = {
                (s["session_id"], v): item["expected_value"]
                for s in data["sessions"]
                for v, item in enumerate(s["versions"])
                if v
            }
            if len(gold) != 288 or (expected and set(gold) != expected):
                raise ValueError("unexpected or unpaired input events")
            expected = set(gold)
            all_rows = json.loads((path / "results.json").read_text())
            for arm in ("full", "prefix", "cacheblend10"):
                rows = [r for r in all_rows if r["arm"] == arm]
                if (
                    len(rows) != 288
                    or {(r["session_id"], r["version"]) for r in rows} != expected
                ):
                    raise ValueError("incomplete or duplicated arm")
                for row in rows:
                    target = gold[row["session_id"], row["version"]]
                    if row["expected_value"] != target or row["task_correct"] != score(
                        row["generated_text"], target
                    ):
                        raise ValueError("gold label or score mismatch")
                    row["repeat"] = repeat
                if arm == "cacheblend10":
                    groups[mode].extend(rows)
            native_rows = [r for r in all_rows if r["arm"] in ("full", "prefix")]
            if native is None:
                if manifest.get("controls_from"):
                    raise ValueError("first stage must measure fresh native controls")
                native = native_rows
                for arm in ("full", "prefix"):
                    groups[arm].extend(r for r in native if r["arm"] == arm)
            elif native != native_rows:
                raise ValueError("native references changed within repeat")
    refs = {key(r): r for r in groups["full"]}
    summaries = {m: summary(rows, refs) for m, rows in groups.items()}
    layouts, comparisons = {}, {}
    for layout in ("all", "natural4k", "early8k"):
        subset = {
            m: [r for r in rows if layout == "all" or r["placement"] == layout]
            for m, rows in groups.items()
        }
        layouts[layout] = {m: summary(rows, refs) for m, rows in subset.items()}
        comparisons[layout] = {}
        for a, b in (
            ("tail64", "query"),
            ("boundary", "query"),
            ("tail64", "boundary"),
        ):
            aa, bb = layouts[layout][a], layouts[layout][b]
            interval = compare(subset[a], subset[b])
            comparisons[layout][f"{a}_vs_{b}"] = {
                "quality_gain": aa["accuracy"] - bb["accuracy"],
                "mean_ttft_change": aa["mean_ttft_ms"] / bb["mean_ttft_ms"] - 1,
                "p95_ttft_change": aa["p95_ttft_ms"] / bb["p95_ttft_ms"] - 1,
                "source_cluster_95ci": interval,
            }
    families = {
        family: {
            m: summary([r for r in rows if r["family"] == family], refs)
            for m, rows in groups.items()
        }
        for family in sorted({r["family"] for r in refs.values()})
    }
    repeats = {
        rep: {
            m: summary([r for r in rows if r["repeat"] == rep], refs)
            for m, rows in groups.items()
        }
        for rep in (0, 1)
    }
    versions = {
        v: {
            m: summary([r for r in rows if r["version"] == v], refs)
            for m, rows in groups.items()
        }
        for v in (1, 2, 3)
    }
    indexed = {m: {key(r): r for r in rows} for m, rows in groups.items()}
    flips = {
        m: sum(
            rows[0, sid, v]["task_correct"] != rows[1, sid, v]["task_correct"]
            for sid, v in expected
        )
        for m, rows in indexed.items()
    }
    cases = [
        {
            "session_id": sid,
            "version": v,
            "family": refs[0, sid, v]["family"],
            "placement": refs[0, sid, v]["placement"],
            "gold": refs[0, sid, v]["expected_value"],
            "methods": {
                m: [
                    {
                        "repeat": rep,
                        "answer": rows[rep, sid, v]["generated_text"],
                        "correct": rows[rep, sid, v]["task_correct"],
                    }
                    for rep in (0, 1)
                ]
                for m, rows in indexed.items()
            },
        }
        for sid, v in sorted(expected)
    ]
    result = {
        "summary": summaries,
        "layouts": layouts,
        "comparisons": comparisons,
        "families": families,
        "repetitions": repeats,
        "versions": versions,
        "quality_repeat_flips": flips,
        "scope": (
            "48 new sources; paired original and constructed layouts; "
            "288 unique edits; single-flight HTTP; one model; "
            "boundary rule adaptation only"
        ),
    }
    (root / "query_fair_summary.json").write_text(json.dumps(result, indent=2))
    (root / "query_fair_cases.json").write_text(json.dumps(cases, indent=2))
    lines = [
        "# 共同问题刷新：新来源验证",
        "",
        "48 个新来源，两个布局，288 个编辑事件；每方案两次独立服务/缓存重复。"
        "full/prefix 每轮新测。",
        "TTFT 包含 B′ 准备、检索搬运、选择修复、日志与 HTTP 首 token。"
        "未包含初次播种及分词；非并发吞吐测试。",
        "",
    ]
    for layout, groups_summary in layouts.items():
        lines += [
            f"## {layout}",
            "",
            "| 方法 | 正确率 | TTFT 均值 ms | P95 ms | full 对而本组错（两轮观测） |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for mode, s in groups_summary.items():
            lines.append(
                f"| {mode} | {s['accuracy']:.2%} | {s['mean_ttft_ms']:.2f} "
                f"| {s['p95_ttft_ms']:.2f} | {s['cache_induced_errors']} |"
            )
        lines.append("")
        for name, c in comparisons[layout].items():
            ci = c["source_cluster_95ci"]["quality_gain"]
            lines.append(
                f"- {name}: 准确率差 {c['quality_gain'] * 100:+.2f} pp，"
                f"来源聚类 95% CI [{ci[0] * 100:+.2f}, {ci[1] * 100:+.2f}]；"
                f"均值 TTFT {c['mean_ttft_change']:+.2%}，"
                f"P95 {c['p95_ttft_change']:+.2%}。"
            )
        lines.append("")
    lines += [
        "boundary 为 EPIC 块首选点思路的适配：仍有 LMCache 早层探测，"
        "预算按总长 10%，不是 EPIC 原系统复现。没有实施 ProphetKV/全套强基线。",
        "natural4k 保持公开来源布局，early8k 人工移动事实并重复无事实背景；"
        "均为编辑衍生负载，不是官方 BABILong 分数。",
        "主要比较 tail64-query；其他比较和分层作探索。两次重复不增加独立来源数。"
        "三次编辑不证明近似缓存持续回写的误差累积。",
        f"各组重复间正确性翻转：{flips}。",
        "",
    ]
    (root / "query_fair_report.md").write_text("\n".join(lines))
    print(json.dumps({"summary": summaries, "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
