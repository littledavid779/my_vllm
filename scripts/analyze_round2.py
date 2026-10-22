"""Paired, source-clustered screening analysis; no model-output filtering."""

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path


def percentile(values, q):
    values = sorted(values)
    index = (len(values) - 1) * q
    lo, hi = int(index), min(int(index) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def clustered_gap(group, references):
    clusters = defaultdict(list)
    for row in group:
        ref = references[row["session_id"], row["version"]]
        clusters[row["source_id"]].append(
            int(row["task_correct"]) - int(ref["task_correct"])
        )
    counts = [(sum(v), len(v)) for v in clusters.values()]
    rng = random.Random(22026)
    bootstrap = []
    for _ in range(1000):
        sample = rng.choices(counts, k=len(counts))
        bootstrap.append(sum(s for s, _ in sample) / sum(n for _, n in sample))
    return [percentile(bootstrap, 0.025), percentile(bootstrap, 0.975)]


def summarize(rows, references, placement, family, arm):
    group = [
        r
        for r in rows
        if r["arm"] == arm
        and (placement == "all" or r["placement"] == placement)
        and (family == "all" or r["family"] == family)
    ]
    eligible = [
        r for r in group if references[r["session_id"], r["version"]]["task_correct"]
    ]
    errors = [r for r in eligible if not r["task_correct"]]
    gains = [
        r
        for r in group
        if r["task_correct"]
        and not references[r["session_id"], r["version"]]["task_correct"]
    ]
    times = [r["client_ttft_ms"] for r in group]
    return {
        "placement": placement,
        "family": family,
        "arm": arm,
        "events": len(group),
        "sources": len({r["source_id"] for r in group}),
        "task_accuracy": statistics.mean(r["task_correct"] for r in group),
        "reference_correct": len(eligible),
        "cache_induced_errors": len(errors),
        "cache_induced_error_sources": len({r["source_id"] for r in errors}),
        "conditional_error_rate": len(errors) / len(eligible) if eligible else None,
        "correct_when_reference_wrong": len(gains),
        "quality_gap_vs_full_95ci": clustered_gap(group, references),
        "output_disagreements": sum(
            r["generated_text"]
            != references[r["session_id"], r["version"]]["generated_text"]
            for r in group
        ),
        "ttft_mean_ms": statistics.mean(times),
        "ttft_p50_ms": statistics.median(times),
        "ttft_p95_ms": percentile(times, 0.95),
        "main_request_ttft_mean_ms": statistics.mean(
            r["main_request_ttft_ms"] for r in group
        ),
        "new_chunk_prepare_mean_ms": statistics.mean(
            r["new_chunk_prepare_ms"] for r in group
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    root = args.run
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("only analyze a complete run")
    rows = json.loads((root / "results.json").read_text())
    data = json.loads((root / "input.json").read_text())
    sessions = {s["session_id"]: s for s in data["sessions"]}
    layout = []
    for placement in ("natural4k", "early8k"):
        group = [s for s in sessions.values() if s["placement"] == placement]
        layout.append(
            {
                "placement": placement,
                "mean_prompt_tokens": statistics.mean(
                    s["prompt_tokens"] for s in group
                ),
                "mean_suffix_tokens": statistics.mean(
                    s["suffix_tokens"] for s in group
                ),
                "mean_edit_fraction": statistics.mean(
                    s["editable_token_span"][0] / s["prompt_tokens"] for s in group
                ),
            }
        )
    (root / "layout.json").write_text(json.dumps(layout, indent=2))
    arms = manifest["config"]["arms"]
    references = {
        (r["session_id"], r["version"]): r for r in rows if r["arm"] == "full"
    }
    expected = {
        (s["session_id"], version)
        for s in data["sessions"]
        for version in range(1, len(s["versions"]))
    }
    for arm in arms:
        group = [r for r in rows if r["arm"] == arm]
        assert len(group) == len(expected)
        assert {(r["session_id"], r["version"]) for r in group} == expected
    summary = [
        summarize(rows, references, placement, family, arm)
        for placement in ("all", "natural4k", "early8k")
        for family in ("all", *sorted({r["family"] for r in rows}))
        for arm in arms
    ]
    (root / "round2_summary.json").write_text(json.dumps(summary, indent=2))
    with (root / "round2_summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    keyed = {(r["arm"], r["session_id"], r["version"]): r for r in rows}
    error_cases = []
    for key, ref in references.items():
        sid, version = key
        if not ref["task_correct"]:
            continue
        approximate = {
            a: keyed[a, sid, version] for a in arms if a.startswith("cacheblend")
        }
        if all(r["task_correct"] for r in approximate.values()):
            continue
        session = sessions[sid]
        error_cases.append(
            {
                "session_id": sid,
                "source_id": session["source_id"],
                "family": session["family"],
                "placement": session["placement"],
                "version": version,
                "expected_value": ref["expected_value"],
                "previous_expected_value": session["versions"][version - 1][
                    "expected_value"
                ],
                "seed_expected_value": session["versions"][0]["expected_value"],
                "definition": session["versions"][version]["definition"],
                "previous_definition": session["versions"][version - 1]["definition"],
                "metadata": session["metadata"],
                "arms": {
                    a: {
                        k: r[k]
                        for k in ("task_correct", "generated_text", "client_ttft_ms")
                    }
                    for a, r in approximate.items()
                },
            }
        )
    (root / "error_cases.json").write_text(json.dumps(error_cases, indent=2))
    recovery = []
    for placement in ("natural4k", "early8k"):
        low_errors = [
            r
            for r in error_cases
            if r["placement"] == placement
            and not r["arms"]["cacheblend5"]["task_correct"]
        ]
        recovery.append(
            {
                "placement": placement,
                "errors_at_5pct": len(low_errors),
                "fixed_at_15pct": sum(
                    r["arms"]["cacheblend15"]["task_correct"] for r in low_errors
                ),
                "fixed_at_30pct": sum(
                    r["arms"]["cacheblend30"]["task_correct"] for r in low_errors
                ),
            }
        )
    (root / "budget_recovery.json").write_text(json.dumps(recovery, indent=2))
    budget_steps = []
    for placement in ("natural4k", "early8k"):
        for lower, higher in (
            ("cacheblend5", "cacheblend15"),
            ("cacheblend15", "cacheblend30"),
        ):
            low = next(
                g
                for g in summary
                if g["placement"] == placement
                and g["family"] == "all"
                and g["arm"] == lower
            )
            high = next(
                g
                for g in summary
                if g["placement"] == placement
                and g["family"] == "all"
                and g["arm"] == higher
            )
            selected = [
                r for r in rows if r["arm"] == higher and r["placement"] == placement
            ]
            lower_refs = {
                (r["session_id"], r["version"]): r for r in rows if r["arm"] == lower
            }
            budget_steps.append(
                {
                    "placement": placement,
                    "lower": lower,
                    "higher": higher,
                    "accuracy_gain": high["task_accuracy"] - low["task_accuracy"],
                    "accuracy_gain_95ci": clustered_gap(selected, lower_refs),
                    "mean_ttft_change": high["ttft_mean_ms"] / low["ttft_mean_ms"] - 1,
                    "p95_ttft_change": high["ttft_p95_ms"] / low["ttft_p95_ms"] - 1,
                }
            )
    (root / "budget_steps.json").write_text(json.dumps(budget_steps, indent=2))
    by_edit = []
    for placement in ("natural4k", "early8k"):
        for version in (1, 2, 3):
            for arm in arms:
                group = [
                    r
                    for r in rows
                    if r["placement"] == placement
                    and r["version"] == version
                    and r["arm"] == arm
                ]
                eligible = [
                    r
                    for r in group
                    if references[r["session_id"], r["version"]]["task_correct"]
                ]
                by_edit.append(
                    {
                        "placement": placement,
                        "version": version,
                        "arm": arm,
                        "accuracy": statistics.mean(r["task_correct"] for r in group),
                        "reference_correct": len(eligible),
                        "cache_induced_errors": sum(
                            not r["task_correct"] for r in eligible
                        ),
                    }
                )
    (root / "by_edit.json").write_text(json.dumps(by_edit, indent=2))
    lines = [
        "# 第二轮：官方 CacheBlend 研究空间筛查",
        "",
        "108 个独立来源，每个来源两种布局，各连续三次等长编辑；"
        "每种方法 648 次编辑测量，共 3240 次。不是 3240 个独立样本。",
        "任务、输入改造与运行边界见仓库 docs/ROUND2_PROTOCOL.md。"
        "CacheBlend 临时准备新 B′ 的时间已计入客户端 TTFT。",
        "",
        "| 布局 | 类别 | 方法 | 正确率 | 全算正确 | 缓存新增错误/来源数 "
        "| TTFT均值 ms | P95 ms |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for g in summary:
        if g["placement"] == "all":
            continue
        lines.append(
            f"| {g['placement']} | {g['family']} | {g['arm']} "
            f"| {g['task_accuracy']:.1%} | {g['reference_correct']} "
            f"| {g['cache_induced_errors']}/{g['cache_induced_error_sources']} "
            f"| {g['ttft_mean_ms']:.2f} | {g['ttft_p95_ms']:.2f} |"
        )
    lines += ["", "## 预算响应", ""]
    for g in recovery:
        lines.append(
            f"- {g['placement']}：5% 预算新增 {g['errors_at_5pct']} 个错误；"
            f"15% 修复其中 {g['fixed_at_15pct']} 个，"
            f"30% 修复其中 {g['fixed_at_30pct']} 个。"
        )
    lines += [
        "",
        "## 解释边界",
        "",
        "保真度变化和答错分开记录；错误清单只列完整重算正确的事件，"
        "汇总正确率仍使用全部事件。近似方法在完整重算错误时答对的次数也保留。",
        "质量差值置信区间在 JSON/CSV 中，按源样本聚类 bootstrap 1000 次；"
        "它反映本轮选定来源的抽样不确定性，不能代表整个公开榜单。",
        "early8k 同时改变证据位置、长度和背景分布，不能单独归因于长度。"
        "长背景重复；未按模型质量筛题；每次请求只计时一次。",
        "GPU 0 有其他任务，未隔离 CPU/内存资源；GPU 1 每 10 条链采样并检查"
        "外来进程。延迟只用于筛查，不能替代重复测量和线上并发验证。",
        "官方 sparse 后端尚未扫描；预算改善并不等于我们已提出改进算法。",
        "按编辑轮次的结果不构成误差累积因果证据：题目答案、修改内容与轮次"
        "同时变化，而且上游不保证每次把近似修复后的 C 刷新成下一次缓存。",
        "",
    ]
    (root / "round2_report.md").write_text("\n".join(lines))
    print(
        json.dumps(
            {
                "summary_groups": len(summary),
                "error_cases": len(error_cases),
                "budget_recovery": recovery,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
