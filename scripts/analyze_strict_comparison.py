"""Paired source-cluster analysis of fixed-budget, fresh-cache repeated runs."""

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from analyze_round2 import percentile


def cluster_intervals(pairs, draws=1500):
    clusters = defaultdict(list)
    for strict, block in pairs:
        clusters[strict["source_id"]].append((strict, block))
    groups = list(clusters.values())
    rng = random.Random(22026)
    quality, means, tails = [], [], []
    for _ in range(draws):
        sample = [p for group in rng.choices(groups, k=len(groups)) for p in group]
        a = [s["client_ttft_ms"] for s, _ in sample]
        b = [r["client_ttft_ms"] for _, r in sample]
        quality.append(
            statistics.mean(
                int(s["task_correct"]) - int(r["task_correct"]) for s, r in sample
            )
        )
        means.append(statistics.mean(a) / statistics.mean(b) - 1)
        tails.append(percentile(a, 0.95) / percentile(b, 0.95) - 1)
    return {
        name: [percentile(values, 0.025), percentile(values, 0.975)]
        for name, values in (
            ("quality_gain", quality),
            ("mean_ttft_change", means),
            ("p95_ttft_change", tails),
        )
    }


def summarize(rows, references):
    eligible = [
        r for r in rows if references[r["session_id"], r["version"]]["task_correct"]
    ]
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
            r["task_correct"]
            and not references[r["session_id"], r["version"]]["task_correct"]
            for r in rows
        ),
        "mean_ttft_ms": statistics.mean(r["client_ttft_ms"] for r in rows),
        "p50_ttft_ms": statistics.median(r["client_ttft_ms"] for r in rows),
        "p95_ttft_ms": percentile([r["client_ttft_ms"] for r in rows], 0.95),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    root = args.run
    progress = json.loads((root / "progress.json").read_text())
    if progress["status"] != "completed":
        raise ValueError("comparison must finish before final analysis")
    calibration = json.loads((root / "calibration.json").read_text())
    ratio = calibration["selected_strict_ratio_percent"]
    data = json.loads((root / "rep0_strict/input.json").read_text())
    sessions = {s["session_id"]: s for s in data["sessions"]}
    native = json.loads((root / "rep0_strict/results.json").read_text())
    refs = {(r["session_id"], r["version"]): r for r in native if r["arm"] == "full"}
    all_rows, pairs, repetitions = {"strict": [], "block": []}, [], []
    by_rep = {}
    for repeat in (0, 1):
        block_path, strict_path = (
            root / f"rep{repeat}_block",
            root / f"rep{repeat}_strict",
        )
        keyed = {}
        for kind, path in (("block", block_path), ("strict", strict_path)):
            manifest = json.loads((path / "manifest.json").read_text())
            assert manifest["status"] == "completed"
            assert manifest["config"]["precheck_kv_write"]
            assert manifest["config"]["strict_causal"] == (kind == "strict")
            rows = [
                r
                for r in json.loads((path / "results.json").read_text())
                if r["arm"].startswith("cacheblend")
            ]
            expected_arm = f"cacheblend{ratio}" if kind == "strict" else "cacheblend15"
            assert len(rows) == 288 and {r["arm"] for r in rows} == {expected_arm}
            keyed[kind] = {(r["session_id"], r["version"]): r for r in rows}
            all_rows[kind] += rows
            repetitions.append(
                {"repeat": repeat, "method": kind, **summarize(rows, refs)}
            )
        assert set(keyed["strict"]) == set(refs) == set(keyed["block"])
        by_rep[repeat] = keyed
        pairs += [(keyed["strict"][key], keyed["block"][key]) for key in refs]
    summary = {
        "full": summarize([r for r in native if r["arm"] == "full"], refs),
        "prefix": summarize([r for r in native if r["arm"] == "prefix"], refs),
        **{kind: summarize(rows, refs) for kind, rows in all_rows.items()},
    }
    comparison = {
        "quality_gain": summary["strict"]["accuracy"] - summary["block"]["accuracy"],
        "mean_ttft_change": summary["strict"]["mean_ttft_ms"]
        / summary["block"]["mean_ttft_ms"]
        - 1,
        "p95_ttft_change": summary["strict"]["p95_ttft_ms"]
        / summary["block"]["p95_ttft_ms"]
        - 1,
        "cluster_95ci": cluster_intervals(pairs),
        "strict_quality_repeat_flips": sum(
            by_rep[0]["strict"][k]["task_correct"]
            != by_rep[1]["strict"][k]["task_correct"]
            for k in refs
        ),
        "block_quality_repeat_flips": sum(
            by_rep[0]["block"][k]["task_correct"]
            != by_rep[1]["block"][k]["task_correct"]
            for k in refs
        ),
    }
    families = {}
    for family in sorted({r["family"] for r in native}):
        families[family] = {
            kind: summarize([r for r in rows if r["family"] == family], refs)
            for kind, rows in all_rows.items()
        }
        for arm in ("full", "prefix"):
            families[family][arm] = summarize(
                [r for r in native if r["family"] == family and r["arm"] == arm], refs
            )
    cases = []
    for key in refs:
        s0, b0 = by_rep[0]["strict"][key], by_rep[0]["block"][key]
        if refs[key]["task_correct"] and (
            not s0["task_correct"] or not b0["task_correct"]
        ):
            session = sessions[key[0]]
            cases.append(
                {
                    "session_id": key[0],
                    "source_id": session["source_id"],
                    "version": key[1],
                    "family": session["family"],
                    "expected_value": s0["expected_value"],
                    "seed_expected_value": session["versions"][0]["expected_value"],
                    "previous_expected_value": session["versions"][key[1] - 1][
                        "expected_value"
                    ],
                    "definition": session["versions"][key[1]]["definition"],
                    "metadata": session["metadata"],
                    "strict_answer": s0["generated_text"],
                    "block_answer": b0["generated_text"],
                    "strict_correct": s0["task_correct"],
                    "block_correct": b0["task_correct"],
                }
            )
    result = {
        "calibration": calibration,
        "summary": summary,
        "comparison": comparison,
        "repetitions": repetitions,
        "families": families,
    }
    (root / "strict_summary.json").write_text(json.dumps(result, indent=2))
    (root / "strict_error_cases.json").write_text(json.dumps(cases, indent=2))
    lines = [
        "# 严格因果 mask：独立扩展结果",
        "",
        "固定小对照以外 96 个来源，仅 early8k 长后缀布局；每来源三次编辑。"
        "两次服务重启重复，严格/block 的运行顺序交换；重复不增加独立来源数。",
        f"严格 mask 的比例 {ratio}% 只按开发小对照 TTFT 选定，"
        "block mask 比例固定 15%；两组都启用相同的检查层前 KV 写回修正。",
        "每组每轮 288 事件；full/prefix 只重测一轮，复用记录未重复计入。",
        "",
        "| 方法 | 观测数/独立来源 | 正确率 | 缓存新增错误 | 平均TTFT ms | P95 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, group in summary.items():
        lines.append(
            f"| {name} | {group['observations']}/{group['sources']} "
            f"| {group['accuracy']:.1%} | {group['cache_induced_errors']} "
            f"| {group['mean_ttft_ms']:.2f} | {group['p95_ttft_ms']:.2f} |"
        )
    lines += [
        "",
        f"正确率差值 {100 * comparison['quality_gain']:+.2f} 个百分点；"
        f"平均 TTFT 变化 {comparison['mean_ttft_change']:+.2%}；"
        f"P95 变化 {comparison['p95_ttft_change']:+.2%}。",
        f"按来源聚类 bootstrap 的 95% 区间：{comparison['cluster_95ci']}。",
        "",
        "这个比较用于执行校准，不是新的选择算法；不同预算用于匹配实测延迟。"
        "同预算 15% 对照与 100% 执行控制仅在开发小样本上完成。",
        "early8k 是事实前移加重复背景的衍生压力测试；未隔离整机 CPU/内存资源。"
        "两轮计时和来源 bootstrap 不能替代长期系统噪声、线上并发及原论文负载验证。",
        "完整重算本身较差的任务全部保留；输出不一致不直接当成答错。"
        "完整重算错误时近似方法答对的次数与新增错误分开保存。",
        "连续三次编辑遵循上游实际的缓存保存策略，不宣称验证了每轮近似 KV"
        "写回 CPU 后的误差累积。",
        "",
    ]
    (root / "strict_report.md").write_text("\n".join(lines))
    print(json.dumps({"summary": summary, "comparison": comparison}, indent=2))


if __name__ == "__main__":
    main()
