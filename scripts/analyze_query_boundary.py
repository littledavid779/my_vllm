"""Analyze native-question boundary controls with paired source-level intervals."""

import argparse
import json
import statistics
from pathlib import Path

from analyze_query_controls import compare
from analyze_round2 import percentile
from run_http_pilot import score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    root = args.run
    progress = json.loads((root / "progress.json").read_text())
    if progress["status"] != "completed" or progress["restore_errors"]:
        raise ValueError("run incomplete")
    frozen = json.loads((root / "frozen_config.json").read_text())
    groups = {name: [] for name in ("full", "prefix", "topk", "query", "fresh")}
    refs = {}
    for rep, order in enumerate(frozen["orders"]):
        for mode in order:
            stage = root / f"rep{rep}_{mode}"
            manifest = json.loads((stage / "manifest.json").read_text())
            if (
                manifest["status"] != "completed"
                or manifest["input_sha256"] != frozen["input_sha256"]
            ):
                raise ValueError("incomplete or mismatched stage")
            audited = json.loads((stage / "selection_audit.json").read_text())
            if audited["measured_calls"] != 288 or audited["mode"] != mode:
                raise ValueError("runtime audit mismatch")
            rows = json.loads((stage / "results.json").read_text())
            for row in rows:
                if row["task_correct"] != score(
                    row["generated_text"], row["expected_value"]
                ):
                    raise ValueError("score differs")
                row["repeat"] = rep
                if row["arm"] == "cacheblend10":
                    groups[mode].append(row)
                elif mode == order[0]:
                    groups[row["arm"]].append(row)
                    if row["arm"] == "full":
                        refs[rep, row["session_id"], row["version"]] = row
    for mode, rows in groups.items():
        if (
            len(rows) != 576
            or len({(r["repeat"], r["session_id"], r["version"]) for r in rows}) != 576
        ):
            raise ValueError(f"missing or duplicate rows: {mode}")

    def summary(rows):
        times = sorted(r["client_ttft_ms"] for r in rows)
        return {
            "observations": len(rows),
            "sources": len({r["source_id"] for r in rows}),
            "accuracy": statistics.mean(r["task_correct"] for r in rows),
            "mean_ttft_ms": statistics.mean(times),
            "p95_ttft_ms": percentile(times, 0.95),
            "cache_induced_errors": sum(
                refs[r["repeat"], r["session_id"], r["version"]]["task_correct"]
                and not r["task_correct"]
                for r in rows
            ),
            "correct_when_full_wrong": sum(
                not refs[r["repeat"], r["session_id"], r["version"]]["task_correct"]
                and r["task_correct"]
                for r in rows
            ),
        }

    result = {
        "summary": {m: summary(r) for m, r in groups.items()},
        "by_variant": {},
        "by_family": {},
        "comparisons": {},
    }
    lines = [
        "# 问题缓存边界与问法验证",
        "",
        "32 个已评测来源，3 种问法/答案表达、3 次等长编辑、"
        "两次反序重复；这是探索性机制筛查。",
        "",
        "完整重算和精确前缀缓存每轮新测一次，复用到同轮其他选择器作配对参考；不重复计入样本。",
        "",
        "topk/query/fresh 使用相同 token-ID 输入及隐式问题分段。que"
        "ry 强制修复完整问题和模板；fresh 从缓存中排除它们，由 vLLM 正常前向计算。",
        "",
        "名义预算为全输入的 10%；fresh 扣除原生问题计算位置，缓存组仍有一次正常最"
        "后 token 计算。早层探测范围也有差别，不能把名义预算视为完全等 FLOPs。",
        "",
        "| 策略 | 准确率 | 平均 TTFT ms | P95 ms | full 对而本组错 两轮观测 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for mode, s in result["summary"].items():
        lines.append(
            f"| {mode} | {s['accuracy']:.2%} | {s['mean_ttft_ms']:.2f} "
            f"| {s['p95_ttft_ms']:.2f} | {s['cache_induced_errors']} |"
        )
    for field, key in (("variant", "by_variant"), ("family", "by_family")):
        for value in sorted({r[field] for r in groups["query"]}):
            subset = {
                m: [r for r in rows if r[field] == value] for m, rows in groups.items()
            }
            result[key][value] = {m: summary(rows) for m, rows in subset.items()}
    for variant in ("all", "original", "paraphrase", "labels"):
        subset = {
            m: [r for r in rows if variant == "all" or r["variant"] == variant]
            for m, rows in groups.items()
        }
        comparisons = {}
        for a, b in (("query", "topk"), ("fresh", "topk"), ("query", "fresh")):
            aa, bb = summary(subset[a]), summary(subset[b])
            comparisons[a + "_vs_" + b] = {
                "accuracy_gain": aa["accuracy"] - bb["accuracy"],
                "mean_ttft_change": aa["mean_ttft_ms"] / bb["mean_ttft_ms"] - 1,
                "source_cluster_95ci": compare(subset[a], subset[b]),
            }
        result["comparisons"][variant] = comparisons
    lines += [
        "",
        "## 按输入变体",
        "",
        "| 变体 | full | topk | query | fresh |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for variant, s in result["by_variant"].items():
        lines.append(
            "| "
            + variant
            + " | "
            + " | ".join(
                f"{s[m]['accuracy']:.2%}" for m in ("full", "topk", "query", "fresh")
            )
            + " |"
        )
    lines += [
        "",
        "置信区间按来源聚类并携带全部变体/编辑/重复，未作多重比较校正；零差异不证明等效。",
        "",
        "输入含重新标记的答案表达，原生 full 的变化用于判断标签/指令难度。该实验不"
        "能证明跨模型泛化，也不等于 EPIC 或 CacheBlend 论文完整性能复现。",
        "",
    ]
    (root / "query_boundary_summary.json").write_text(json.dumps(result, indent=2))
    (root / "query_boundary_report.md").write_text("\n".join(lines))
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
