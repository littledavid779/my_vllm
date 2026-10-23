"""Report strict question/template effects without reusing native latency."""

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from analyze_query_controls import compare
from analyze_round2 import percentile
from analyze_strict_comparison import summarize
from apply_query_template_patch import MODES
from run_http_pilot import score
from run_query_template import DATASETS

COMPARISONS = (
    ("both", "question"),  # Incremental template effect with question present.
    ("both", "template"),  # Incremental question effect with template present.
    ("question", "none"),
    ("template", "none"),
    ("both", "topk"),
    ("none", "topk"),
)


def indexed(rows):
    result = {(r["repeat"], r["session_id"], r["version"]): r for r in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate measurement")
    return result


def interaction(groups, draws=1500):
    arms = {m: indexed(groups[m]) for m in ("none", "question", "template", "both")}
    keys = set(arms["both"])
    if any(set(a) != keys for a in arms.values()):
        raise ValueError("unpaired factorial cells")
    by_source = defaultdict(list)
    for key in sorted(keys):
        values = {m: int(a[key]["task_correct"]) for m, a in arms.items()}
        effect = (
            values["both"] - values["question"] - values["template"] + values["none"]
        )
        by_source[arms["both"][key]["source_id"]].append(effect)
    clusters = list(by_source.values())
    rng = random.Random(231026)
    samples = [
        statistics.mean(v for c in rng.choices(clusters, k=len(clusters)) for v in c)
        for _ in range(draws)
    ]
    return {
        "effect": statistics.mean(v for c in clusters for v in c),
        "source_cluster_95ci": [percentile(samples, 0.025), percentile(samples, 0.975)],
    }


def analysis_dataset(root, dataset, frozen, coverage):
    groups = {mode: [] for mode in MODES}
    all_audits = {mode: [] for mode in MODES}
    refs = None
    expected = {
        (r["session_id"], r["version"]) for r in coverage["prompts"] if r["version"] > 0
    }
    repetitions = []
    for repeat in (0, 1):
        for mode in MODES:
            path = root / f"{dataset}_rep{repeat}_{mode}"
            manifest = json.loads((path / "manifest.json").read_text())
            config = manifest["config"]
            if (
                manifest["status"] != "completed"
                or config["query_template"] != mode
                or not config["strict_causal"]
                or not config["precheck_kv_write"]
                or config["ratios"] != {"cacheblend10": 0.1}
                or manifest["input_sha256"] != frozen["input_sha256"][dataset]
                or not manifest.get("controls_from")
            ):
                raise ValueError("stage differs from frozen protocol")
            data = json.loads((path / "input.json").read_text())
            gold = {
                (s["session_id"], v): item["expected_value"]
                for s in data["sessions"]
                for v, item in enumerate(s["versions"])
                if v > 0
            }
            all_rows = json.loads((path / "results.json").read_text())
            references = {
                (r["session_id"], r["version"]): r
                for r in all_rows
                if r["arm"] == "full"
            }
            rows = [r for r in all_rows if r["arm"] == "cacheblend10"]
            if (
                len(rows) != len(expected)
                or {(r["session_id"], r["version"]) for r in rows} != expected
                or set(references) != expected
            ):
                raise ValueError("incomplete measurement or reference")
            if refs is not None and refs != references:
                raise ValueError("native references differ between stages")
            refs = references
            for row in [*rows, *refs.values()]:
                target = gold[row["session_id"], row["version"]]
                if row["expected_value"] != target or row["task_correct"] != score(
                    row["generated_text"], target
                ):
                    raise ValueError("label or scoring mismatch")
            for row in rows:
                row["repeat"] = repeat
            audit = json.loads((path / "selection_audit.json").read_text())
            if audit["measured_calls"] != len(expected) or audit["mode"] != mode:
                raise ValueError("selection audit is incomplete")
            groups[mode].extend(rows)
            all_audits[mode].extend(audit["entries"])
            repetitions.append(
                {"repeat": repeat, "mode": mode, **summarize(rows, refs)}
            )
    if refs is None:
        raise ValueError("missing references")
    summaries = {m: summarize(rows, refs) for m, rows in groups.items()}
    layouts = sorted({r["placement"] for r in groups["both"]})
    comparisons, by_layout = {}, {}
    for layout in ("all", *layouts):
        selected = {
            m: [r for r in rows if layout == "all" or r["placement"] == layout]
            for m, rows in groups.items()
        }
        by_layout[layout] = {m: summarize(rows, refs) for m, rows in selected.items()}
        comparisons[layout] = {}
        for a, b in COMPARISONS:
            aa, bb = by_layout[layout][a], by_layout[layout][b]
            comparisons[layout][f"{a}_vs_{b}"] = {
                "quality_gain": aa["accuracy"] - bb["accuracy"],
                "mean_ttft_change": aa["mean_ttft_ms"] / bb["mean_ttft_ms"] - 1,
                "p95_ttft_change": aa["p95_ttft_ms"] / bb["p95_ttft_ms"] - 1,
                "source_cluster_95ci": compare(selected[a], selected[b]),
            }
        comparisons[layout]["factorial_interaction"] = interaction(selected)
    indices = {m: indexed(rows) for m, rows in groups.items()}
    flips = {
        m: sum(
            a[0, sid, v]["task_correct"] != a[1, sid, v]["task_correct"]
            for sid, v in expected
        )
        for m, a in indices.items()
    }
    coverage_stats = {
        m: {
            "measured_observations": len(entries),
            "mean_question_selected": statistics.mean(
                e["question_bits"].bit_count() for e in entries
            ),
            "mean_template_selected": statistics.mean(
                e["template_bits"].bit_count() for e in entries
            ),
            "fully_selected_question_fraction": statistics.mean(
                e["question_bits"] == (1 << e["question_tokens"]) - 1 for e in entries
            ),
            "fully_selected_template_fraction": statistics.mean(
                e["template_bits"] == (1 << e["template_tokens"]) - 1 for e in entries
            ),
        }
        for m, entries in all_audits.items()
    }
    families = {
        f: {
            m: summarize([r for r in rows if r["family"] == f], refs)
            for m, rows in groups.items()
        }
        for f in sorted({r["family"] for r in refs.values()})
    }
    versions = {
        v: {
            m: summarize([r for r in rows if r["version"] == v], refs)
            for m, rows in groups.items()
        }
        for v in (1, 2, 3)
    }
    cases = [
        {
            "session_id": sid,
            "version": v,
            "family": refs[sid, v]["family"],
            "placement": refs[sid, v]["placement"],
            "gold": refs[sid, v]["expected_value"],
            "full_answer": refs[sid, v]["generated_text"],
            "full_correct": refs[sid, v]["task_correct"],
            "methods": {
                m: [
                    {
                        "repeat": rep,
                        "answer": a[rep, sid, v]["generated_text"],
                        "correct": a[rep, sid, v]["task_correct"],
                    }
                    for rep in (0, 1)
                ]
                for m, a in indices.items()
            },
        }
        for sid, v in sorted(expected)
    ]
    return {
        "summary": summaries,
        "comparisons": comparisons,
        "layouts": by_layout,
        "repetitions": repetitions,
        "quality_repeat_flips": flips,
        "families": families,
        "versions": versions,
        "selection_coverage": coverage_stats,
        "native_quality_reference": {
            "events": len(refs),
            "correct": sum(r["task_correct"] for r in refs.values()),
            "fresh_latency": False,
        },
    }, cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    root = args.run
    progress = json.loads((root / "progress.json").read_text())
    if progress["status"] != "completed" or progress["restore_errors"]:
        raise ValueError("measurements and restoration must finish")
    frozen = json.loads((root / "frozen_config.json").read_text())
    coverage = json.loads((root / "prompt_coverage.json").read_text())
    results, cases = {}, {}
    lines = [
        "# 问题与模板：严格拆分验证",
        "",
        "每数据集 288 个编辑事件，各组两次新服务/新缓存重复，原始 topk 另作参照。",
        "none/question/template/both 是严格排除/纳入的四组；"
        "不是仅强制后允许剩余 top-k 再选另一部分。",
        "模板为固定 9 token，包括用户结束、assistant 起始和空 thinking 模板。"
        "关闭只作用于检查层后的 KV 修复选择；早层探测与正常首 token 生成仍然执行。",
        "总预算 10%，TTFT 包含 B′ 准备与主请求首 token、检索搬运、修复选择及日志；"
        "native 仅沿用质量参考，不比较旧延迟。",
        "",
    ]
    for dataset in DATASETS:
        result, case_rows = analysis_dataset(root, dataset, frozen, coverage[dataset])
        results[dataset], cases[dataset] = result, case_rows
        ref = result["native_quality_reference"]
        lines += [
            f"## {dataset}",
            "",
            f"完整重算质量参考：{ref['correct']}/{ref['events']}；"
            "两次重复不增加独立来源数。",
            "",
            "| 修复选择 | 正确率 | 平均 TTFT ms | P95 ms "
            "| full 对、本组错（两轮观测） |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for m, s in result["summary"].items():
            lines.append(
                f"| {m} | {s['accuracy']:.2%} | {s['mean_ttft_ms']:.2f} "
                f"| {s['p95_ttft_ms']:.2f} | {s['cache_induced_errors']} |"
            )
        lines.append("")
        for name, c in result["comparisons"]["all"].items():
            if name == "factorial_interaction":
                ci = c["source_cluster_95ci"]
                lines.append(
                    "- 交互作用 both-question-template+none："
                    f"{c['effect'] * 100:+.2f} pp，"
                    f"95% CI [{ci[0] * 100:+.2f}, {ci[1] * 100:+.2f}]。"
                )
            else:
                ci = c["source_cluster_95ci"]["quality_gain"]
                lines.append(
                    f"- {name}：正确率差 {c['quality_gain'] * 100:+.2f} pp，"
                    f"来源聚类 95% CI [{ci[0] * 100:+.2f}, {ci[1] * 100:+.2f}]；"
                    f"均值 TTFT {c['mean_ttft_change']:+.2%}，"
                    f"P95 {c['p95_ttft_change']:+.2%}。"
                )
        lines += [
            "",
            f"重复间正确性翻转：{result['quality_repeat_flips']}。",
            "",
            "| 模式 | 平均选中问题 token | 平均选中模板 token "
            "| 完整问题覆盖率 | 完整模板覆盖率 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for m, s in result["selection_coverage"].items():
            lines.append(
                f"| {m} | {s['mean_question_selected']:.2f} "
                f"| {s['mean_template_selected']:.2f} "
                f"| {s['fully_selected_question_fraction']:.2%} "
                f"| {s['fully_selected_template_fraction']:.2%} |"
            )
        lines.append("")
    lines += [
        "两份输入都已经用于本地实验。旧集是人工 early8k 压力负载；"
        "后一份含原始 natural4k 布局及 early8k，均非官方 BABILong 分数。",
        "主要检验 both-question 与 both-template；其他对比及交互作用作机制解释。"
        "区间未做多重比较校正，零差异不能证明等效。",
        "所有效应是在固定总预算下重新分配位置的效应，"
        "不是脱离其他修复位置的独立 token 因果贡献。",
        "即使模板有效，也只支持当前 Qwen3/no-thinking 模板；"
        "不能据此证明新颖性或跨模型泛化。",
        "",
    ]
    (root / "query_template_summary.json").write_text(json.dumps(results, indent=2))
    (root / "query_template_cases.json").write_text(json.dumps(cases, indent=2))
    (root / "query_template_report.md").write_text("\n".join(lines))
    print(
        json.dumps(
            {
                d: {"summary": r["summary"], "comparisons": r["comparisons"]["all"]}
                for d, r in results.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
