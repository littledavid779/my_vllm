"""Analyze the predeclared query-refresh controls without pooling reused timings."""

import argparse
import json
from pathlib import Path

from analyze_query_tail import load
from analyze_strict_comparison import cluster_intervals, summarize
from apply_query_control_patch import MODES


def compare(candidate, reference):
    def index(rows):
        keyed = {(r["repeat"], r["session_id"], r["version"]): r for r in rows}
        if len(keyed) != len(rows):
            raise ValueError("duplicate measurement")
        return keyed

    a, b = index(candidate), index(reference)
    if set(a) != set(b):
        raise ValueError("unpaired measurements")
    pairs = [(a[k], b[k]) for k in a]
    # All intervals resample sources, carrying edits and repeats together.
    return cluster_intervals(pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    root = args.run
    progress = json.loads((root / "progress.json").read_text())
    if progress["status"] != "completed" or progress["restore_errors"]:
        raise ValueError("measurements and restoration must finish first")
    groups = {mode: [] for mode in MODES}
    per_repeat, keyed, refs = [], {}, {}
    expected_hash = json.loads((root / "frozen_config.json").read_text())[
        "input_sha256"
    ]["strict_holdout_v1"]
    for repeat in (0, 1):
        keyed[repeat] = {}
        for mode in MODES:
            path = root / f"rep{repeat}_{mode}"
            rows, references, manifest = load(path)
            config = manifest["config"]
            if (
                len(rows) != 288
                or len(references) != 288
                or manifest["input_sha256"] != expected_hash
                or config["query_control"] != mode
                or not config["strict_causal"]
                or not config["precheck_kv_write"]
                or config["ratios"] != {"cacheblend10": 0.1}
            ):
                raise ValueError("stage differs from frozen experiment")
            if refs and refs != references:
                raise ValueError("native references differ")
            refs = references
            index = {(r["session_id"], r["version"]): r for r in rows}
            if len(index) != len(rows) or set(index) != set(refs):
                raise ValueError("incomplete or duplicate events")
            keyed[repeat][mode] = index
            for row in rows:
                row["repeat"] = repeat
            groups[mode].extend(rows)
            per_repeat.append({"repeat": repeat, "mode": mode, **summarize(rows, refs)})
    summary = {mode: summarize(rows, refs) for mode, rows in groups.items()}
    comparisons = {}
    for reference in ("query", "topk", "pre_tail64", "random64"):
        a, b = summary["tail64"], summary[reference]
        comparisons["tail64_vs_" + reference] = {
            "quality_gain": a["accuracy"] - b["accuracy"],
            "mean_ttft_change": a["mean_ttft_ms"] / b["mean_ttft_ms"] - 1,
            "p95_ttft_change": a["p95_ttft_ms"] / b["p95_ttft_ms"] - 1,
            "source_cluster_95ci": compare(groups["tail64"], groups[reference]),
        }
    families = {
        family: {
            mode: summarize([r for r in rows if r["family"] == family], refs)
            for mode, rows in groups.items()
        }
        for family in sorted({r["family"] for r in refs.values()})
    }
    versions = {
        version: {
            mode: summarize([r for r in rows if r["version"] == version], refs)
            for mode, rows in groups.items()
        }
        for version in (1, 2, 3)
    }
    pilot = {}
    for mode in MODES:
        rows, references, manifest = load(root / ("pilot_" + mode))
        if manifest["config"]["query_control"] != mode or len(rows) != 72:
            raise ValueError("pilot incomplete")
        pilot[mode] = {
            layout: summarize([r for r in rows if r["placement"] == layout], references)
            for layout in ("natural4k", "early8k")
        }
    cases = [
        {
            "session_id": key[0],
            "version": key[1],
            "family": full["family"],
            "gold": full["expected_value"],
            "full_answer": full["generated_text"],
            "full_correct": full["task_correct"],
            "methods": {
                mode: {
                    "answer": keyed[0][mode][key]["generated_text"],
                    "correct": keyed[0][mode][key]["task_correct"],
                }
                for mode in MODES
            },
        }
        for key, full in refs.items()
    ]
    agreement = {
        "same_correctness": sum(
            keyed[0]["tail64"][k]["task_correct"]
            == keyed[0]["query"][k]["task_correct"]
            for k in refs
        ),
        "tail_only_correct": sum(
            keyed[0]["tail64"][k]["task_correct"]
            and not keyed[0]["query"][k]["task_correct"]
            for k in refs
        ),
        "query_only_correct": sum(
            keyed[0]["query"][k]["task_correct"]
            and not keyed[0]["tail64"][k]["task_correct"]
            for k in refs
        ),
        "same_text": sum(
            keyed[0]["tail64"][k]["generated_text"]
            == keyed[0]["query"][k]["generated_text"]
            for k in refs
        ),
    }
    result = {
        "scope": "Seen BABILong edit stress test, single-flight HTTP, one model",
        "native_quality_reference": {
            "correct": sum(r["task_correct"] for r in refs.values()),
            "events": len(refs),
            "fresh_latency": False,
        },
        "summary": summary,
        "comparisons": comparisons,
        "repetitions": per_repeat,
        "families": families,
        "versions": versions,
        "pilot": pilot,
        "tail_query_agreement_rep0": agreement,
        "quality_repeat_flips": {
            mode: sum(
                keyed[0][mode][k]["task_correct"] != keyed[1][mode][k]["task_correct"]
                for k in refs
            )
            for mode in MODES
        },
    }
    (root / "query_controls_summary.json").write_text(json.dumps(result, indent=2))
    (root / "query_controls_cases.json").write_text(json.dumps(cases, indent=2))
    lines = [
        "# 问题刷新机制：固定预算对照",
        "",
        "96 个此前已评估来源，288 个编辑事件；每组独立重启两轮。各组预算均为 10%，"
        "主对照是固定尾部 64 与完整问题/模板重算；其余位置按 K 差异 top-k 补足。",
        "客户端 TTFT 包含 B′ 准备、位置选择、日志同步与首 token。"
        "复用的 full/prefix 只作质量参考，不作为新测延迟。",
        "",
        "| 方法 | 正确率 | 均值 ms | P95 ms | full 正确而本方法错误（两轮观测） |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for mode, item in summary.items():
        lines.append(
            f"| {mode} | {item['accuracy']:.2%} | {item['mean_ttft_ms']:.2f} "
            f"| {item['p95_ttft_ms']:.2f} | {item['cache_induced_errors']} |"
        )
    lines += ["", "主比较与辅助比较（95% 区间按来源聚类，非独立重复事件）：", ""]
    for name, item in comparisons.items():
        ci = item["source_cluster_95ci"]
        lines.append(
            f"- {name}：准确率差 {100 * item['quality_gain']:+.2f} 个百分点，"
            f"95% CI [{100 * ci['quality_gain'][0]:+.2f}, "
            f"{100 * ci['quality_gain'][1]:+.2f}]；"
            f"均值 TTFT {item['mean_ttft_change']:+.2%}，"
            f"P95 {item['p95_ttft_change']:+.2%}。"
        )
    lines += [
        "",
        f"第一轮 tail64 与 query 逐事件对照：{agreement}。",
        "",
        "解释规则：若完整问题重算取得相近增益，现有证据主要支持问题 KV 刷新的作用；"
        "不能据此把固定 64 当成独立的新机制。未显著优于也不等于已证明等效。"
        "若固定尾部优于该控制，需要再排查额外边界 token、形状和预算分配。"
        "辅助比较只作探索，不能从四个区间中挑一个宣称确认性发现。",
        "",
        "输入为 early8k 人工长后缀压力负载，开发另含 natural4k；"
        "未覆盖新模型、自然长文、并发吞吐。"
        "query 组仍会读到近似上下文 KV，不等于完整请求重算，也不是 ProphetKV 复现。"
        "三次编辑沿用上游缓存保存行为，不证明每轮近似缓存回写后的误差累积。",
        "",
    ]
    (root / "query_controls_report.md").write_text("\n".join(lines))
    print(
        json.dumps(
            {"summary": summary, "comparisons": comparisons, "agreement": agreement},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
