"""Keep raw/default/corrected release configurations separate on one fixed subset."""

import argparse
import json
import statistics
from pathlib import Path

from analyze_round2 import percentile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-run", type=Path, required=True)
    parser.add_argument("--raw-probe", type=Path, required=True)
    parser.add_argument("--fixed-probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roots = {
        "FA release": args.main_run,
        "FI release": args.raw_probe,
        "FI block-mask probe": args.fixed_probe,
    }
    subset = json.loads((args.raw_probe / "input.json").read_text())
    ids = {s["session_id"] for s in subset["sessions"]}
    summary, per_run = [], {}
    for label, root in roots.items():
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest["status"] != "completed":
            raise ValueError(f"incomplete probe: {root}")
        rows = [
            r
            for r in json.loads((root / "results.json").read_text())
            if r["session_id"] in ids
        ]
        refs = {(r["session_id"], r["version"]): r for r in rows if r["arm"] == "full"}
        per_run[label] = {(r["arm"], r["session_id"], r["version"]): r for r in rows}
        for placement in ("natural4k", "early8k"):
            for arm in ("full", "prefix", "cacheblend15"):
                group = [
                    r for r in rows if r["placement"] == placement and r["arm"] == arm
                ]
                assert len(group) == 36
                errors = [
                    r
                    for r in group
                    if not r["task_correct"]
                    and refs[r["session_id"], r["version"]]["task_correct"]
                ]
                times = [r["client_ttft_ms"] for r in group]
                summary.append(
                    {
                        "configuration": label,
                        "placement": placement,
                        "arm": arm,
                        "events": len(group),
                        "correct": sum(r["task_correct"] for r in group),
                        "reference_correct": sum(
                            refs[r["session_id"], r["version"]]["task_correct"]
                            for r in group
                        ),
                        "cache_induced_errors": len(errors),
                        "error_sources": len({r["source_id"] for r in errors}),
                        "ttft_mean_ms": statistics.mean(times),
                        "ttft_p95_ms": percentile(times, 0.95),
                    }
                )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "probe_comparison.json").write_text(json.dumps(summary, indent=2))
    changes = []
    for key, raw in per_run["FI release"].items():
        if key[0] != "cacheblend15":
            continue
        fixed = per_run["FI block-mask probe"][key]
        if raw["task_correct"] != fixed["task_correct"]:
            changes.append(
                {
                    "session_id": key[1],
                    "version": key[2],
                    "placement": raw["placement"],
                    "expected_value": raw["expected_value"],
                    "raw_answer": raw["generated_text"],
                    "fixed_answer": fixed["generated_text"],
                    "fixed_correct": fixed["task_correct"],
                }
            )
    (args.output / "probe_changed_answers.json").write_text(
        json.dumps(changes, indent=2)
    )
    controls = {}
    for arm in ("full", "prefix"):
        controls[arm] = sum(
            raw["task_correct"] != per_run["FI block-mask probe"][key]["task_correct"]
            for key, raw in per_run["FI release"].items()
            if key[0] == arm
        )
    (args.output / "control_differences.json").write_text(
        json.dumps(controls, indent=2)
    )
    lines = [
        "# 后端诊断：相同固定子集",
        "",
        "每类固定前四个来源，共 12 来源，两个布局、三次编辑。"
        "每行 36 个事件，不与 108 来源主实验的总体数值混用。",
        "FA release 是默认 FlashAttention；FI release 是官方 sparse 开关"
        "但原代码未激活的发布实现；FI block-mask probe 激活已构造的 sparse mask，"
        "阻止 token 索引原地改写，并使用 FlashInfer 公共 wrapper 适配 GQA 布局。"
        "后者是局部正确性诊断，不是新算法，也不代表严格因果已验证。",
        "",
        "| 布局 | 配置 | 方法 | 正确/事件 | 缓存新增错误 | 平均TTFT ms | P95 ms |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for g in summary:
        lines.append(
            f"| {g['placement']} | {g['configuration']} | {g['arm']} "
            f"| {g['correct']}/{g['events']} | {g['cache_induced_errors']} "
            f"| {g['ttft_mean_ms']:.2f} | {g['ttft_p95_ms']:.2f} |"
        )
    lines += [
        "",
        f"FI 原始/修正版控制组质量差异：{controls}。",
        "FA 与 FI 的内核及批容量不同，不能将所有差异归因于 mask。"
        "FI 原始/修正版使用相同后端、批容量 16、相同输入和生成设置。",
        "每请求仅一次计时，且未隔离整机资源；不能据此证明 TTFT 基本不变。"
        "修正版若仍有错误，还需要区分 32-token 块 mask 近似、缓存分隔符、"
        "新块预计算及 token 选择策略；本轮没有完成这些归因。",
        "",
    ]
    (args.output / "probe_report.md").write_text("\n".join(lines))
    print(
        json.dumps(
            {
                "rows": len(summary),
                "changed_answers": len(changes),
                "control_differences": controls,
            }
        )
    )


if __name__ == "__main__":
    main()
