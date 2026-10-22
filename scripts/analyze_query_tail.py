"""Task accuracy and current paired timings; reused native timing is excluded."""

import argparse
import json
from pathlib import Path

from analyze_strict_comparison import cluster_intervals, summarize


def load(path):
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("incomplete stage")
    rows = json.loads((path / "results.json").read_text())
    refs = {(r["session_id"], r["version"]): r for r in rows if r["arm"] == "full"}
    rows = [r for r in rows if r["arm"] == "cacheblend10"]
    return rows, refs, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    root = args.run
    if json.loads((root / "progress.json").read_text())["status"] != "completed":
        raise ValueError("run must complete before final analysis")
    groups = {"baseline": [], "query": []}
    refs, pairs, repeats = {}, [], []
    by_repeat = {}
    for repeat in (0, 1):
        keyed = {}
        for kind, tail in (("baseline", 0), ("query", 64)):
            rows, references, manifest = load(root / f"rep{repeat}_{kind}")
            assert len(rows) == len(references) == 288
            assert manifest["config"]["query_tail_tokens"] == tail
            assert (
                manifest["config"]["strict_causal"]
                and manifest["config"]["precheck_kv_write"]
            )
            if refs:
                assert refs == references
            refs = references
            keyed[kind] = {(r["session_id"], r["version"]): r for r in rows}
            assert set(keyed[kind]) == set(refs)
            groups[kind] += rows
            repeats.append({"repeat": repeat, "method": kind, **summarize(rows, refs)})
        pairs += [(keyed["query"][k], keyed["baseline"][k]) for k in refs]
        by_repeat[repeat] = keyed
    summary = {kind: summarize(rows, refs) for kind, rows in groups.items()}
    baseline, query = summary["baseline"], summary["query"]
    comparison = {
        "quality_gain": query["accuracy"] - baseline["accuracy"],
        "mean_ttft_change": query["mean_ttft_ms"] / baseline["mean_ttft_ms"] - 1,
        "p95_ttft_change": query["p95_ttft_ms"] / baseline["p95_ttft_ms"] - 1,
        "cluster_95ci": cluster_intervals(pairs),
        "quality_repeat_flips": {
            kind: sum(
                by_repeat[0][kind][k]["task_correct"]
                != by_repeat[1][kind][k]["task_correct"]
                for k in refs
            )
            for kind in groups
        },
    }
    comparison["meets_point_estimate_gate"] = (
        comparison["quality_gain"] >= 0.05
        and comparison["mean_ttft_change"] <= 0.05
        and comparison["p95_ttft_change"] <= 0.05
    )
    families = {
        family: {
            kind: summarize([r for r in rows if r["family"] == family], refs)
            for kind, rows in groups.items()
        }
        for family in sorted({r["family"] for r in groups["query"]})
    }
    versions = {
        str(version): {
            kind: summarize([r for r in rows if r["version"] == version], refs)
            for kind, rows in groups.items()
        }
        for version in (1, 2, 3)
    }
    pilot = {}
    for kind in groups:
        rows, references, _ = load(root / f"pilot_{kind}")
        pilot[kind] = {
            layout: summarize([r for r in rows if r["placement"] == layout], references)
            for layout in ("early8k", "natural4k")
        }
    cases = []
    for key, full in refs.items():
        a, b = by_repeat[0]["baseline"][key], by_repeat[0]["query"][key]
        if a["task_correct"] != b["task_correct"] or (
            full["task_correct"] and not b["task_correct"]
        ):
            cases.append(
                {
                    "session_id": key[0],
                    "version": key[1],
                    "family": a["family"],
                    "gold": full["expected_value"],
                    "full_answer": full["generated_text"],
                    "baseline_answer": a["generated_text"],
                    "query_answer": b["generated_text"],
                    "baseline_correct": a["task_correct"],
                    "query_correct": b["task_correct"],
                    "full_correct": full["task_correct"],
                }
            )
    native_quality = {
        "correct": sum(r["task_correct"] for r in refs.values()),
        "events": len(refs),
        "note": "Reused native quality reference; no fresh native latency claim.",
    }
    result = {
        "native_quality_reference": native_quality,
        "summary": summary,
        "comparison": comparison,
        "repetitions": repeats,
        "families": families,
        "versions": versions,
        "pilot": pilot,
    }
    (root / "query_tail_summary.json").write_text(json.dumps(result, indent=2))
    (root / "query_tail_cases.json").write_text(json.dumps(cases, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
