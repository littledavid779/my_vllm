"""Frozen 10% budget: top-k versus reserve-last-64, with fresh repeated runs."""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from run_smoke import gpu_state

PROJECT = Path(__file__).resolve().parents[1]


def audit(path, tail):
    rows = json.loads((path / "results.json").read_text())
    measured = [r for r in rows if r["arm"] == "cacheblend10"]
    text = (path / "cacheblend10.log").read_text()
    entries = [
        dict(
            zip(
                ("total", "selected", "forced", "covered"), map(int, match), strict=True
            )
        )
        for match in re.findall(
            r"QUERY_TAIL total=(\d+) selected=(\d+) forced=(\d+) covered=(\d+)", text
        )
    ]
    long_entries = [e for e in entries if e["total"] >= 1024]
    if len(long_entries) < len(measured):
        raise ValueError("missing selector evidence for long-prompt requests")
    for entry in entries:
        budget = max(1, int(entry["total"] * 0.1))
        expected = min(tail, budget) if entry["total"] >= 1024 else 0
        if entry["selected"] != budget or entry["forced"] != expected:
            raise ValueError(f"budget mismatch: {entry}")
        if expected and entry["covered"] < expected:
            raise ValueError(f"mandatory tail not covered: {entry}")
    (path / "selection_audit.json").write_text(
        json.dumps(
            {"tail_tokens": tail, "measured_events": len(measured), "entries": entries},
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    progress = {
        "status": "preflight",
        "started": time.time(),
        "stages": [],
        "policy": {"ratio": 0.1, "tail_tokens": 64, "minimum_length": 1024},
    }

    def save():
        (root / "progress.json").write_text(json.dumps(progress, indent=2))

    def script(name, *flags, label=None):
        command = [sys.executable, str(PROJECT / "scripts" / name), *map(str, flags)]
        with (root / ((label or name) + ".log")).open("w") as stream:
            subprocess.run(
                command,
                cwd=PROJECT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )

    def run(name, data, controls, tail):
        entry = {"name": name, "tail_tokens": tail, "started": time.time()}
        progress.update(status="running", current_stage=name)
        progress["stages"].append(entry)
        save()
        print(f"starting {name}", flush=True)
        destination = root / name
        script(
            "run_http_pilot.py",
            "--input",
            data,
            "--lock",
            data.with_suffix(".lock.json"),
            "--output",
            destination,
            "--gpu",
            args.gpu,
            "--port",
            18265,
            "--max-model-len",
            12288,
            "--max-num-seqs",
            16,
            "--attention-backend",
            "sparse",
            "--strict-causal",
            "--precheck-kv-write",
            "--query-tail-tokens",
            tail,
            "--arms",
            "full",
            "prefix",
            "cacheblend10",
            "--controls-from",
            controls,
            label=name + ".driver",
        )
        audit(destination, tail)
        entry.update(status="completed", finished=time.time())
        save()

    save()
    applied = []
    try:
        state = gpu_state(args.gpu)
        if state["pids"] or state["memory_mib"] > 100:
            raise RuntimeError("test GPU busy")
        script("verify_runtime.py", label="runtime_before")
        script("parent_guard.py", "check", label="parent_before")
        for name in (
            "apply_strict_causal_patch.py",
            "apply_precheck_kv_patch.py",
            "apply_query_tail_patch.py",
        ):
            script(name)
            applied.append(name)
        pilot = PROJECT / "data/round2_sparse_probe.json"
        pilot_controls = PROJECT / "results/strict_causal_pilot_20261022"
        run("pilot_baseline", pilot, pilot_controls, 0)
        run("pilot_query", pilot, pilot_controls, 64)
        # No answer-dependent tuning or early stopping after this pilot.
        (root / "frozen_config.json").write_text(
            json.dumps(progress["policy"], indent=2)
        )
        data = PROJECT / "data/strict_holdout_v1.json"
        controls = PROJECT / "results/strict_comparison_20261022/rep0_strict"
        for repeat, order in ((0, ("baseline", "query")), (1, ("query", "baseline"))):
            for name in order:
                run(f"rep{repeat}_{name}", data, controls, 64 if name == "query" else 0)
        progress["status"] = "completed"
    except BaseException as exc:
        progress.update(status="failed", error=repr(exc))
        raise
    finally:
        errors = []
        for name in reversed(applied):
            try:
                script(name, "--restore", label="restore_" + name)
            except Exception as exc:
                errors.append({"script": name, "error": repr(exc)})
        progress["restore_errors"] = errors
        progress["finished"] = time.time()
        progress["gpu_after"] = gpu_state(args.gpu)
        if errors:
            progress["status"] = "restore_failed"
        save()
        if not errors:
            script("verify_runtime.py", label="runtime_after")
            script("parent_guard.py", "check", label="parent_after")
    print(json.dumps(progress, indent=2))


if __name__ == "__main__":
    main()
