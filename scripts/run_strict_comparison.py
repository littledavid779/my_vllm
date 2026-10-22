"""Time-only pilot calibration, then two fresh-cache long-suffix comparisons."""

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

from run_smoke import gpu_state

PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    state = gpu_state(args.gpu)
    if state["pids"] or state["memory_mib"] > 100:
        raise RuntimeError("GPU busy")
    progress = {"status": "running", "started": time.time(), "stages": []}
    current_mode = "strict"  # Prior validated pilot ends with strict patches installed.

    def save():
        (root / "progress.json").write_text(json.dumps(progress, indent=2))

    def script(name, *flags, log=None):
        command = [sys.executable, str(PROJECT / "scripts" / name), *map(str, flags)]
        if log:
            with log.open("w") as stream:
                subprocess.run(
                    command,
                    cwd=PROJECT,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        else:
            subprocess.run(command, cwd=PROJECT, check=True, stdout=subprocess.DEVNULL)

    def mode(target):
        nonlocal current_mode
        if current_mode != target:
            if current_mode == "strict":
                script("apply_strict_causal_patch.py", "--restore")
            else:
                script("apply_sparse_index_patch.py", "--restore")
            current_mode = None
            name = (
                "apply_strict_causal_patch.py"
                if target == "strict"
                else "apply_sparse_index_patch.py"
            )
            script(name)
            current_mode = target
        script("apply_precheck_kv_patch.py")

    def run(name, variant, data, arms, controls=None):
        mode(variant)
        destination = root / name
        entry = {
            "name": name,
            "variant": variant,
            "started": time.time(),
            "input": str(data),
            "arms": arms,
        }
        progress["stages"].append(entry)
        progress["current_stage"] = name
        save()
        print(f"starting {name}", flush=True)
        flags = [
            "--input",
            data,
            "--lock",
            data.with_suffix(".lock.json"),
            "--gpu",
            args.gpu,
            "--max-model-len",
            12288,
            "--max-num-seqs",
            16,
            "--port",
            18265,
            "--attention-backend",
            "sparse",
            "--precheck-kv-write",
            "--strict-causal" if variant == "strict" else "--sparse-correctness-fix",
            "--arms",
            *arms,
            "--output",
            destination,
        ]
        if controls:
            flags += ["--controls-from", controls]
        script("run_http_pilot.py", *flags, log=root / f"{name}.driver.log")
        entry["finished"] = time.time()
        entry["status"] = "completed"
        save()
        return destination

    def early_mean(path, arm):
        # Deliberately never inspect gold labels or generated answers for calibration.
        rows = json.loads((path / "results.json").read_text())
        return statistics.mean(
            r["client_ttft_ms"]
            for r in rows
            if r["arm"] == arm and r["placement"] == "early8k"
        )

    try:
        pilot = PROJECT / "data/round2_sparse_probe.json"
        controls = PROJECT / "results/strict_causal_pilot_20261022"
        low = run(
            "calibration_strict",
            "strict",
            pilot,
            ["full", "prefix", "cacheblend5", "cacheblend10"],
            controls,
        )
        block = run(
            "calibration_block",
            "block",
            pilot,
            ["full", "prefix", "cacheblend15"],
            controls,
        )
        candidates = {
            5: early_mean(low, "cacheblend5"),
            10: early_mean(low, "cacheblend10"),
            15: early_mean(
                PROJECT / "results/strict_causal_kvwrite_pilot_20261022", "cacheblend15"
            ),
        }
        target = early_mean(block, "cacheblend15")
        closest = min(candidates, key=lambda ratio: abs(candidates[ratio] - target))
        # One timing-only interpolated probe if the initial grid misses the target.
        if abs(candidates[closest] / target - 1) > 0.05:
            slope = (candidates[10] - candidates[5]) / 5
            if slope > 0:
                estimate = max(1, min(15, round(5 + (target - candidates[5]) / slope)))
                if estimate not in candidates:
                    extra = run(
                        "calibration_interpolated",
                        "strict",
                        pilot,
                        ["full", "prefix", f"cacheblend{estimate}"],
                        controls,
                    )
                    candidates[estimate] = early_mean(extra, f"cacheblend{estimate}")
                    closest = min(
                        candidates, key=lambda ratio: abs(candidates[ratio] - target)
                    )
        calibration = {
            "criterion": "closest long-suffix mean TTFT; no quality used",
            "strict_candidates_ms": candidates,
            "block15_ms": target,
            "selected_strict_ratio_percent": closest,
            "mean_change": candidates[closest] / target - 1,
        }
        (root / "calibration.json").write_text(json.dumps(calibration, indent=2))
        progress["calibration"] = calibration
        save()
        print(json.dumps(calibration), flush=True)
        heldout = PROJECT / "data/strict_holdout_v1.json"
        arms = ["full", "prefix", f"cacheblend{closest}"]
        strict0 = run("rep0_strict", "strict", heldout, arms)
        run("rep0_block", "block", heldout, ["full", "prefix", "cacheblend15"], strict0)
        # Reverse order; every stage starts a new server/cache. Native controls reused.
        run("rep1_block", "block", heldout, ["full", "prefix", "cacheblend15"], strict0)
        run("rep1_strict", "strict", heldout, arms, strict0)
        progress["status"] = "completed"
    except BaseException as exc:
        progress.update(status="failed", error=repr(exc))
        raise
    finally:
        if current_mode:
            name = (
                "apply_strict_causal_patch.py"
                if current_mode == "strict"
                else "apply_sparse_index_patch.py"
            )
            script(name, "--restore")
        script("apply_precheck_kv_patch.py", "--restore")
        progress["finished"] = time.time()
        progress["gpu_after"] = gpu_state(args.gpu)
        save()


if __name__ == "__main__":
    main()
