"""Queue and run cached versus native-question controls on frozen prompt variants."""

import argparse
import collections
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

from apply_query_boundary_patch import HELPER, MODES
from run_query_template import atomic_json
from run_smoke import gpu_state

PROJECT = Path(__file__).resolve().parents[1]
INPUT = PROJECT / "data/query_boundary_v1.json"
SCOPE = {}
exec(compile(HELPER, "boundary-preflight", "exec"), SCOPE)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_inputs(path):
    data = json.loads(path.read_text())
    if (
        digest(path)
        != json.loads(path.with_suffix(".lock.json").read_text())["input_sha256"]
    ):
        raise ValueError("input lock mismatch")
    prompts = []
    for session in data["sessions"]:
        for version, item in enumerate(session["versions"]):
            ids = item["token_ids"]
            q = SCOPE["query_offset"](ids)
            if (
                q is None
                or ids[-9:] != SCOPE["TEMPLATE"]
                or not 0 < len(ids) - q < min(64, int(len(ids) * 0.1))
            ):
                raise ValueError("invalid complete question and template")
            if len(ids) != session["prompt_tokens"]:
                raise ValueError("unequal edit length")
            prompts.append(
                {
                    "session_id": session["session_id"],
                    "version": version,
                    "total": len(ids),
                    "query_start": q,
                    "query_tokens": len(ids) - q,
                    "prompt_sha": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                }
            )
    return {"input_sha256": digest(path), "prompts": prompts}


def audit(path, mode, coverage, ratio=0.1):
    manifest = json.loads((path / "manifest.json").read_text())
    arm = f"cacheblend{round(ratio * 100)}"
    if (
        manifest["status"] != "completed"
        or manifest["config"]["query_boundary"] != mode
    ):
        raise ValueError("incomplete or wrong stage")
    expected = {p["prompt_sha"]: p for p in coverage["prompts"] if p["version"] > 0}
    text = (path / f"{arm}.log").read_text()
    outputs = {}
    for label in ("LOAD", "SELECT"):
        records = [
            json.loads(s)
            for s in re.findall(r"BOUNDARY_" + label + r" (\{[^\n]+\})", text)
        ]
        observed = collections.Counter()
        picked = []
        for record in records:
            key = record["prompt_sha"]
            if key not in expected:
                continue
            prompt = expected[key]
            n, q, span = prompt["total"], prompt["query_start"], prompt["query_tokens"]
            for field in ("total", "query_start", "query_tokens"):
                if record[field] != prompt[field]:
                    raise ValueError("runtime boundaries differ from frozen input")
            hit = q if mode == "fresh" else n
            native = span if mode == "fresh" else 1
            if (
                record["mode"] != mode
                or record["hit_tokens"] != hit
                or record["actual_native_tokens"] != native
                or record["native_prompt_tokens"] != native
            ):
                raise ValueError(
                    "cache hit or actual native prefill differs from protocol"
                )
            if label == "SELECT":
                selected = int(n * ratio) - (span if mode == "fresh" else 0)
                if (
                    record["selected"] != selected
                    or record["unique"] != selected
                    or record["blend_tokens"] != hit
                    or not 0 <= record["min_index"] <= record["max_index"] < hit
                ):
                    raise ValueError(
                        "repair budget, uniqueness, or original positions differ"
                    )
                if mode == "fresh" and (
                    record["question_selected"] or record["forced"]
                ):
                    raise ValueError("fresh question leaked into cached repair")
                if mode == "query" and (
                    record["question_selected"] != span or record["forced"] != span
                ):
                    raise ValueError("query repair is incomplete")
            observed[key] += 1
            picked.append(
                {
                    **record,
                    "session_id": prompt["session_id"],
                    "version": prompt["version"],
                }
            )
        if set(observed) != set(expected) or any(n != 1 for n in observed.values()):
            raise ValueError("missing or duplicate runtime evidence")
        outputs[label.lower()] = picked
    rows = json.loads((path / "results.json").read_text())
    measured = [r for r in rows if r["arm"] == arm]
    refs = {(r["session_id"], r["version"]): r for r in rows if r["arm"] == "full"}
    if len(measured) != len(expected):
        raise ValueError("incomplete measured rows")
    if ratio == 1 and any(
        r["task_correct"] != refs[r["session_id"], r["version"]]["task_correct"]
        for r in measured
    ):
        raise ValueError(
            "full-budget smoke does not preserve reference task correctness"
        )
    atomic_json(
        path / "selection_audit.json",
        {"mode": mode, "ratio": ratio, "measured_calls": len(expected), **outputs},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[1, 0])
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--max-wait-hours", type=float, default=24)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        print(json.dumps({"prompts": len(inspect_inputs(INPUT)["prompts"])}))
        return
    if Path(sys.prefix).resolve() != PROJECT / ".venv":
        raise RuntimeError("use dedicated environment")
    if not 1 <= args.poll_seconds <= 60 or args.max_wait_hours <= 0:
        parser.error("invalid wait settings")
    lock = (PROJECT / "state/query_controls.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    progress = {
        "status": "preflight",
        "pid": os.getpid(),
        "started": time.time(),
        "gpu": None,
        "stages": [],
        "planned_measurement_stages": 6,
        "planned_smoke_stages": 3,
    }
    applied, child = [], None

    def save():
        progress["updated"] = time.time()
        atomic_json(root / "progress.json", progress)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)

    def script(name, *flags, label=None):
        nonlocal child
        with (root / ((label or name) + ".log")).open("w") as stream:
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-B",
                    str(PROJECT / "scripts" / name),
                    *map(str, flags),
                ],
                cwd=PROJECT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                code = child.wait()
                if code:
                    raise RuntimeError(
                        f"{name} exited {code}; inspect {label or name}.log"
                    )
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGINT)
                    try:
                        child.wait(timeout=45)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGTERM)
                        child.wait(timeout=10)
                child = None

    def idle(candidates):
        counts = dict.fromkeys(candidates, 0)
        started = time.time()
        while True:
            states = {g: gpu_state(g) for g in candidates}
            progress.update(status="waiting_gpu", gpu_states=states)
            save()
            for gpu, state in states.items():
                quiet = (
                    not state["pids"]
                    and state["memory_mib"] <= 100
                    and state["utilization"] <= 5
                )
                counts[gpu] = counts[gpu] + 1 if quiet else 0
                if counts[gpu] >= 3:
                    return gpu
            if time.time() - started >= args.max_wait_hours * 3600:
                raise TimeoutError("GPU wait expired")
            time.sleep(args.poll_seconds)

    def run(name, mode, path, cov, controls=None, ratio=10):
        idle([progress["gpu"]])
        for p, h in frozen["scripts_sha256"].items():
            if digest(PROJECT / "scripts" / p) != h:
                raise ValueError("script changed during experiment")
        if digest(path) != cov["input_sha256"]:
            raise ValueError("input changed")
        entry = {
            "name": name,
            "mode": mode,
            "started": time.time(),
            "ratio": ratio,
            "status": "running",
        }
        progress["stages"].append(entry)
        progress.update(status="running", current_stage=name)
        save()
        script(
            "run_http_pilot.py",
            "--input",
            path,
            "--lock",
            path.with_suffix(".lock.json"),
            "--output",
            root / name,
            "--gpu",
            progress["gpu"],
            "--port",
            18267,
            "--max-model-len",
            12288,
            "--max-num-seqs",
            16,
            "--attention-backend",
            "sparse",
            "--strict-causal",
            "--precheck-kv-write",
            "--query-boundary",
            mode,
            "--arms",
            "full",
            "prefix",
            f"cacheblend{ratio}",
            *(["--controls-from", controls] if controls else []),
            label=name + ".driver",
        )
        audit(root / name, mode, cov, ratio / 100)
        entry.update(status="completed", finished=time.time())
        save()

    save()
    try:
        coverage = inspect_inputs(INPUT)
        atomic_json(root / "prompt_coverage.json", coverage)
        data = json.loads(INPUT.read_text())
        smoke = dict(data)
        smoke["sessions"] = data["sessions"][:3]
        smoke_path = root / "smoke_input.json"
        atomic_json(smoke_path, smoke)
        atomic_json(
            smoke_path.with_suffix(".lock.json"), {"input_sha256": digest(smoke_path)}
        )
        smoke_coverage = inspect_inputs(smoke_path)
        frozen = {
            "input_sha256": coverage["input_sha256"],
            "modes": MODES,
            "orders": [list(MODES), list(reversed(MODES))],
            "ratio": 0.1,
            "check_layer": 1,
            "native_controls": (
                "fresh full and prefix for each repeat; reused only within that repeat"
            ),
            "source_scope": (
                "32 previously evaluated sources; new prompt variants; exploratory"
            ),
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
            ).strip(),
            "scripts_sha256": {
                p.name: digest(p) for p in (PROJECT / "scripts").glob("*.py")
            },
        }
        atomic_json(root / "frozen_config.json", frozen)
        for name, flags in (
            ("verify_sources.py", ()),
            ("verify_runtime.py", ()),
            ("parent_guard.py", ("check",)),
        ):
            script(name, *flags, label="before_" + name)
        progress["gpu"] = idle(args.gpus)
        for name in (
            "apply_strict_causal_patch.py",
            "apply_precheck_kv_patch.py",
            "apply_query_boundary_patch.py",
        ):
            applied.append(name)
            script(name)
        for mode in MODES:
            run(
                "smoke_" + mode,
                mode,
                smoke_path,
                smoke_coverage,
                root / "smoke_topk" if mode != "topk" else None,
                100,
            )
        for rep, order in enumerate(frozen["orders"]):
            for idx, mode in enumerate(order):
                run(
                    f"rep{rep}_{mode}",
                    mode,
                    INPUT,
                    coverage,
                    root / f"rep{rep}_{order[0]}" if idx else None,
                )
        progress["status"] = "measurements_completed"
    except KeyboardInterrupt as exc:
        progress.update(status="cancelled", error=repr(exc))
    except Exception as exc:
        progress.update(status="failed", error=repr(exc))
    finally:
        errors = []
        for name in reversed(applied):
            try:
                script(name, "--restore", label="restore_" + name)
            except Exception as exc:
                errors.append({"script": name, "error": repr(exc)})
        if not errors:
            for name, flags in (
                ("verify_runtime.py", ()),
                ("parent_guard.py", ("check",)),
            ):
                try:
                    script(name, *flags, label="after_" + name)
                except Exception as exc:
                    errors.append({"script": name, "error": repr(exc)})
        progress["restore_errors"] = errors
        if errors:
            progress["status"] = "restore_failed"
        elif progress["status"] == "measurements_completed":
            progress["status"] = "completed"
        progress["finished"] = time.time()
        save()
    if progress["status"] == "completed":
        try:
            script("analyze_query_boundary.py", "--run", root, label="analysis")
            progress["analysis_status"] = "completed"
        except Exception as exc:
            progress.update(analysis_status="failed", analysis_error=repr(exc))
        save()
    print(json.dumps(progress, indent=2), flush=True)
    if (
        progress["status"] != "completed"
        or progress.get("analysis_status") != "completed"
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
