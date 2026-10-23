"""Run a new source split with shared query refresh and a boundary selector control."""

import argparse
import ast
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

from apply_query_fair_patch import ALL_HELPERS, MODES, SEPARATOR
from run_smoke import gpu_state

PROJECT = Path(__file__).resolve().parents[1]


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    os.replace(temporary, path)


def signature(ids):
    return hashlib.sha256(json.dumps(ids[-64:]).encode()).hexdigest()[:16]


def inspect_inputs():
    from transformers import AutoTokenizer

    scope = {}
    exec(compile(ast.parse(ALL_HELPERS), "query-fair-boundaries", "exec"), scope)
    span, boundaries = scope["query_control_span"], scope["fair_boundaries"]
    path = PROJECT / "data/query_fair_v1.json"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != json.loads(path.with_suffix(".lock.json").read_text())["input_sha256"]:
        raise ValueError("input lock mismatch")
    data = json.loads(path.read_text())
    tokenizer = AutoTokenizer.from_pretrained(data["model"], local_files_only=True)
    if tokenizer.encode("Question:", add_special_tokens=False) != [14582, 25]:
        raise ValueError("question marker changed")
    if tokenizer.encode("<|im_start|>assistant", add_special_tokens=False) != [
        151644,
        77091,
    ]:
        raise ValueError("assistant marker changed")
    if data["pilot_provenance"]["separator_ids"] != SEPARATOR:
        raise ValueError("chunk separator changed")
    prior = set(data["protocol"]["previously_evaluated_source_ids"])
    records = []
    sources = set()
    for session in data["sessions"]:
        if session["source_id"] in prior or session["metadata"]["source_index"] < 66:
            raise ValueError("input overlaps earlier source split")
        sources.add(session["source_id"])
        for version, item in enumerate(session["versions"]):
            ids = item["token_ids"]
            length = span(ids[-64:])
            starts = boundaries(ids)
            if len(starts) != 2 or not 0 < length <= min(64, int(len(ids) * 0.1)):
                raise ValueError("invalid prompt structure or query budget")
            text = tokenizer.decode(ids[-length:])
            if session["metadata"]["question"].strip() not in text:
                raise ValueError("query boundary misses question")
            if "<|im_start|>assistant" not in text:
                raise ValueError("query boundary misses assistant template")
            positions = scope["boundary_positions"](
                len(ids), int(len(ids) * 0.1), length, starts
            )
            records.append(
                {
                    "session_id": session["session_id"],
                    "version": version,
                    "length": len(ids),
                    "query_length": length,
                    "signature": signature(ids),
                    "boundaries": starts,
                    "boundary_exact_prefix_padding": sum(
                        p < starts[0] for p in positions
                    ),
                    "prompt_sha": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                    "boundary_selection_sha": hashlib.sha256(
                        json.dumps(positions).encode()
                    ).hexdigest(),
                }
            )
    if len(sources) != 48 or len(data["sessions"]) != 96 or len(records) != 384:
        raise ValueError("unexpected frozen input size")
    return {
        "query_fair_v1": {
            "input_sha256": digest,
            "prompts": records,
            "sources": len(sources),
            "query_length_range": [
                min(r["query_length"] for r in records),
                max(r["query_length"] for r in records),
            ],
        }
    }


def audit(path, mode, coverage):
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["status"] != "completed" or manifest["config"]["query_control"] != mode:
        raise ValueError("incomplete or mismatched stage")
    expression = (
        r"QUERY_CONTROL mode=(\w+) total=(\d+) selected=(\d+) forced=(\d+) "
        r"query=(\d+) qcovered=(\d+) tailcovered=(\d+) signature=([a-f0-9]+)"
    )
    entries = []
    for match in re.findall(expression, (path / "cacheblend10.log").read_text()):
        entry = {"mode": match[0], "signature": match[-1]}
        entry.update(
            zip(
                ("total", "selected", "forced", "query", "qcovered", "tailcovered"),
                map(int, match[1:-1]),
                strict=True,
            )
        )
        if entry["mode"] != mode or entry["selected"] != max(
            1, int(entry["total"] * 0.1)
        ):
            raise ValueError(f"selection budget/mode mismatch: {entry}")
        expected = 0
        if entry["total"] >= 1024:
            expected = {"query": entry["query"], "boundary": entry["selected"]}.get(
                mode, 64
            )
        if entry["forced"] != expected:
            raise ValueError(f"mandatory count mismatch: {entry}")
        if mode in MODES and entry["qcovered"] != entry["query"]:
            raise ValueError(f"query not fully recomputed: {entry}")
        if mode == "tail64" and entry["total"] >= 1024 and entry["tailcovered"] != 64:
            raise ValueError(f"tail coverage mismatch: {entry}")
        entries.append(entry)
    expected = collections.Counter(
        (r["signature"], r["length"], r["query_length"])
        for r in coverage["prompts"]
        if r["version"] > 0
    )
    actual = collections.Counter(
        (e["signature"], e["total"], e["query"]) for e in entries if e["query"] > 0
    )
    if expected - actual:
        raise ValueError("missing selector evidence for measured prompts")
    full_entries = re.findall(
        r"QUERY_FAIR mode=(\w+) promptsha=([a-f0-9]{64}) selectedsha=([a-f0-9]{64})",
        (path / "cacheblend10.log").read_text(),
    )
    expected_full = {
        r["prompt_sha"]: r for r in coverage["prompts"] if r["version"] > 0
    }
    observed = collections.Counter()
    for actual_mode, prompt_sha, selected_sha in full_entries:
        if actual_mode != mode:
            raise ValueError("full selector audit mode mismatch")
        if prompt_sha in expected_full:
            observed[prompt_sha] += 1
            if (
                mode == "boundary"
                and selected_sha != expected_full[prompt_sha]["boundary_selection_sha"]
            ):
                raise ValueError("boundary positions differ from CPU input oracle")
    if set(observed) != set(expected_full) or any(v != 1 for v in observed.values()):
        raise ValueError("missing or duplicated full-input selection evidence")
    atomic_json(
        path / "selection_audit.json",
        {
            "mode": mode,
            "entries": entries,
            "expected_measured_calls": sum(expected.values()),
            "full_input_audit_calls": sum(observed.values()),
            "boundary_positions_verified": mode == "boundary",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--max-wait-hours", type=float, default=24)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        findings = inspect_inputs()
        print(
            json.dumps(
                {
                    k: {
                        "prompts": len(v["prompts"]),
                        "query_length_range": v["query_length_range"],
                    }
                    for k, v in findings.items()
                },
                indent=2,
            )
        )
        return
    if not 1 <= args.poll_seconds <= 60 or args.max_wait_hours <= 0:
        parser.error("poll must be 1..60 seconds and wait limit positive")
    if Path(sys.prefix).resolve() != PROJECT / ".venv":
        raise RuntimeError("use the dedicated environment")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    lock = (PROJECT / "state/query_controls.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.time()
    progress = {
        "status": "preflight",
        "pid": os.getpid(),
        "started": started,
        "stages": [],
        "gpu": None,
        "poll_seconds": args.poll_seconds,
        "max_wait_hours": args.max_wait_hours,
    }
    applied = []
    child = None

    def save():
        progress["updated"] = time.time()
        atomic_json(root / "progress.json", progress)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupted)

    def script(name, *flags, label=None):
        nonlocal child
        command = [sys.executable, str(PROJECT / "scripts" / name), *map(str, flags)]
        with (root / ((label or name) + ".log")).open("w") as stream:
            child = subprocess.Popen(
                command,
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

    def wait_idle(candidates):
        consecutive = {gpu: 0 for gpu in candidates}
        wait_started = time.time()
        while True:
            states = {gpu: gpu_state(gpu) for gpu in candidates}
            progress.update(status="waiting_gpu", gpu_states=states)
            save()
            for gpu, state in states.items():
                idle = (
                    not state["pids"]
                    and state["memory_mib"] <= 100
                    and state["utilization"] <= 5
                )
                consecutive[gpu] = consecutive[gpu] + 1 if idle else 0
                if consecutive[gpu] >= 3:
                    return gpu
            if time.time() - wait_started >= args.max_wait_hours * 3600:
                raise TimeoutError("GPU wait expired; no experiment result is claimed")
            time.sleep(args.poll_seconds)

    def run(name, mode, coverage, controls=None):
        dataset = "query_fair_v1"
        wait_idle([progress["gpu"]])
        entry = {"name": name, "mode": mode, "started": time.time()}
        progress.update(status="running", current_stage=name)
        progress["stages"].append(entry)
        save()
        data = PROJECT / "data" / (dataset + ".json")
        script(
            "run_http_pilot.py",
            "--input",
            data,
            "--lock",
            data.with_suffix(".lock.json"),
            "--output",
            root / name,
            "--gpu",
            progress["gpu"],
            "--port",
            18266,
            "--max-model-len",
            12288,
            "--max-num-seqs",
            16,
            "--attention-backend",
            "sparse",
            "--strict-causal",
            "--precheck-kv-write",
            "--query-control",
            mode,
            "--query-fair",
            "--arms",
            "full",
            "prefix",
            "cacheblend10",
            *(["--controls-from", controls] if controls else []),
            label=name + ".driver",
        )
        audit(root / name, mode, coverage[dataset])
        entry.update(status="completed", finished=time.time())
        save()

    save()
    try:
        coverage = inspect_inputs()
        atomic_json(root / "prompt_coverage.json", coverage)
        frozen = {
            "modes": MODES,
            "ratio": 0.1,
            "check_layer": 1,
            "tail_tokens": 64,
            "boundary_rule": (
                "equal-depth nonfirst chunk prefixes, including delimiters"
            ),
            "native_controls": "fresh full and prefix in each repeat",
            "comparison_limits": (
                "boundary is selection-rule-only; all arms retain early-layer probe"
            ),
            "minimum_length": 1024,
            "primary_comparison": "tail64 versus query",
            "input_sha256": {k: v["input_sha256"] for k, v in coverage.items()},
            "orders": [list(MODES), list(reversed(MODES))],
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
            ).strip(),
            "scripts_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (PROJECT / "scripts").glob("*.py")
            },
        }
        atomic_json(root / "frozen_config.json", frozen)
        script("verify_sources.py", label="sources_before")
        script("verify_runtime.py", label="runtime_before")
        script("parent_guard.py", "check", label="parent_before")
        progress["gpu"] = wait_idle(args.gpus)
        actual_hashes = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (PROJECT / "scripts").glob("*.py")
        }
        if actual_hashes != frozen["scripts_sha256"]:
            raise ValueError("experiment scripts changed while waiting")
        progress["status"] = "applying_patches"
        save()
        # Register before applying so a partially completed operation is restored.
        for name in (
            "apply_strict_causal_patch.py",
            "apply_precheck_kv_patch.py",
            "apply_query_fair_patch.py",
        ):
            applied.append(name)
            script(name)
        script(
            "validate_query_fair_gpu.py",
            "--gpu",
            progress["gpu"],
            label="gpu_selector_validation",
        )
        # A fixed size screen; no response-dependent tuning or stage omission.
        for repeat, order in enumerate(frozen["orders"]):
            controls = None
            for mode in order:
                name = f"rep{repeat}_{mode}"
                run(name, mode, coverage, controls)
                if controls is None:
                    controls = root / name
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
        if progress["status"] == "measurements_completed":
            progress["status"] = "completed"
        progress["finished"] = time.time()
        save()
    if progress["status"] == "completed":
        try:
            script("analyze_query_fair.py", "--run", root, label="analysis")
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
