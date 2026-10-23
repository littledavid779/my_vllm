"""Run strict question/template ablations on the original and later source splits."""

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

from apply_query_template_patch import HELPER, MODES, TEMPLATE
from run_smoke import gpu_state

PROJECT = Path(__file__).resolve().parents[1]


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    os.replace(temporary, path)


DATASETS = ("strict_holdout_v1", "query_fair_v1")
CONTROLS = {
    "strict_holdout_v1": PROJECT / "results/query_controls_20261023/rep0_topk",
    "query_fair_v1": PROJECT / "results/query_fair_20261023/rep0_query",
}


def inspect_inputs():
    from apply_query_control_patch import HELPER as PREVIOUS_HELPER
    from transformers import AutoTokenizer

    scope = {}
    exec(
        compile(
            ast.parse(PREVIOUS_HELPER + HELPER), "query-template-boundaries", "exec"
        ),
        scope,
    )
    findings = {}
    for dataset in DATASETS:
        path = PROJECT / "data" / (dataset + ".json")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if (
            digest
            != json.loads(path.with_suffix(".lock.json").read_text())["input_sha256"]
        ):
            raise ValueError("input lock mismatch")
        data = json.loads(path.read_text())
        tokenizer = AutoTokenizer.from_pretrained(data["model"], local_files_only=True)
        if tokenizer.encode("Question:", add_special_tokens=False) != [14582, 25]:
            raise ValueError("question marker changed")
        if (
            tokenizer.decode(TEMPLATE)
            != "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        ):
            raise ValueError("template tokenization changed")
        records = []
        for session in data["sessions"]:
            for version, item in enumerate(session["versions"]):
                ids = item["token_ids"]
                qlen, tlen = scope["query_template_spans"](ids[-64:])
                if (
                    not qlen
                    or tlen != 9
                    or qlen + tlen != scope["query_control_span"](ids[-64:])
                ):
                    raise ValueError("split does not partition previous combined span")
                qtext = tokenizer.decode(ids[-qlen - tlen : -tlen])
                if (
                    qtext.strip()
                    != ("Question: " + session["metadata"]["question"]).strip()
                ):
                    raise ValueError("question span misses text or includes template")
                if ids[-tlen:] != TEMPLATE or qlen + tlen > int(len(ids) * 0.1):
                    raise ValueError("invalid template or repair budget")
                records.append(
                    {
                        "session_id": session["session_id"],
                        "version": version,
                        "length": len(ids),
                        "question_tokens": qlen,
                        "template_tokens": tlen,
                        "prompt_sha": hashlib.sha256(
                            json.dumps(ids).encode()
                        ).hexdigest(),
                    }
                )
        if len(records) != 384 or len(data["sessions"]) != 96:
            raise ValueError("unexpected fixed input size")
        for field in ("warmup_seed_ids", "warmup_edit_ids"):
            if scope["query_template_spans"](data[field][-64:]) != (0, 0):
                raise ValueError("warmup unexpectedly contains a question")
        controls = CONTROLS[dataset]
        manifest = json.loads((controls / "manifest.json").read_text())
        if manifest["status"] != "completed" or manifest["input_sha256"] != digest:
            raise ValueError("native quality reference mismatch")
        gold = {
            (s["session_id"], i): v["expected_value"]
            for s in data["sessions"]
            for i, v in enumerate(s["versions"])
            if i > 0
        }
        rows = [
            json.loads(x)
            for x in (controls / "requests.jsonl").read_text().splitlines()
        ]
        for arm in ("full", "prefix"):
            selected = [r for r in rows if r["arm"] == arm]
            if len(selected) != len(gold) or {
                (r["session_id"], r["version"]) for r in selected
            } != set(gold):
                raise ValueError("native reference is incomplete")
            if any(
                r["expected_value"] != gold[r["session_id"], r["version"]]
                for r in selected
            ):
                raise ValueError("native reference labels differ")
        findings[dataset] = {
            "input_sha256": digest,
            "prompts": records,
            "sources": len({s["source_id"] for s in data["sessions"]}),
            "question_length_range": [
                min(r["question_tokens"] for r in records),
                max(r["question_tokens"] for r in records),
            ],
            "query_length_range": [
                min(r["question_tokens"] + 9 for r in records),
                max(r["question_tokens"] + 9 for r in records),
            ],
            "template_tokens": 9,
            "native_control_files": {
                str(controls / n): hashlib.sha256(
                    (controls / n).read_bytes()
                ).hexdigest()
                for n in ("manifest.json", "requests.jsonl")
            },
        }
    return findings


def audit(path, mode, coverage):
    manifest = json.loads((path / "manifest.json").read_text())
    if (
        manifest["status"] != "completed"
        or manifest["config"]["query_template"] != mode
    ):
        raise ValueError("incomplete or mismatched stage")
    expected = {r["prompt_sha"]: r for r in coverage["prompts"] if r["version"] > 0}
    if len(expected) != sum(r["version"] > 0 for r in coverage["prompts"]):
        raise ValueError("duplicate measured prompt")
    observed = collections.Counter()
    measured = []
    entries = [
        json.loads(s)
        for s in re.findall(
            r"QUERY_TEMPLATE (\{[^\n]+\})", (path / "cacheblend10.log").read_text()
        )
    ]
    for entry in entries:
        total, selected = entry["total"], entry["selected"]
        qlen, tlen = entry["question_tokens"], entry["template_tokens"]
        if (
            entry["mode"] != mode
            or selected != max(1, int(total * 0.1))
            or entry["unique"] != selected
            or not 0 <= entry["min_index"] <= entry["max_index"] < total
        ):
            raise ValueError("selection mode, budget, or index mismatch")
        if (
            not 0 <= entry["question_bits"] < 2**qlen
            or not 0 <= entry["template_bits"] < 2**tlen
        ):
            raise ValueError("coverage bits outside region")
        forced = 0
        if total >= 1024 and qlen + tlen and mode != "topk":
            qwant = (1 << qlen) - 1 if mode in ("question", "both") else 0
            twant = (1 << tlen) - 1 if mode in ("template", "both") else 0
            forced = qwant.bit_count() + twant.bit_count()
            if entry["question_bits"] != qwant or entry["template_bits"] != twant:
                raise ValueError("required region missing or excluded region selected")
        if entry["forced"] != forced:
            raise ValueError("mandatory count mismatch")
        sha = entry["prompt_sha"]
        if sha in expected:
            prompt = expected[sha]
            if (total, qlen, tlen) != (
                prompt["length"],
                prompt["question_tokens"],
                prompt["template_tokens"],
            ):
                raise ValueError("runtime prompt boundaries differ from CPU preflight")
            observed[sha] += 1
            measured.append(
                {
                    **entry,
                    "session_id": prompt["session_id"],
                    "version": prompt["version"],
                }
            )
    if set(observed) != set(expected) or any(n != 1 for n in observed.values()):
        raise ValueError("missing or duplicated measured selection evidence")
    atomic_json(
        path / "selection_audit.json",
        {
            "mode": mode,
            "measured_calls": len(measured),
            "entries": measured,
            "all_calls": len(entries),
            "strict_exclusion_verified": mode != "topk",
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
                        "question_length_range": v["question_length_range"],
                        "template_tokens": v["template_tokens"],
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

    def run(name, dataset, mode, coverage):
        controls = CONTROLS[dataset]
        wait_idle([progress["gpu"]])
        entry = {"name": name, "mode": mode, "started": time.time()}
        progress.update(status="running", current_stage=name)
        progress["stages"].append(entry)
        save()
        data = PROJECT / "data" / (dataset + ".json")
        if (
            hashlib.sha256(data.read_bytes()).hexdigest()
            != coverage[dataset]["input_sha256"]
        ):
            raise ValueError("input changed while waiting")
        for path, digest in coverage[dataset]["native_control_files"].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise ValueError("native reference changed while waiting")
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
            "--query-template",
            mode,
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
            "template_ids": TEMPLATE,
            "native_controls": "reused for quality only, no fresh native timings",
            "strict_factorial_modes": ["none", "question", "template", "both"],
            "anchor": "unchanged global topk; may naturally include Q/T",
            "minimum_length": 1024,
            "primary_comparisons": ["both versus question", "both versus template"],
            "datasets": DATASETS,
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
            "apply_query_template_patch.py",
        ):
            applied.append(name)
            script(name)
        script(
            "validate_query_template_gpu.py",
            "--gpu",
            progress["gpu"],
            label="gpu_selector_validation",
        )
        # Complete both datasets without response-dependent tuning or omission.
        for dataset in DATASETS:
            for repeat, order in enumerate(frozen["orders"]):
                for mode in order:
                    name = f"{dataset}_rep{repeat}_{mode}"
                    run(name, dataset, mode, coverage)
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
            script("analyze_query_template.py", "--run", root, label="analysis")
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
