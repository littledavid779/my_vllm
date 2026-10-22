"""Run isolated official CacheBlend smoke and require real selector activity."""

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def gpu_state(gpu):
    row = (
        subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                str(gpu),
                "--query-gpu=uuid,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        .strip()
        .split(",")
    )
    apps = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return {
        "uuid": row[0].strip(),
        "memory_mib": int(row[1]),
        "utilization": int(row[2]),
        "pids": [
            int(line.split(",")[1])
            for line in apps.splitlines()
            if line.split(",")[0].strip() == row[0].strip()
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/smoke.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if Path(sys.prefix).resolve() != PROJECT / ".venv":
        raise RuntimeError("use this project's .venv/bin/python")
    config = json.loads(args.config.read_text())
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    subprocess.run(
        [sys.executable, str(PROJECT / "scripts/verify_sources.py")], check=True
    )
    subprocess.run(
        [sys.executable, str(PROJECT / "scripts/verify_runtime.py")], check=True
    )
    subprocess.run(
        [sys.executable, str(PROJECT / "scripts/parent_guard.py"), "check"], check=True
    )
    before = gpu_state(config["gpu"])
    manifest = {
        "status": "planned",
        "gpu_before": before,
        "config": config,
        "sources": json.loads((PROJECT / "sources.lock.json").read_text()),
        "requirements_lock_sha256": hashlib.sha256(
            (PROJECT / "requirements.lock").read_bytes()
        ).hexdigest(),
        "scope": "Official functional smoke; no quality/performance claim",
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    if before["pids"] or before["memory_mib"] > 100 or before["utilization"] > 5:
        manifest["status"] = "deferred_gpu_busy"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        return 75
    env = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "LMCACHE_CONFIG_FILE"):
        env.pop(name, None)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(config["gpu"]),
            "VIRTUAL_ENV": str(PROJECT / ".venv"),
            "PATH": str(PROJECT / ".venv/bin") + os.pathsep + env.get("PATH", ""),
            "XDG_CACHE_HOME": str(PROJECT / ".cache"),
            "VLLM_USE_V1": "1",
            "PYTHONHASHSEED": "0",
            "MAX_JOBS": "4",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HOME": str(PROJECT / ".cache/huggingface"),
            "TORCHINDUCTOR_CACHE_DIR": str(PROJECT / ".cache/inductor"),
            "TRITON_CACHE_DIR": str(PROJECT / ".cache/triton"),
            "FLASHINFER_WORKSPACE_DIR": str(PROJECT / ".cache/flashinfer"),
            "VLLM_CACHE_ROOT": str(PROJECT / ".cache/vllm"),
            "VLLM_NO_USAGE_STATS": "1",
            "DO_NOT_TRACK": "1",
        }
    )
    command = [
        sys.executable,
        "-u",
        str(PROJECT / "scripts/official_smoke_worker.py"),
        "--config",
        str(args.config.resolve()),
        "--output",
        str(args.output / "rows.json"),
    ]
    child = None
    try:
        with (args.output / "run.log").open("w") as log:
            child = subprocess.Popen(
                command,
                env=env,
                cwd=PROJECT,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            manifest.update(status="running", worker_pid=child.pid)
            manifest_path.write_text(json.dumps(manifest, indent=2))
            code = child.wait(timeout=600)
        if code:
            raise RuntimeError(f"official smoke exited {code}; inspect run.log")
        text = (args.output / "run.log").read_text()
        selected = [
            int(n) for n in re.findall(r"Number of indices picked: (\d+)", text)
        ]
        if not selected or not all(n > 0 for n in selected):
            raise RuntimeError("no proof that official selective recomputation ran")
        rows = json.loads((args.output / "rows.json").read_text())["rows"]
        manifest.update(
            status="completed",
            selected_token_counts=selected,
            stages=len(rows),
            selector_verified=True,
        )
    except BaseException as exc:
        manifest.update(status="failed", error=repr(exc))
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        if child is not None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        manifest["gpu_after"] = gpu_state(config["gpu"])
        manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
