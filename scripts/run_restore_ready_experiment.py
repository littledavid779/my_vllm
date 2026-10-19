# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed mixed actions, opposite-order ready-first diagnostics and confirmation."""

import argparse
import csv
import hashlib
from pathlib import Path

from analyze_promotion_restore_joint import join_restore_labels
from analyze_promotion_ttft_labels import build_labels
from analyze_restore_host import analyze_host
from analyze_restore_readiness import analyze_readiness
from run_lifecycle_calibrated_smoke import ROOT, digest, read, run_arm, save
from run_lifecycle_fixed_workload import audit_arm, expected_requests
from run_promotion_ttft_calibration import mixed_config


def model_fingerprint(model):
    root = Path(model)
    if not (root / "config.json").is_file():
        raise ValueError(f"Local model configuration missing: {root}")
    files = sorted(
        p for p in root.iterdir() if p.suffix in (".json", ".txt", ".safetensors")
    )
    if not any(p.suffix == ".safetensors" for p in files):
        raise ValueError(f"Local model weights missing: {root}")
    hashes = {}
    for p in files:
        with p.open("rb") as f:
            hashes[p.name] = hashlib.file_digest(f, "sha256").hexdigest()
    return hashes


def main(args):
    old = read(args.source / "manifest.json")
    if old["model"] != args.model:
        if not args.allow_model_relocation:
            raise ValueError("Model mismatch; relocation requires an explicit flag")
        if Path(old["model"]).name != Path(args.model).name:
            raise ValueError("Relocation must preserve the model directory name")
    model_hashes = model_fingerprint(args.model)
    original = Path(old["source"])
    data = read(args.source / "calibration.json")
    fit = read(args.source / "calibration_fit.json")
    probe = read(original / "manifest.json")["cold_probe_config"]
    expected = expected_requests(data)
    diagnostic = args.phase in ("diagnostic", "readiness", "host-execute")
    readiness = args.phase in ("readiness", "host-execute")
    host_trace = args.phase == "host-execute"
    modes = (
        (("observe", "on"), ("on", "observe"))
        if diagnostic
        else (("no-offload", "off", "on"), ("on", "off", "no-offload"))
    )
    if readiness:
        modes = (("observe",), ("observe",))
    configs = {str(seed): mixed_config(data, fit, probe, seed) for seed in (17, 29)}
    # Keep the bounded action audit required by calibration_mixed in both
    # phases. Confirmation disables request-path and restore-stage timing.
    for name, cfg in configs.items():
        if cfg != old["run_configurations"]["mixed_" + name]:
            raise ValueError("Action assignments or configuration changed")
    files = list(old["code_sha256"]) + [
        "vllm/v1/kv_offload/restore_ready.py",
        "scripts/run_restore_ready_experiment.py",
        "scripts/analyze_restore_readiness.py",
        "scripts/analyze_restore_host.py",
        "vllm/v1/kv_offload/restore_host_trace.py",
        "vllm/v1/worker/gpu_worker.py",
        "vllm/v1/worker/gpu_model_runner.py",
        "vllm/v1/worker/kv_connector_model_runner_mixin.py",
    ]
    manifest = dict(
        phase=args.phase,
        orders=modes,
        assignment_seeds=[17, 29],
        source=str(args.source),
        model=args.model,
        source_model=old["model"],
        model_relocated=old["model"] != args.model,
        model_sha256=model_hashes,
        gpu=args.gpu,
        input_sha256=digest(args.source / "calibration.json"),
        configurations=configs,
        code_sha256={p: digest(ROOT / p) for p in files},
        limitations=[
            "Diagnostic uses restore/request path timing; confirmation disables these.",
            "Bounded decision trace retained for fixed mixed action checks.",
            "Same requests/assignments do not imply identical runtime/cache states.",
            "Source has no model-file hashes; "
            "historical weight identity is unverified.",
        ],
    )
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "manifest.json", manifest)
    save(args.output / "calibration.json", data)
    all_metrics, signature = [], None
    for repeat, (seed, order) in enumerate(zip((17, 29), modes)):
        for position, mode in enumerate(order):
            for p, sha in manifest["code_sha256"].items():
                if digest(ROOT / p) != sha:
                    raise ValueError(f"Frozen source changed: {p}")
            if digest(args.output / "calibration.json") != manifest["input_sha256"]:
                raise ValueError("Input changed")
            config = None if mode == "no-offload" else configs[str(seed)]
            actual_mode = "off" if mode == "no-offload" else mode
            name = f"repeat_{repeat}/{mode}"
            m = run_arm(
                args,
                name,
                args.output / "calibration.json",
                config,
                0,
                complete_workload=True,
                correlate_requests=True,
                request_path_trace=diagnostic,
                restore_path_trace=diagnostic,
                restore_ready_mode=actual_mode,
                restore_readiness_trace=readiness,
                restore_host_trace=host_trace,
            )
            sig = audit_arm(m, expected, probe)
            if signature is not None and signature != sig:
                raise ValueError("Input/client signature changed")
            signature = sig
            path = Path(m["stats_path"])
            stats = read(path)["stats"]
            if stats.get("restore_ready_first_mode") != actual_mode:
                raise ValueError("Ready-first mode did not take effect")
            if stats.get("restore_ready_first_unsupported_future", 0):
                raise ValueError("Unsupported Future implementation")
            consumed = stats.get("restore_ready_first_consumed", 0)
            eligible = stats.get("restore_ready_first_eligible", 0)
            if consumed != (eligible if mode == "on" else 0):
                raise ValueError("Early-consumption accounting mismatch")
            if config is not None:
                assignments = config["kv_connector_extra_config"][
                    "calibration_action_assignments"
                ]
                events = stats.get("promotion_ttft_trace_events", [])
                if len(events) != stats.get(
                    "offload_promotion_decision_count"
                ) or stats.get("promotion_ttft_trace_dropped", 0):
                    raise ValueError("Incomplete decision trace")
                for e in events:
                    if (
                        assignments.get(e["request_id"].rsplit("-", 1)[0])
                        != e["action"]
                    ):
                        raise ValueError("Action differs from frozen assignment")
            with path.with_name("request_summary.csv").open() as f:
                rows = list(csv.DictReader(f))
            if diagnostic:
                labels = build_labels(rows, stats)
                joined = join_restore_labels(rows, stats)
                save(args.output / name / "promotion_ttft_labels.json", labels)
                save(args.output / name / "promotion_restore_joint.json", joined)
                m["ready_to_engine_ms"] = joined["restore_path"]["feedback_means"][
                    "ready_to_engine_ms"
                ]
                m["before_schedule_ms"] = sum(joined["means"].values())
                m["restore_events"] = joined["count"]
                if readiness:
                    analysis = analyze_readiness(stats, joined)
                    save(args.output / name / "restore_readiness.json", analysis)
                    m["readiness"] = {
                        k: v for k, v in analysis.items() if k != "events"
                    }
                    if host_trace:
                        host_analysis = analyze_host(stats, analysis)
                        save(args.output / name / "restore_host.json", host_analysis)
                        m["host_execution"] = {
                            k: v for k, v in host_analysis.items() if k != "events"
                        }
            elif stats.get("request_path_trace_enabled", 0) or stats.get(
                "restore_path_trace_enabled", 0
            ):
                raise ValueError("Timing trace unexpectedly enabled")
            for key in (
                "offload_cost_events_enabled",
                "offload_admission_trace_enabled",
                "lcuc_guard_transitions",
            ):
                if stats.get(key, 0):
                    raise ValueError(f"Unexpected instrumentation or guard: {key}")
            m.update(
                repeat=repeat,
                assignment_seed=seed,
                position=position,
                mode=mode,
                ready_first={
                    k: v
                    for k, v in stats.items()
                    if k.startswith("restore_ready_first_")
                },
                d2h_bytes=None,
                h2d_bytes=None,
            )
            save(args.output / name / "metrics.json", m)
            all_metrics.append(m)
            save(args.output / "summary.json", all_metrics)
            print(
                f"AUDIT PASS {name}: eligible={eligible}, consumed={consumed}",
                flush=True,
            )
    if model_fingerprint(args.model) != model_hashes:
        raise ValueError("Model files changed during the experiment")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=("diagnostic", "confirmation", "readiness", "host-execute"),
        required=True,
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--port", type=int, default=8238)
    parser.add_argument("--model", default="/public/model_zoo/Qwen3-0.6B")
    parser.add_argument("--allow-model-relocation", action="store_true")
    main(parser.parse_args())
