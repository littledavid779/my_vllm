#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two opposite-order pairs with exact request-to-restore correlation."""

import argparse
import csv
from pathlib import Path

from analyze_lifecycle_request_path import attach_path, validate_path
from analyze_lifecycle_request_ttft import (
    compare,
    group_stats,
    index_rows,
    restore_audit,
)
from analyze_lifecycle_restore_path import analyze_restore_path
from run_lifecycle_fixed_workload import (
    CODE,
    ROOT,
    audit_arm,
    digest,
    expected_requests,
    read,
    run_arm,
    save,
)
from run_lifecycle_ttft_confirmation import config

ORDERS = [["no-offload", "selective-2g"], ["selective-2g", "no-offload"]]


def correlate(rows, stats):
    indexed = index_rows(rows)
    external = {}
    for key, row in indexed.items():
        request_id = row.get("external_request_id")
        if request_id != f"{key[0]}-turn{key[1]}" or request_id in external:
            raise ValueError("Missing, duplicate, or incorrect external request ID")
        external[request_id] = key
    # Shared fail-closed audit checks log completeness and every event ID.
    events = restore_audit(stats, set(external))
    actions = {}
    for rid, key in external.items():
        completed = events.get(rid, [])
        actions[key] = dict(
            action="restore" if completed else "no_external_restore",
            restore_events=len(completed),
            summed_restore_wait_ms=sum(e["observed_wait_s"] * 1000 for e in completed),
            restore_blocks=0,
        )
    # Multiple promotions on one request remain explicit, not silently deduped.
    for event in stats.get("lcuc_events", []):
        if event["kind"] == "promotion":
            rid = event["request_id"].removeprefix("chatcmpl-").rsplit("-", 1)[0]
            actions[external[rid]]["restore_blocks"] += event["blocks"]
    return indexed, actions


def summarize_pair(base_rows, selected_rows, stats, repeat):
    candidate, actions = correlate(selected_rows, stats)
    paired = compare(index_rows(base_rows), candidate, {}, repeat, "no-offload")
    for row in paired:
        action = actions[(row["conversation_id"], row["message_count"])]
        row.update(action)
        row.pop("restore_group")
        row["action_length"] = f"{row['action']}/{row['length_group']}"
    groups = {
        d: group_stats(paired, d) for d in ("action", "action_length", "turn_group")
    }
    return dict(
        repeat=repeat,
        groups=groups,
        top_regressions=sorted(paired, key=lambda r: r["delta_ms"], reverse=True)[:20],
    ), paired


def main(args):
    path_trace = args.request_path_trace
    restore_trace = args.restore_path_trace
    if restore_trace and not path_trace:
        raise ValueError("Restore path diagnosis requires --request-path-trace")
    source = read(args.source / "manifest.json")
    if args.model != source["model"]:
        raise ValueError("Model differs from calibration")
    data = read(args.source / "evaluation.json")
    fit = read(args.source / "calibration_fit.json")
    probe = source["cold_probe_config"]
    expected = expected_requests(data)
    files = CODE + [
        "scripts/run_lifecycle_ttft_confirmation.py",
        "scripts/run_lifecycle_correlated_diagnostic.py",
        "scripts/analyze_lifecycle_request_ttft.py",
        "vllm/v1/core/block_pool.py",
        "vllm/v1/engine/input_processor.py",
        "vllm/v1/core/sched/request_trace.py",
        "vllm/v1/core/sched/scheduler.py",
        "vllm/v1/engine/core.py",
        "scripts/analyze_lifecycle_request_path.py",
        "scripts/analyze_lifecycle_restore_path.py",
        "vllm/v1/kv_offload/restore_trace.py",
        "vllm/v1/executor/uniproc_executor.py",
    ]
    manifest = dict(
        suite="correlated-request-diagnostic",
        orders=ORDERS,
        gpu=args.gpu,
        port=args.port,
        model=args.model,
        requests_per_arm=sum(expected.values()),
        source=str(args.source.resolve()),
        input_sha256=digest(args.source / "evaluation.json"),
        fit_sha256=digest(args.source / "calibration_fit.json"),
        code_sha256={p: digest(ROOT / p) for p in files},
        configurations={a: config(a, fit, probe) for a in ORDERS[0]},
        correlate_requests=True,
        retries=0,
        preserve_conversation_ownership=True,
        request_path_trace=path_trace,
        restore_path_trace=restore_trace,
        restore_feedback_required=restore_trace,
        limitations=[
            "Diagnostic, two pairs only; not performance confirmation.",
            "Same input is not same runtime state; action groups are post-treatment.",
            "No external restore does not mean no GPU cache hit.",
            "Restore wait is observed, recompute counterfactual is not.",
            "Shared host, target-GPU sampling is not exclusive reservation.",
        ],
    )
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "manifest.json", manifest)
    save(args.output / "evaluation.json", data)
    save(args.output / "calibration_fit.json", fit)
    metrics_rows, reports, paired_rows = [], [], []
    signature = None
    for repeat, order in enumerate(ORDERS):
        cells = {}
        for position, arm in enumerate(order):
            for p, sha in manifest["code_sha256"].items():
                if digest(ROOT / p) != sha:
                    raise ValueError(f"Frozen code changed: {p}")
            for p, key in (
                ("evaluation.json", "input_sha256"),
                ("calibration_fit.json", "fit_sha256"),
            ):
                if digest(args.output / p) != manifest[key]:
                    raise ValueError(f"Frozen input changed: {p}")
            m = run_arm(
                args,
                f"repeat_{repeat}/{arm}",
                args.output / "evaluation.json",
                manifest["configurations"][arm],
                0,
                complete_workload=True,
                correlate_requests=True,
                request_path_trace=path_trace,
                restore_path_trace=restore_trace,
            )
            sig = audit_arm(m, expected, probe)
            if signature is not None and sig != signature:
                raise ValueError("Input token signature mismatch")
            signature = sig
            path = Path(m["stats_path"])
            stats = read(path)["stats"]
            for key in (
                "offload_cost_events_enabled",
                "offload_admission_trace_enabled",
                "lcuc_guard_transitions",
                "lcuc_cold_probe_blocks_spent",
            ):
                if stats.get(key, 0):
                    raise ValueError(f"Unexpected instrumentation/action: {key}")
            with path.with_name("request_summary.csv").open() as f:
                rows = list(csv.DictReader(f))
            _, actions = correlate(rows, stats)
            if path_trace:
                validate_path(rows, stats)
            if restore_trace and arm == "selective-2g":
                save(
                    args.output / m["name"] / "restore_path_analysis.json",
                    analyze_restore_path(stats, require_feedback=True),
                )
            if arm == "selective-2g" and not any(
                a["restore_events"] for a in actions.values()
            ):
                raise ValueError("No restore events; diagnosis has no target")
            spec = read(args.output / m["name"] / "spec.json")
            expected_owner_map = {}
            for row in rows:
                external = "chatcmpl-" + row["external_request_id"]
                expected_owner_map[external] = "chatcmpl-" + row["conversation_id"]
            if spec.get("request_owner_map") != expected_owner_map:
                raise ValueError("Conversation owner map mismatch")
            m.update(
                repeat=repeat,
                arm=arm,
                position=position,
                d2h_bytes=None,
                h2d_bytes=None,
                dma_measurement_available=False,
                event_mapping_complete=True,
                restore_requests=sum(
                    a["action"] == "restore" for a in actions.values()
                ),
            )
            save(args.output / m["name"] / "metrics.json", m)
            metrics_rows.append(m)
            save(args.output / "evaluation_summary.json", metrics_rows)
            cells[arm] = rows, stats
            print(
                f"CORRELATION PASS {len(metrics_rows)}/4 {arm}: "
                f"{m['restore_requests']} restore requests",
                flush=True,
            )
        report, paired = summarize_pair(
            cells["no-offload"][0], *cells["selective-2g"], repeat
        )
        if path_trace:
            report["path_groups"] = attach_path(
                paired, *cells["no-offload"], *cells["selective-2g"]
            )
        reports.append(report)
        paired_rows.extend(paired)
        save(args.output / "request_analysis.json", reports)
        with (args.output / "paired_requests.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(paired_rows[0]))
            writer.writeheader()
            writer.writerows(paired_rows)
    print(
        "COMPLETE: four runs; exact event mapping; descriptive analysis only.",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--port", type=int, default=8238)
    parser.add_argument("--model", default="/public/model_zoo/Qwen3-0.6B")
    parser.add_argument("--request-path-trace", action="store_true")
    parser.add_argument("--restore-path-trace", action="store_true")
    main(parser.parse_args())
