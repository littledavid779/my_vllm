# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize readiness locations, retaining output-stamp/Future distinctions."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean

from run_lifecycle_calibrated_smoke import ROOT, digest, read, save


def main(args):
    manifest = read(args.source / "manifest.json")
    for name, sha in manifest["code_sha256"].items():
        if digest(ROOT / name) != sha:
            raise ValueError(f"Frozen source changed: {name}")
    if digest(args.source / "calibration.json") != manifest["input_sha256"]:
        raise ValueError("Input changed")
    rows, samples = [], 0
    for arm in read(args.source / "summary.json"):
        path = args.source / arm["name"]
        analysis = read(path / "restore_readiness.json")
        stats = read(Path(arm["stats_path"]))["stats"]
        gpu = read(path / "gpu_samples.json")
        if gpu["foreign_pids"]:
            raise ValueError("Foreign GPU processes")
        samples += len(gpu["samples"])
        workers = {
            e["request_id"]: e["workers"][0] for e in stats["restore_path_trace_events"]
        }
        batches = {
            rid: b
            for b in stats["restore_readiness_trace_events"]
            for rid in b["request_ids"]
        }
        for event in analysis["events"]:
            rid = event["request_id"]
            batch, worker = batches[rid], workers[rid]
            timeline = {p["stage"]: p for p in batch["consumption"]}
            output_s = worker["executor_output_ready_s"]
            start = timeline["before_execute"]["start_s"]
            end = timeline["after_execute"]["end_s"]
            if event["entry_done"] is True:
                location = "ready_before_entry"
            elif event["created_and_consumed_same_loop"]:
                location = "created_this_loop"
            else:
                location = event["first_done_stage"]
            rows.append(
                dict(
                    arm=arm["name"],
                    request_id=rid,
                    batch_id=event["batch_id"],
                    location=location,
                    first_done_stage=event["first_done_stage"],
                    output_stamp_phase=event["output_stamp_phase"],
                    entry_done=event["entry_done"],
                    execute_ms=1000 * (end - start),
                    output_to_execute_return_ms=1000
                    * max(0, end - max(start, output_s)),
                    execute_return_to_consumption_ms=(
                        1000 * (worker["engine_output_consumed_s"] - end)
                    ),
                    ready_to_engine_ms=event["ready_to_engine_ms"],
                    before_schedule_ms=event["before_schedule_ms"],
                    visibility_bracket_ms=event["visibility_bracket_ms"],
                )
            )
    groups = {}
    for location in sorted({r["location"] for r in rows}):
        selected = [r for r in rows if r["location"] == location]
        groups[location] = dict(
            requests=len(selected),
            **{
                k: mean(r[k] for r in selected)
                for k in (
                    "execute_ms",
                    "output_to_execute_return_ms",
                    "execute_return_to_consumption_ms",
                    "ready_to_engine_ms",
                )
            },
        )
    summary = dict(
        source=str(args.source),
        requests=len(rows),
        gpu_samples=samples,
        source_files_verified=len(manifest["code_sha256"]),
        locations=dict(Counter(r["location"] for r in rows)),
        groups=groups,
        limitations=[
            "Counts and means are request-weighted, not unique-batch-weighted.",
            "Output stamp precedes Future completion; no exact completion clock.",
            "Observed intervals do not establish recoverable TTFT savings.",
            "GPU1 runs other jobs; shared host interference is not excluded.",
        ],
    )
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "summary.json", summary)
    with (args.output / "events.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
