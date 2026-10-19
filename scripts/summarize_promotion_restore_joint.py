# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Describe joined restore phases against an unchanged historical cost model."""

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from analyze_promotion_restore_joint import PHASES, join_restore_labels
from diagnose_promotion_ttft_errors import attach_paths, raw_predict
from fit_promotion_ttft import fit, group_split, load_labels, predict, read

GROUPS = {
    "decision_to_submit_ms": PHASES[:5],
    "host_submit_ms": ("host_submit_ms",),
    "submit_to_worker_observed_ms": ("submit_return_to_worker_observed_ms",),
    "worker_to_output_ready_ms": ("worker_to_output_ready_ms",),
    "output_ready_to_engine_ms": ("ready_to_engine_ms",),
    "engine_to_connector_ms": ("engine_to_update_ms", "update_to_connector_ms"),
    "scheduler_to_compute_ms": ("scheduler_to_first_compute_ms",),
}


def correlation(x, y):
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def summarize(events):
    before = [e["before_schedule_ms"] for e in events]
    error = [e["before_error_ms"] for e in events]
    phases = {}
    for name, fields in GROUPS.items():
        values = [sum(e[f] for f in fields) for e in events]
        phases[name] = dict(
            mean_ms=float(np.mean(values)),
            std_ms=float(np.std(values)),
            p95_ms=float(np.quantile(values, 0.95)),
            corr_with_before_error=correlation(values, error),
        )
    if not np.isclose(sum(p["mean_ms"] for p in phases.values()), np.mean(before)):
        raise ValueError("Aggregated partition mismatch")
    return dict(
        n=len(events),
        conversations=len({e["conversation"] for e in events}),
        before_mean_ms=float(np.mean(before)),
        before_std_ms=float(np.std(before)),
        before_mae_ms=float(np.mean(np.abs(error))),
        before_bias_ms=float(np.mean(error)),
        total_mae_ms=float(np.mean([abs(e["total_error_ms"]) for e in events])),
        dma_mean_ms=float(np.mean([e["dma_elapsed_ms"] for e in events])),
        phases=phases,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--frozen-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    provenance = {}

    def tracked(path):
        provenance[str(path)] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        return read(path)

    frozen = tracked(args.frozen_model)
    roots = [Path(p) for p in frozen["sources"]]
    paths = {}
    for root in roots:
        tracked(root / "manifest.json")
        if (
            provenance[str(root / "manifest.json")]
            != frozen["source_manifest_sha256"][str(root)]
        ):
            raise ValueError("Historical manifest differs")
        for arm in tracked(root / "summary.json"):
            for e in tracked(arm["stats_path"])["stats"]["request_path_trace_events"]:
                paths[(str(root / arm["name"]), e["request_id"])] = e
    old = attach_paths(load_labels(roots), paths)
    train = [e for e in old if e["split"] == "train" and e["action"] == "restore"]
    total_model = frozen["actions"]["restore"]["models"]["ridge"]["model"]
    reproduced = fit(train, "ridge")
    for field in ("center", "scale", "coef", "intercept"):
        if not np.allclose(reproduced[field], total_model[field], rtol=1e-9, atol=1e-9):
            raise ValueError("Historical model changed")
    before_model = fit(
        [dict(e, decision_to_first_token_ms=e["before_schedule_ms"]) for e in train],
        "ridge",
    )
    manifest = tracked(args.source / "manifest.json")
    if not manifest.get("restore_diagnostic"):
        raise ValueError("Not a joint diagnostic run")
    for root in roots:
        previous = read(root / "manifest.json")
        if any(previous[k] != manifest[k] for k in ("model", "input_sha256")):
            raise ValueError("Incompatible model or workload")
    events = []
    for arm in tracked(args.source / "summary.json"):
        path = Path(arm["stats_path"])
        stats = tracked(path)["stats"]
        with path.with_name("request_summary.csv").open() as f:
            rows = list(csv.DictReader(f))
        provenance[str(path.with_name("request_summary.csv"))] = hashlib.sha256(
            path.with_name("request_summary.csv").read_bytes()
        ).hexdigest()
        joined = join_restore_labels(rows, stats)
        if joined != tracked(
            args.source / arm["name"] / "promotion_restore_joint.json"
        ):
            raise ValueError("Joint records differ from raw data")
        owners = {r["external_request_id"]: r["conversation_id"] for r in rows}
        for e in joined["events"]:
            match = re.fullmatch(r"chatcmpl-(.+)-[0-9a-f]{8}", e["request_id"])
            if match is None or match[1] not in owners:
                raise ValueError("Missing conversation")
            conv = owners[match[1]]
            pred_before = float(raw_predict(before_model, [e])[0])
            pred_total = float(predict(total_model, [e])[0])
            events.append(
                dict(
                    e,
                    conversation=conv,
                    split=group_split(conv),
                    arm=arm["name"],
                    predicted_before_ms=pred_before,
                    before_error_ms=pred_before - e["before_schedule_ms"],
                    predicted_total_ms=pred_total,
                    total_error_ms=pred_total - e["decision_to_first_token_ms"],
                )
            )
    by_arm, by_split = defaultdict(list), defaultdict(list)
    for e in events:
        by_arm[e["arm"]].append(e)
        by_split[e["split"]].append(e)
    for file in (
        Path(__file__),
        Path(__file__).with_name("fit_promotion_ttft.py"),
        Path(__file__).with_name("diagnose_promotion_ttft_errors.py"),
        Path(__file__).with_name("analyze_promotion_restore_joint.py"),
    ):
        provenance[str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
    report = dict(
        all=summarize(events),
        by_arm={a: summarize(rs) for a, rs in by_arm.items()},
        by_split={s: summarize(rs) for s, rs in by_split.items()},
        top_before_errors=sorted(
            events, key=lambda e: abs(e["before_error_ms"]), reverse=True
        )[:10],
        frozen_before_model=before_model,
        provenance_sha256=provenance,
        limitations=[
            "No refit on new timing labels; "
            "new events repeat historical conversations.",
            "Phase size, variability and error correlation "
            "do not establish causal attribution.",
            "Host-observed completion includes polling; DMA is not an additive phase.",
            "Trace-on diagnostic, "
            "no performance comparison with earlier trace-off runs.",
        ],
    )
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "summary.json").open("w") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    with (args.output / "restore_events.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(events[0]))
        writer.writeheader()
        writer.writerows(events)
    print(json.dumps(report["all"], indent=2))


if __name__ == "__main__":
    main()
