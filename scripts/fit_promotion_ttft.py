# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline, grouped calibration diagnostics; never exports a serving policy."""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from analyze_promotion_ttft_labels import build_labels

FEATURES = (
    "candidate_tokens",
    "remaining_prefill_if_restore",
    "gpu_hit_tokens",
    "tracked_requests",
    "pending_load_blocks",
    "pending_store_blocks",
)
ACTIONS = ("restore", "recompute")
SPLITS = ("train", "calibrate", "validate")


def read(path):
    return json.loads(Path(path).read_text())


def group_split(conversation):
    # Fixed before looking at label values. All turns/repeats stay together.
    n = int(hashlib.sha256(("promotion-v1:" + conversation).encode()).hexdigest(), 16)
    return "train" if n % 10 < 6 else ("calibrate" if n % 10 < 8 else "validate")


def load_labels(sources):
    labels, seen = [], set()
    input_hash = None
    for source in sources:
        manifest = read(source / "manifest.json")
        current = (manifest["input_sha256"], manifest["model"])
        if input_hash is not None and current != input_hash:
            raise ValueError("Different input or model: analyze separately")
        input_hash = current
        for arm in read(source / "summary.json"):
            stats_path = Path(arm["stats_path"])
            if stats_path.resolve() in seen:
                raise ValueError("Duplicate arm")
            seen.add(stats_path.resolve())
            with stats_path.with_name("request_summary.csv").open() as f:
                rows = list(csv.DictReader(f))
            owners = {r["external_request_id"]: r["conversation_id"] for r in rows}
            validated = build_labels(rows, read(stats_path)["stats"])
            if validated != read(source / arm["name"] / "promotion_ttft_labels.json"):
                raise ValueError("Saved labels differ from raw observations")
            for e in validated["labels"]:
                match = re.fullmatch(r"chatcmpl-(.+)-[0-9a-f]{8}", e["request_id"])
                if match is None or match[1] not in owners:
                    raise ValueError("Missing conversation ownership")
                owner = owners[match[1]]
                labels.append(
                    dict(
                        e,
                        conversation=owner,
                        split=group_split(owner),
                        arm=str(source / arm["name"]),
                    )
                )
    if not labels:
        raise ValueError("No labels")
    return labels


def cell(e):
    # Coarse diagnostic only, NOT a causal comparability guarantee.
    size = "small" if e["candidate_tokens"] <= 512 else "large"
    return (
        size,
        e["tracked_requests"] >= 2,
        e["pending_load_blocks"] > 0,
        e["pending_store_blocks"] > 0,
    )


def coverage(labels):
    counts = {a: Counter(cell(e) for e in labels if e["action"] == a) for a in ACTIONS}
    common = {
        c
        for c in set(counts[ACTIONS[0]]) | set(counts[ACTIONS[1]])
        if all(counts[a][c] >= 3 for a in ACTIONS)
    }
    return dict(
        cells=[
            dict(cell=list(c), **{a: counts[a][c] for a in ACTIONS})
            for c in sorted(set(counts[ACTIONS[0]]) | set(counts[ACTIONS[1]]))
        ],
        min_per_action=3,
        jointly_supported_events=sum(cell(e) in common for e in labels),
        total_events=len(labels),
        pending_load_nonzero={
            a: sum(e["action"] == a and e["pending_load_blocks"] > 0 for e in labels)
            for a in ACTIONS
        },
    )


def features(rows):
    return np.log1p(np.array([[e[f] for f in FEATURES] for e in rows], dtype=float))


def fit(rows, kind):
    if kind not in ("constant", "ridge"):
        raise ValueError("Unknown model kind")
    if not rows:
        raise ValueError("Empty training partition")
    y = np.array([e["decision_to_first_token_ms"] for e in rows])
    x = features(rows)
    center = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1
    z = (x - center) / scale
    # Preselected regularization, not tuned against validation labels.
    coef = (
        np.linalg.solve(z.T @ z + 10 * np.eye(len(FEATURES)), z.T @ (y - y.mean()))
        if kind == "ridge"
        else np.zeros(len(FEATURES))
    )
    return dict(
        kind=kind,
        intercept=float(y.mean()),
        center=center.tolist(),
        scale=scale.tolist(),
        coef=coef.tolist(),
        alpha=10,
    )


def predict(model, rows):
    x = (features(rows) - model["center"]) / model["scale"]
    return np.maximum(0, model["intercept"] + x @ model["coef"])


def errors(model, rows, margin=None):
    if not rows:
        return dict(n=0)
    residual = predict(model, rows) - [e["decision_to_first_token_ms"] for e in rows]
    ae = np.abs(residual)
    result = dict(
        n=len(rows),
        mae_ms=float(ae.mean()),
        p90_absolute_error_ms=float(np.quantile(ae, 0.9)),
        bias_ms=float(residual.mean()),
    )
    if margin is not None:
        result["empirical_margin_coverage"] = float(np.mean(ae <= margin))
    return result


def analyze(labels):
    report = dict(features=list(FEATURES), coverage=coverage(labels), actions={})
    for action in ACTIONS:
        parts = {
            s: [e for e in labels if e["action"] == action and e["split"] == s]
            for s in SPLITS
        }
        if any(not p for p in parts.values()):
            raise ValueError(f"Insufficient grouped partitions: {action}")
        models = {}
        train_cells = Counter(cell(e) for e in parts["train"])
        for kind in ("constant", "ridge"):
            model = fit(parts["train"], kind)
            cal = parts["calibrate"]
            margin = float(
                np.quantile(
                    np.abs(
                        predict(model, cal)
                        - [e["decision_to_first_token_ms"] for e in cal]
                    ),
                    0.9,
                )
            )
            models[kind] = dict(
                model=model,
                empirical_p90_margin_ms=margin,
                validation=errors(model, parts["validate"], margin),
            )
        report["actions"][action] = dict(
            partitions={
                s: dict(
                    events=len(p), conversations=len({e["conversation"] for e in p})
                )
                for s, p in parts.items()
            },
            validation_events_in_train_cells_with_3_samples=sum(
                train_cells[cell(e)] >= 3 for e in parts["validate"]
            ),
            models=models,
        )
    validation = [e for e in labels if e["split"] == "validate"]
    train_support = {
        a: Counter(
            cell(e) for e in labels if e["split"] == "train" and e["action"] == a
        )
        for a in ACTIONS
    }
    supported = [
        e for e in validation if all(train_support[a][cell(e)] >= 3 for a in ACTIONS)
    ]
    r, c = (report["actions"][a]["models"]["ridge"] for a in ACTIONS)
    margin_sum = r["empirical_p90_margin_ms"] + c["empirical_p90_margin_ms"]
    gains = (
        predict(c["model"], supported) - predict(r["model"], supported)
        if supported
        else np.array([])
    )
    report["shadow_decision_diagnostic"] = dict(
        validation_events=len(validation),
        jointly_train_supported_events=len(supported),
        delta_ms=0,
        summed_empirical_margin_ms=margin_sum,
        point_estimate_restore_count=int(np.sum(gains > 0)),
        margin_protected_restore_count=int(np.sum(gains > margin_sum)),
        note="Predictions only, no counterfactual correctness or speedup labels.",
    )
    report["deployment_approved"] = False
    report["limitations"] = [
        "Observed-action prediction errors, not counterfactual policy gains.",
        "Repeated events within a conversation are correlated; "
        "margins are empirical, not guarantees.",
        "State bins are coarse proxies, not proof of exchangeability or full support.",
        "No online policy exported; no TTFT speedup or no-offload comparison.",
    ]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    labels = load_labels(args.source)
    report = analyze(labels)
    report["sources"] = [str(p) for p in args.source]
    report["analysis_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["source_manifest_sha256"] = {
        str(p): hashlib.sha256((p / "manifest.json").read_bytes()).hexdigest()
        for p in args.source
    }
    report["conversation_splits"] = {e["conversation"]: e["split"] for e in labels}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("conversation_splits",)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
