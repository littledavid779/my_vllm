# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decompose frozen-model errors. Offline diagnostics, not policy evaluation."""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from fit_promotion_ttft import ACTIONS, cell, features, fit, load_labels, predict, read


def raw_predict(model, rows):
    return (
        model["intercept"]
        + ((features(rows) - model["center"]) / model["scale"]) @ model["coef"]
    )


def attach_paths(labels, paths):
    result = []
    for e in labels:
        p = paths[(e["arm"], e["request_id"])]
        start, first, end = (
            e["decision_recorded_s"],
            p["first_compute_s"],
            p["first_output_s"],
        )
        if (
            not all(math.isfinite(t) for t in (start, first, end))
            or not start <= first <= end
        ):
            raise ValueError("Decision/schedule/output ordering mismatch")
        before, after = 1000 * (first - start), 1000 * (end - first)
        if not math.isclose(
            before + after, e["decision_to_first_token_ms"], abs_tol=1e-7
        ):
            raise ValueError("Time partition mismatch")
        cached = min(
            e["prompt_tokens"] - 1,
            e["gpu_hit_tokens"]
            + (e["candidate_tokens"] if e["action"] == "restore" else 0),
        )
        if cached != p["first_compute_cached_tokens"]:
            raise ValueError("Actual action/cache mismatch")
        result.append(
            dict(
                e,
                before_schedule_ms=before,
                after_schedule_ms=after,
                completed_prefill_tokens=p["completed_prefill_tokens"],
                prefill_steps=p["prefill_steps"],
            )
        )
    return result


def diagnose(labels, frozen):
    output = []
    support = {
        a: Counter(
            cell(e) for e in labels if e["split"] == "train" and e["action"] == a
        )
        for a in ACTIONS
    }
    models = {a: frozen["actions"][a]["models"]["ridge"]["model"] for a in ACTIONS}
    margins = {
        a: frozen["actions"][a]["models"]["ridge"]["empirical_p90_margin_ms"]
        for a in ACTIONS
    }
    for action in ACTIONS:
        train = [e for e in labels if e["action"] == action and e["split"] == "train"]
        rows = [e for e in labels if e["action"] == action]
        # Reproduce the existing coefficients, not a new hyperparameter search.
        reproduced = fit(train, "ridge")
        for field in ("coef", "center", "scale", "intercept"):
            if not np.allclose(
                reproduced[field], models[action][field], rtol=1e-9, atol=1e-9
            ):
                raise ValueError(
                    "Frozen model cannot be reproduced from training partition"
                )
        stage_models = {
            stage: fit(
                [dict(e, decision_to_first_token_ms=e[stage]) for e in train], "ridge"
            )
            for stage in ("before_schedule_ms", "after_schedule_ms")
        }
        pre = raw_predict(stage_models["before_schedule_ms"], rows)
        post = raw_predict(stage_models["after_schedule_ms"], rows)
        raw_total = raw_predict(models[action], rows)
        if not np.allclose(pre + post, raw_total, rtol=1e-9, atol=1e-9):
            raise ValueError("Linear prediction partition mismatch")
        preds = {a: predict(models[a], rows) for a in ACTIONS}
        for i, e in enumerate(rows):
            predicted = float(preds[action][i])
            error = predicted - e["decision_to_first_token_ms"]
            before_error = float(pre[i] - e["before_schedule_ms"])
            after_error = float(post[i] - e["after_schedule_ms"])
            clip = predicted - float(raw_total[i])
            if not math.isclose(error, before_error + after_error + clip, abs_tol=1e-7):
                raise ValueError("Residual partition mismatch")
            gain = float(preds["recompute"][i] - preds["restore"][i])
            output.append(
                dict(
                    e,
                    predicted_ms=predicted,
                    signed_error_ms=error,
                    absolute_error_ms=abs(error),
                    diagnostic_before_prediction_ms=float(pre[i]),
                    diagnostic_after_prediction_ms=float(post[i]),
                    before_error_ms=before_error,
                    after_error_ms=after_error,
                    clipping_adjustment_ms=clip,
                    observed_action_train_support=support[action][cell(e)],
                    joint_train_support=all(support[a][cell(e)] >= 3 for a in ACTIONS),
                    predicted_restore_ms=float(preds["restore"][i]),
                    predicted_recompute_ms=float(preds["recompute"][i]),
                    predicted_gain_ms=gain,
                    protected_gain_ms=gain - sum(margins.values()),
                    margin_exceeded=abs(error) > margins[action],
                )
            )
    return output


def summarize(rows):
    def avg(key):
        return float(np.mean([e[key] for e in rows]))

    ae = [e["absolute_error_ms"] for e in rows]
    return dict(
        n=len(rows),
        conversations=len({e["conversation"] for e in rows}),
        mean_observed_ms=avg("decision_to_first_token_ms"),
        mae_ms=float(np.mean(ae)),
        p90_absolute_error_ms=float(np.quantile(ae, 0.9)),
        bias_ms=avg("signed_error_ms"),
        before_mean_ms=avg("before_schedule_ms"),
        after_mean_ms=avg("after_schedule_ms"),
        before_mae_ms=float(np.mean([abs(e["before_error_ms"]) for e in rows])),
        after_mae_ms=float(np.mean([abs(e["after_error_ms"]) for e in rows])),
        before_bias_ms=avg("before_error_ms"),
        after_bias_ms=avg("after_error_ms"),
        before_mse=float(np.mean([e["before_error_ms"] ** 2 for e in rows])),
        after_mse=float(np.mean([e["after_error_ms"] ** 2 for e in rows])),
        cross_term=float(
            np.mean([2 * e["before_error_ms"] * e["after_error_ms"] for e in rows])
        ),
        clipping_adjustment_max_ms=max(abs(e["clipping_adjustment_ms"]) for e in rows),
        unsupported=sum(not e["joint_train_support"] for e in rows),
        margin_exceeded=sum(e["margin_exceeded"] for e in rows),
    )


def grouped(rows, fields):
    groups = defaultdict(list)
    for e in rows:
        groups[tuple(e[f] for f in fields)].append(e)
    return [
        dict(zip(fields, key), **summarize(rs)) for key, rs in sorted(groups.items())
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frozen = read(args.model)
    roots = [Path(p) for p in frozen["sources"]]
    labels = load_labels(roots)
    paths, provenance = {}, {}

    def record(path):
        provenance[str(path)] = hashlib.sha256(Path(path).read_bytes()).hexdigest()

    for root in roots:
        record(root / "manifest.json")
        if (
            provenance[str(root / "manifest.json")]
            != frozen["source_manifest_sha256"][str(root)]
        ):
            raise ValueError("Source manifest differs from frozen fit")
        record(root / "summary.json")
        for arm in read(root / "summary.json"):
            p = Path(arm["stats_path"])
            for artifact in (
                p,
                p.with_name("request_summary.csv"),
                root / arm["name"] / "promotion_ttft_labels.json",
            ):
                record(artifact)
            for e in read(p)["stats"]["request_path_trace_events"]:
                key = (str(root / arm["name"]), e["request_id"])
                if key in paths:
                    raise ValueError("Duplicate path")
                paths[key] = e
    if {e["conversation"]: e["split"] for e in labels} != frozen["conversation_splits"]:
        raise ValueError("Conversation partitions changed")
    rows = diagnose(attach_paths(labels, paths), frozen)
    validation = [e for e in rows if e["split"] == "validate"]
    for action in ACTIONS:
        part = [e for e in validation if e["action"] == action]
        old = frozen["actions"][action]["models"]["ridge"]["validation"]
        if len(part) != old["n"] or not math.isclose(
            summarize(part)["mae_ms"], old["mae_ms"], abs_tol=1e-8
        ):
            raise ValueError("Frozen validation metrics changed")
    record(args.model)
    record(Path(__file__))
    record(Path(__file__).with_name("fit_promotion_ttft.py"))
    report = dict(
        validation=grouped(validation, ("action",)),
        by_split=grouped(rows, ("action", "split")),
        by_arm=grouped(validation, ("action", "arm")),
        by_conversation=grouped(validation, ("action", "conversation")),
        by_support=grouped(validation, ("action", "joint_train_support")),
        top_errors={
            a: sorted(
                [e for e in validation if e["action"] == a],
                key=lambda e: e["absolute_error_ms"],
                reverse=True,
            )[:5]
            for a in ACTIONS
        },
        positive_gain_supported=[
            e
            for e in validation
            if e["joint_train_support"] and e["predicted_gain_ms"] > 0
        ],
        provenance_sha256=provenance,
        limitations=[
            "Positive signed error means overprediction; "
            "diagnostic components may be negative.",
            "Stage fits use only training data "
            "and the unchanged linear design/regularization.",
            "Component errors add with clipping adjustment; "
            "absolute errors do not add.",
            "Schedule timestamp is not GPU start; "
            "post-schedule is not pure compute time.",
            "Descriptive attribution, not causal identification "
            "or paired counterfactual gains.",
            "No policy change, no GPU run, "
            "no removal of outliers or hyperparameter tuning.",
        ],
    )
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "summary.json").open("w") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    for name, selected in (
        ("all_events.csv", rows),
        ("validation_events.csv", validation),
    ):
        with (args.output / name).open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(selected[0]))
            writer.writeheader()
            writer.writerows(selected)
    print(json.dumps(report["validation"], indent=2))


if __name__ == "__main__":
    main()
