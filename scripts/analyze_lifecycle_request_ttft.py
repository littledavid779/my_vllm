#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Descriptive request comparisons; never infer a turn from a restore event."""

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ARMS = ("no-offload", "all-block-2g", "selective-2g")


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def index_rows(rows):
    indexed = {}
    for row in rows:
        key = (row["conversation_id"], int(row["input_num_turns"]))
        if key in indexed:
            raise ValueError(f"Duplicate conversation/message-count key: {key}")
        if not math.isfinite(float(row["ttft_ms"])) or float(row["ttft_ms"]) < 0:
            raise ValueError("Invalid TTFT")
        indexed[key] = row
    return indexed


def restore_audit(stats, conversations):
    events = stats.get("lcuc_events", [])
    promotions = [e for e in events if e["kind"] == "promotion"]
    completions = [e for e in events if e["kind"] == "completion"]
    if len(promotions) != stats.get("lcuc_restore_decisions", 0):
        raise ValueError("Promotion log/counter mismatch")
    if len(completions) != stats.get("lcuc_restore_observations", 0):
        raise ValueError("Completion log/counter mismatch")
    if len(promotions) != stats.get("offload_load_request_count", 0):
        raise ValueError("Load log/counter mismatch")
    for name in (
        "lcuc_decision_events_dropped",
        "lcuc_incomplete_promotions",
        "lcuc_pending",
        "lcuc_recompute_decisions",
    ):
        if stats.get(name, 0):
            raise ValueError(f"Cannot classify with {name}")

    def signatures(es):
        return Counter((e["request_id"], e["target"]) for e in es)

    if signatures(promotions) != signatures(completions):
        raise ValueError("Promotion/completion mismatch")
    if any(e.get("action") != "restore" for e in promotions + completions):
        raise ValueError("Expected completed restore-only experiment")
    if sum(e["blocks"] for e in promotions) != stats.get(
        "offload_load_completed_blocks", 0
    ):
        raise ValueError("Restored block count mismatch")
    result = defaultdict(list)
    for event in completions:
        match = re.fullmatch(r"chatcmpl-(.+)-([0-9a-f]{8})", event["request_id"])
        if match is None or match[1] not in conversations:
            raise ValueError(f"Unmapped conversation: {event['request_id']}")
        result[match[1]].append(event)
    return result


def length_bucket(tokens):
    for limit in (128, 512, 1024):
        if tokens < limit:
            return {128: "lt128", 512: "128-511", 1024: "512-1023"}[limit]
    return "ge1024"


def compare(base, candidate, restored_conversations, repeat, baseline):
    if base.keys() != candidate.keys():
        raise ValueError("Request set mismatch")
    out = []
    for key, b in base.items():
        c = candidate[key]
        for name in ("input_num_tokens", "client_id"):
            if b[name] != c[name]:
                raise ValueError(f"Request input/assignment mismatch: {key}")
        bt, ct = float(b["ttft_ms"]), float(c["ttft_ms"])
        out.append(
            dict(
                repeat=repeat,
                baseline=baseline,
                conversation_id=key[0],
                message_count=key[1],
                input_tokens=int(c["input_num_tokens"]),
                baseline_ttft_ms=bt,
                selective_ttft_ms=ct,
                delta_ms=ct - bt,
                turn_group="first" if key[1] == 1 else "later",
                length_group=length_bucket(int(c["input_num_tokens"])),
                restore_group=(
                    "restore_conversation_turn_unknown"
                    if key[0] in restored_conversations
                    else "conversation_without_restore"
                ),
            )
        )
    return out


def group_stats(rows, dimension):
    groups = defaultdict(list)
    for row in rows:
        groups[row[dimension]].append(row)
    return [
        dict(
            group=key,
            n=len(rs),
            baseline_mean_ms=statistics.mean(r["baseline_ttft_ms"] for r in rs),
            selective_mean_ms=statistics.mean(r["selective_ttft_ms"] for r in rs),
            delta_mean_ms=statistics.mean(r["delta_ms"] for r in rs),
            contribution_to_overall_mean_ms=sum(r["delta_ms"] for r in rs) / len(rows),
            worse=sum(r["delta_ms"] > 0 for r in rs),
        )
        for key, rs in sorted(groups.items())
    ]


def analyze(root):
    manifest = read(root / "manifest.json")
    if manifest["suite"] != "ttft-confirmation-2g":
        raise ValueError("Expected confirmation suite")
    if digest(root / "evaluation.json") != manifest["input_sha256"]:
        raise ValueError("Evaluation hash mismatch")
    all_rows, audits, tails, hashes = [], [], [], {}
    for repeat, order in enumerate(manifest["orders"]):
        if set(order) != set(ARMS) or len(order) != 3:
            raise ValueError("Invalid run order")
        cells, restores = {}, {}
        for arm in ARMS:
            paths = list(
                (root / f"repeat_{repeat}" / arm).glob(
                    "*/seed_0/lru/request_summary.csv"
                )
            )
            if len(paths) != 1:
                raise ValueError(f"Missing/ambiguous CSV: {repeat}/{arm}")
            path = paths[0]
            with path.open() as f:
                cells[arm] = index_rows(list(csv.DictReader(f)))
            if len(cells[arm]) != manifest["requests_per_arm"]:
                raise ValueError("Request count mismatch")
            sp = path.with_name("value_policy_stats.json")
            stats = read(sp)["stats"]
            restores[arm] = restore_audit(stats, {k[0] for k in cells[arm]})
            for source in (path, sp):
                hashes[str(source.relative_to(root))] = digest(source)
            observed = [
                e["observed_wait_s"] * 1000 for es in restores[arm].values() for e in es
            ]
            audits.append(
                dict(
                    repeat=repeat,
                    arm=arm,
                    requests=len(cells[arm]),
                    restore_events=len(observed),
                    conversations_with_restore=len(restores[arm]),
                    mean_observed_restore_wait_ms=(
                        statistics.mean(observed) if observed else None
                    ),
                    exact_restore_turn_mapping_available=False,
                )
            )
        for baseline in ARMS[:2]:
            paired = compare(
                cells[baseline],
                cells["selective-2g"],
                restores["selective-2g"],
                repeat,
                baseline,
            )
            all_rows.extend(paired)
            # Describe each run's own upper 5%; do not decompose P95 additively.
            for field in ("baseline_ttft_ms", "selective_ttft_ms", "delta_ms"):
                count = math.ceil(len(paired) * 0.05) if field != "delta_ms" else 20
                tail = sorted(paired, key=lambda r: r[field], reverse=True)[:count]
                tails.append(
                    dict(
                        repeat=repeat,
                        baseline=baseline,
                        selection=field,
                        n=len(tail),
                        mean_delta_ms=statistics.mean(r["delta_ms"] for r in tail),
                        first_requests=sum(r["turn_group"] == "first" for r in tail),
                        long_requests=sum(r["input_tokens"] >= 1024 for r in tail),
                        no_restore_conversation_requests=sum(
                            r["restore_group"] == "conversation_without_restore"
                            for r in tail
                        ),
                        requests=tail,
                    )
                )
    groups = []
    for baseline in ARMS[:2]:
        for dimension in ("turn_group", "length_group", "restore_group"):
            rs = [r for r in all_rows if r["baseline"] == baseline]
            for group in group_stats(rs, dimension):
                per_round = []
                for repeat in range(len(manifest["orders"])):
                    subset = [
                        r
                        for r in rs
                        if r["repeat"] == repeat and r[dimension] == group["group"]
                    ]
                    per_round.append(
                        statistics.mean(r["delta_ms"] for r in subset)
                        if subset
                        else None
                    )
                groups.append(
                    dict(
                        baseline=baseline,
                        dimension=dimension,
                        **group,
                        per_round_delta_ms=per_round,
                    )
                )
    return dict(
        source=str(root),
        source_sha256=hashes,
        run_audit=audits,
        grouped_comparisons=groups,
        tails=tails,
    ), all_rows


def markdown(result):
    lines = [
        "# Request-level descriptive TTFT analysis",
        "",
        "Positive delta = selective slower. Counts include repeated runs.",
        "Same conversation/message count is not a matched runtime state.",
        "Restore grouping is conversation-level, NOT per-request action.",
        "Contributions decompose mean TTFT only, never P95 or causal cost.",
        "",
        "| Baseline | Dimension | Group | N | Delta ms | Mean contribution ms |",
        "|---|---|---|---:|---:|---:|",
    ]
    for g in result["grouped_comparisons"]:
        lines.append(
            f"| {g['baseline']} | {g['dimension']} | {g['group']} | "
            f"{g['n']} | {g['delta_mean_ms']:.3f} | "
            f"{g['contribution_to_overall_mean_ms']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result, rows = analyze(args.source)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "summary.md").write_text(markdown(result))
    with (args.output / "paired_requests.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(markdown(result))


if __name__ == "__main__":
    main()
