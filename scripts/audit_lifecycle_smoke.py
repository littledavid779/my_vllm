# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read-only completeness, resource, and request-set audit of lifecycle smoke."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def read(path):
    return path.read_text()


def audit(root):
    manifest = json.loads(read(root / "manifest.json"))
    records = []
    signatures = {}
    for phase in ("calibration", "evaluation"):
        for path in sorted((root / phase).glob("*/metrics.json")):
            metrics = json.loads(read(path))
            stats_path = Path(metrics["stats_path"])
            stats = json.loads(read(stats_path))["stats"]
            with stats_path.with_name("request_summary.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            assert len(rows) == 128
            assert all(float(row["output_num_tokens"]) > 0 for row in rows)
            samples = json.loads(read(path.with_name("gpu_samples.json")))
            assert not samples["foreign_pids"]
            for key in (
                "offload_cost_events_dropped",
                "offload_cost_event_unmatched_transfers",
                "lcuc_pending",
                "lcuc_incomplete_promotions",
            ):
                assert not stats.get(key, 0), (metrics["name"], key)
            cold_blocks = stats.get("lcuc_cold_probe_blocks_spent", 0)
            cold_cost = stats.get("lcuc_cold_probe_cost_spent_s", 0)
            budget = manifest.get("cold_probe_config", {})
            assert cold_blocks <= budget.get("cold_probe_budget_blocks", 128)
            assert cold_cost <= budget.get("cold_probe_budget_s", 0.1) + 1e-12
            record = {
                **metrics,
                "foreign_pids": samples["foreign_pids"],
                "cold_probe_blocks": cold_blocks,
                "cold_probe_estimated_cost_ms": cold_cost * 1000,
                "admission_reasons": dict(
                    Counter(
                        e.get("reason", "unknown")
                        for e in stats.get("lcuc_events", [])
                        if e.get("kind") == "admission"
                    )
                ),
            }
            records.append(record)
            if phase == "evaluation":
                signatures[metrics["name"]] = Counter(
                    (r["conversation_id"], r["input_num_turns"], r["input_num_tokens"])
                    for r in rows
                )
    assert len(records) == 4 + len(manifest["evaluation_order"])
    baseline = signatures["evaluation/no-offload"]
    for record in records:
        if record["name"] in signatures:
            sig = signatures[record["name"]]
            record["request_signature_diff_vs_no_offload"] = {
                "extra": sum((sig - baseline).values()),
                "missing": sum((baseline - sig).values()),
            }
    repo = Path(__file__).resolve().parents[1]
    for filename, digest in manifest["code_sha256"].items():
        assert hashlib.sha256((repo / filename).read_bytes()).hexdigest() == digest
    return {
        "checks_passed": True,
        "scope": "completeness, sampled GPU isolation, frozen code, probe budget",
        "request_signature_note": "Not exact token or arrival-order equivalence",
        "arms": records,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.root), ensure_ascii=False, indent=2))
