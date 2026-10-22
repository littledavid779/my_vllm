"""Repeated runs must not masquerade as additional independent task sources."""

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
from analyze_strict_comparison import cluster_intervals  # noqa: E402


def test_repeating_measurements_preserves_source_cluster_uncertainty():
    pairs = []
    for index in range(8):
        for _ in range(3):
            strict = {
                "source_id": str(index),
                "task_correct": index > 2,
                "client_ttft_ms": 100.0 + index,
            }
            block = {
                "source_id": str(index),
                "task_correct": index < 4,
                "client_ttft_ms": 99.0 + index,
            }
            pairs.append((strict, block))
    once = cluster_intervals(pairs, draws=100)
    repeated = cluster_intervals(pairs * 2, draws=100)
    assert once["quality_gain"] == repeated["quality_gain"]
    assert once["mean_ttft_change"] == repeated["mean_ttft_change"]


def test_paired_quality_gain_is_distinct_from_latency_change():
    pairs = [
        (
            {"source_id": str(i), "task_correct": True, "client_ttft_ms": 100.0},
            {"source_id": str(i), "task_correct": False, "client_ttft_ms": 100.0},
        )
        for i in range(4)
    ]
    result = cluster_intervals(pairs, draws=100)
    assert result["quality_gain"] == [1, 1]
    assert result["mean_ttft_change"] == [0, 0]
