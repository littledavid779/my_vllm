"""Ensure paired quality and source dependence cannot be confused."""

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
from analyze_round2 import clustered_gap, summarize  # noqa: E402


def test_quality_errors_and_gains_are_separate_from_fidelity():
    rows, refs = [], {}
    for index, (full_ok, approx_ok) in enumerate(
        [(True, False), (False, True), (True, True), (False, False)]
    ):
        refs[str(index), 1] = {"task_correct": full_ok, "generated_text": "reference"}
        rows.append(
            {
                "session_id": str(index),
                "source_id": str(index),
                "version": 1,
                "arm": "cacheblend5",
                "placement": "natural4k",
                "family": "qa2",
                "task_correct": approx_ok,
                "generated_text": "different",
                "client_ttft_ms": 20.0,
                "main_request_ttft_ms": 15.0,
                "new_chunk_prepare_ms": 5.0,
            }
        )
    result = summarize(rows, refs, "natural4k", "all", "cacheblend5")
    assert result["task_accuracy"] == 0.5
    assert result["cache_induced_errors"] == 1
    assert result["correct_when_reference_wrong"] == 1
    assert result["output_disagreements"] == 4
    assert result["conditional_error_rate"] == 0.5


def test_repeating_edits_does_not_create_more_independent_sources():
    refs = {(str(i), 1): {"task_correct": True} for i in range(4)}
    rows = [
        {"source_id": str(i), "session_id": str(i), "version": 1, "task_correct": i > 1}
        for i in range(4)
    ]
    assert clustered_gap(rows, refs) == clustered_gap(rows * 10, refs)
