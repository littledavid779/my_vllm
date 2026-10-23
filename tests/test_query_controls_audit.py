"""Reject incomplete evidence instead of reporting misleading mechanism results."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_query_controls import compare  # noqa: E402
from run_query_controls import audit  # noqa: E402


def fixture(tmp_path, line):
    (tmp_path / "manifest.json").write_text(
        json.dumps({"status": "completed", "config": {"query_control": "query"}})
    )
    (tmp_path / "cacheblend10.log").write_text(line)
    return {
        "prompts": [
            {"version": 1, "length": 2048, "query_length": 19, "signature": "abcdef12"}
        ]
    }


def test_audit_accepts_real_budget_and_full_query(tmp_path):
    coverage = fixture(
        tmp_path,
        "QUERY_CONTROL mode=query total=2048 selected=204 "
        "forced=19 query=19 qcovered=19 tailcovered=19 signature=abcdef12",
    )
    audit(tmp_path, "query", coverage)
    result = json.loads((tmp_path / "selection_audit.json").read_text())
    assert result["expected_measured_calls"] == 1


@pytest.mark.parametrize("bad", ("selected=205", "qcovered=18", "signature=aaaa1111"))
def test_audit_rejects_wrong_budget_partial_query_or_missing_input(tmp_path, bad):
    line = (
        "QUERY_CONTROL mode=query total=2048 selected=204 "
        "forced=19 query=19 qcovered=19 tailcovered=19 signature=abcdef12"
    )
    key = bad.split("=")[0]
    fields = [bad if f.startswith(key + "=") else f for f in line.split()]
    coverage = fixture(tmp_path, " ".join(fields))
    with pytest.raises(ValueError):
        audit(tmp_path, "query", coverage)


def test_analysis_rejects_duplicated_or_unpaired_observations():
    row = {"repeat": 0, "session_id": "one", "version": 1}
    with pytest.raises(ValueError, match="duplicate"):
        compare([row, row], [row])
    with pytest.raises(ValueError, match="unpaired"):
        compare([row], [])
