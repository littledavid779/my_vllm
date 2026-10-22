"""Check causality/position invariants before spending GPU time."""

import hashlib
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
from public_oracle import babi_oracle  # noqa: E402


def test_round2_equal_length_edits_and_oracle():
    path = PROJECT / "data/round2_screen.json"
    data = json.loads(path.read_text())
    lock = json.loads(path.with_suffix(".lock.json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == lock["input_sha256"]
    assert len({s["source_id"] for s in data["sessions"]}) == 108
    assert len(data["sessions"]) == 216
    separator = data["pilot_provenance"]["separator_ids"]
    for session in data["sessions"]:
        assert session["metadata"]["source_index"] >= 2
        begin, end = session["editable_token_span"]
        cut_a, cut_c = begin, end + 2 * len(separator)
        versions = session["versions"]
        assert len({tuple(v["token_ids"]) for v in versions}) == 4
        assert len({len(v["token_ids"]) for v in versions}) == 1
        assert versions[0]["expected_value"] != versions[1]["expected_value"]
        for version in versions:
            ids = version["token_ids"]
            assert ids[:cut_a] == versions[0]["token_ids"][:cut_a]
            assert ids[cut_c:] == versions[0]["token_ids"][cut_c:]
            assert ids[cut_a : cut_a + len(separator)] == separator
            assert ids[cut_c - len(separator) : cut_c] == separator
            facts = list(session["metadata"]["original_facts"])
            facts[session["metadata"]["edited_fact_index"]] = version["definition"]
            assert (
                babi_oracle(
                    facts,
                    session["metadata"]["question"],
                    session["family"].removeprefix("babilong_"),
                )
                == version["expected_value"]
            )
        if session["placement"] == "early8k":
            assert session["suffix_tokens"] >= 7600
            assert begin / session["prompt_tokens"] < 0.2


def test_oracle_keeps_dropped_object_and_negation_semantics():
    facts = [
        "Mary went to the kitchen.",
        "Mary got the apple.",
        "Mary dropped the apple.",
        "Mary went to the garden.",
    ]
    assert babi_oracle(facts, "Where is the apple?", "qa2") == "kitchen"
    assert (
        babi_oracle(
            ["Mary is no longer in the kitchen."], "Is Mary in the kitchen?", "qa9"
        )
        == "no"
    )
    assert (
        babi_oracle(
            ["Mary is no longer in the kitchen."], "Is Mary in the office?", "qa9"
        )
        == "unknown"
    )
