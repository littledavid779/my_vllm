# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
from pathlib import Path

import pytest


@pytest.fixture
def mod(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    return importlib.import_module("analyze_lifecycle_request_ttft")


def row(turn=1, tokens=100, ttft=10):
    return dict(
        conversation_id="conv-with-hyphen",
        input_num_turns=str(turn),
        input_num_tokens=str(tokens),
        client_id="0",
        ttft_ms=str(ttft),
    )


def stats():
    event = dict(
        request_id="chatcmpl-conv-with-hyphen-1234abcd", target=16, action="restore"
    )
    return dict(
        lcuc_events=[
            dict(kind="promotion", blocks=1, **event),
            dict(kind="completion", observed_wait_s=0.01, **event),
        ],
        lcuc_restore_decisions=1,
        lcuc_restore_observations=1,
        offload_load_request_count=1,
        offload_load_completed_blocks=1,
    )


def test_duplicate_and_invalid_ttft(mod):
    with pytest.raises(ValueError, match="Duplicate"):
        mod.index_rows([row(), row()])
    with pytest.raises(ValueError, match="Invalid"):
        mod.index_rows([row(ttft=float("nan"))])


def test_fail_closed_join(mod):
    base = mod.index_rows([row()])
    for candidate in ([row(turn=3)], [row(tokens=101)]):
        with pytest.raises(ValueError, match="mismatch"):
            mod.compare(base, mod.index_rows(candidate), {}, 0, "no-offload")


def test_no_inferred_turn_mapping(mod):
    restored = mod.restore_audit(stats(), {"conv-with-hyphen"})
    base = mod.index_rows([row(), row(turn=3)])
    rows = mod.compare(base, base, restored, 0, "no-offload")
    assert {r["restore_group"] for r in rows} == {"restore_conversation_turn_unknown"}
    assert [r["turn_group"] for r in rows] == ["first", "later"]
    assert mod.restore_audit({}, {"conv-with-hyphen"}) == {}


@pytest.mark.parametrize("mutation", ["missing", "counter", "blocks", "id", "action"])
def test_incomplete_logs_rejected(mod, mutation):
    s = stats()
    if mutation == "missing":
        s["lcuc_events"].pop()
    elif mutation == "counter":
        s["offload_load_request_count"] = 2
    elif mutation == "blocks":
        s["offload_load_completed_blocks"] = 2
    elif mutation == "id":
        for e in s["lcuc_events"]:
            e["request_id"] = "chatcmpl-other-1234abcd"
    else:
        s["lcuc_events"][0]["action"] = "recompute"
    with pytest.raises(ValueError):
        mod.restore_audit(s, {"conv-with-hyphen"})


def test_mean_contributions_sum(mod):
    base = mod.index_rows([row(), row(turn=3)])
    candidate = mod.index_rows([row(ttft=12), row(turn=3, ttft=16)])
    rows = mod.compare(base, candidate, {}, 0, "no-offload")
    groups = mod.group_stats(rows, "turn_group")
    assert sum(g["contribution_to_overall_mean_ms"] for g in groups) == 4
    assert {r["restore_group"] for r in rows} == {"conversation_without_restore"}


@pytest.mark.parametrize(
    "counter",
    [
        "lcuc_decision_events_dropped",
        "lcuc_incomplete_promotions",
        "lcuc_pending",
    ],
)
def test_lost_or_incomplete_events_rejected(mod, counter):
    s = stats()
    s[counter] = 1
    with pytest.raises(ValueError, match="Cannot classify"):
        mod.restore_audit(s, {"conv-with-hyphen"})


@pytest.mark.parametrize(
    "tokens,expected",
    [(127, "lt128"), (128, "128-511"), (512, "512-1023"), (1024, "ge1024")],
)
def test_length_boundaries(mod, tokens, expected):
    assert mod.length_bucket(tokens) == expected
