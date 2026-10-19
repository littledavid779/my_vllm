# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import csv
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def modules(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "benchmarks/multi_turn"))
    monkeypatch.syspath_prepend(str(root / "scripts"))
    return (
        importlib.import_module("benchmark_serving_multi_turn"),
        importlib.import_module("run_lifecycle_correlated_diagnostic"),
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_request_header_and_csv_roundtrip(modules, monkeypatch, tmp_path, enabled):
    bench, runner = modules
    captured = []

    async def fake_send(*args, request_id=None):
        captured.append(request_id)
        return bench.ServerResponse(True, 10, 2, 20, 100, "a", "abc", 2)

    monkeypatch.setattr(bench, "send_request", fake_send)
    monkeypatch.setattr(bench, "get_token_count", lambda tokenizer, text: len(text))
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "qq"},
    ]
    args = bench.RequestArgs("url", "model", True, 1, 3, 60, enabled)
    rows = [
        asyncio.run(
            bench.send_turn(
                None, 0, "conv-a", messages, n, None, args, False, False, True
            )
        )
        for n in (1, 3)
    ]
    assert captured == (["conv-a-turn1", "conv-a-turn3"] if enabled else ["conv-a"] * 2)
    path = tmp_path / "requests.csv"
    bench.write_request_summary_csv(str(path), rows)
    with path.open() as f:
        actual = list(csv.DictReader(f))
    assert [r["external_request_id"] for r in actual] == (
        captured if enabled else [""] * 2
    )
    if enabled:
        _, actions = runner.correlate(actual, {})
        assert len(actions) == 2


def records():
    return [
        dict(
            conversation_id="conv-a",
            input_num_turns=str(n),
            input_num_tokens=str(n * 100),
            client_id="0",
            ttft_ms="20",
            external_request_id=f"conv-a-turn{n}",
        )
        for n in (1, 3)
    ]


def stats():
    events = []
    for target in (16, 32):
        event = dict(
            request_id="chatcmpl-conv-a-turn3-abcdef12", target=target, action="restore"
        )
        events.extend(
            [
                dict(kind="promotion", blocks=1, **event),
                dict(kind="completion", observed_wait_s=0.005, **event),
            ]
        )
    return dict(
        lcuc_events=events,
        lcuc_restore_decisions=2,
        lcuc_restore_observations=2,
        offload_load_request_count=2,
        offload_load_completed_blocks=2,
    )


def test_multiple_promotions_map_to_one_turn(modules):
    _, runner = modules
    _, actions = runner.correlate(records(), stats())
    assert actions[("conv-a", 1)]["action"] == "no_external_restore"
    assert actions[("conv-a", 3)] == dict(
        action="restore", restore_events=2, restore_blocks=2, summed_restore_wait_ms=10
    )
    report, paired = runner.summarize_pair(records(), records(), stats(), 0)
    assert len(paired) == 2
    assert sum(g["n"] for g in report["groups"]["action"]) == 2


@pytest.mark.parametrize("fault", ["missing", "duplicate", "orphan", "lost"])
def test_correlation_fails_closed(modules, fault):
    _, runner = modules
    rows, s = records(), stats()
    if fault == "missing":
        rows[0]["external_request_id"] = ""
    elif fault == "duplicate":
        rows[1]["external_request_id"] = rows[0]["external_request_id"]
    elif fault == "orphan":
        for e in s["lcuc_events"]:
            e["request_id"] = "chatcmpl-conv-a-turn5-abcdef12"
    else:
        s["lcuc_decision_events_dropped"] = 1
    with pytest.raises(ValueError):
        runner.correlate(rows, s)


def test_preserve_owner_semantics(modules):
    from vllm.v1.core.block_pool import BlockPool

    owner = SimpleNamespace(
        _value_request_owner_map={
            "chatcmpl-conv-a-turn1": "chatcmpl-conv-a",
            "chatcmpl-conv-a-turn3": "chatcmpl-conv-a",
        }
    )
    assert (
        BlockPool._value_owner_id(owner, "chatcmpl-conv-a-turn1-abcdef12")
        == "chatcmpl-conv-a"
    )
    assert (
        BlockPool._value_owner_id(owner, "chatcmpl-conv-a-turn3-abcdef13")
        == "chatcmpl-conv-a"
    )


def test_default_request_args_unchanged(modules):
    bench, _ = modules
    assert not bench.RequestArgs("url", "model", True, 1, 3, 60).correlate_requests


def test_correlated_mode_rejects_retries(modules):
    bench, _ = modules
    args = SimpleNamespace(
        num_clients=1,
        max_num_requests=None,
        max_active_conversations=1,
        warmup_step=False,
        skip_first_turn=False,
        seed=0,
        max_turns=None,
        verbose=False,
        print_content=False,
        verify_output=False,
        preserve_dataset_assistant_history=True,
        conversation_sampling=bench.ConversationSampling.ROUND_ROBIN,
        request_rate=0,
        max_retries=1,
        limit_min_tokens=1,
        limit_max_tokens=3,
        request_timeout_sec=60,
        correlate_requests=True,
    )
    with pytest.raises(ValueError, match="max-retries 0"):
        bench.get_client_config(args, {"conv": []})
