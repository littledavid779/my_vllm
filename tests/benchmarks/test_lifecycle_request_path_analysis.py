# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
from pathlib import Path

import pytest


@pytest.fixture
def mod(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    return importlib.import_module("analyze_lifecycle_request_path")


def rows():
    return [
        dict(
            conversation_id="conv",
            input_num_turns="3",
            external_request_id="conv-turn3",
        )
    ]


def stats(cached=16, queue=10, post=5):
    return dict(
        request_path_trace_enabled=1,
        request_path_trace_events=[
            dict(
                request_id="chatcmpl-conv-turn3-abcdef12",
                prompt_tokens=100,
                enqueued_s=0,
                first_compute_s=queue / 1000,
                first_output_s=(queue + post) / 1000,
                queue_to_first_compute_ms=queue,
                first_compute_to_output_ms=post,
                scheduler_ttft_ms=queue + post,
                initial_gpu_hit_tokens=16,
                initial_external_match_tokens=cached - 16,
                first_compute_cached_tokens=cached,
                scheduled_prefill_tokens=100 - cached,
                completed_prefill_tokens=100 - cached,
                preemptions=0,
            )
        ],
    )


def test_path_differences_decompose_only_time(mod):
    paired = [
        dict(conversation_id="conv", message_count=3, delta_ms=6, action="restore")
    ]
    report = mod.attach_path(paired, rows(), stats(), rows(), stats(80, 14, 3))
    r = paired[0]
    assert r["delta_queue_to_first_compute_ms"] == 4
    assert r["delta_first_compute_to_output_ms"] == -2
    assert r["delta_outside_scheduler_ms"] == 4
    assert r["delta_completed_prefill_tokens"] == -64
    assert r["work_group"] == "less_work"
    assert report["action_work"][0]["group"] == "restore/less_work"


@pytest.mark.parametrize(
    "error",
    [
        "missing",
        "duplicate",
        "id",
        "nan",
        "negative",
        "time",
        "count",
        "balance",
        "pending",
        "dropped",
    ],
)
def test_reject_bad_path_records(mod, error):
    s = stats()
    e = s["request_path_trace_events"][0]
    if error == "missing":
        s["request_path_trace_events"] = []
    elif error == "duplicate":
        s["request_path_trace_events"].append(e.copy())
    elif error == "id":
        e["request_id"] = "chatcmpl-conv-turn3-not-a-valid-suffix"
    elif error == "nan":
        e["initial_gpu_hit_tokens"] = float("nan")
    elif error == "negative":
        e["queue_to_first_compute_ms"] = -1
    elif error == "time":
        e["scheduler_ttft_ms"] = 99
    elif error == "count":
        e["scheduled_prefill_tokens"] += 1
    elif error == "balance":
        e["first_compute_cached_tokens"] += 1
    elif error == "pending":
        s["request_path_trace_active"] = 1
    else:
        s["request_path_trace_dropped"] = 1
    with pytest.raises(ValueError):
        mod.validate_path(rows(), s)


def test_prompt_lengths_must_match(mod):
    s = stats()
    e = s["request_path_trace_events"][0]
    e["prompt_tokens"] += 1
    e["completed_prefill_tokens"] += 1
    e["scheduled_prefill_tokens"] += 1
    with pytest.raises(ValueError, match="Templated"):
        mod.attach_path(
            [dict(conversation_id="conv", message_count=3)], rows(), stats(), rows(), s
        )
