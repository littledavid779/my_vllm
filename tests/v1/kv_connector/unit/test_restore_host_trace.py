# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from threading import Thread
from types import SimpleNamespace

import pytest

from vllm.v1.kv_offload import restore_host_trace as trace


def schedule():
    return SimpleNamespace(total_num_scheduled_tokens=4, num_scheduled_tokens={"r": 4})


def test_disabled_is_noop(monkeypatch):
    monkeypatch.setattr(trace, "ENABLED", False)
    with trace.execution_scope(schedule()) as record:
        trace.mark("ignored")
    assert record is None


def test_scope_is_thread_local_and_cleans_up(monkeypatch):
    monkeypatch.setattr(trace, "ENABLED", True)
    with (
        pytest.raises(ValueError, match="worker failed"),
        trace.execution_scope(schedule()) as record,
    ):
        trace.mark("runner_setup")
        other = Thread(target=lambda: trace.mark("other_thread"))
        other.start()
        other.join()
        with (
            pytest.raises(RuntimeError, match="Nested"),
            trace.execution_scope(schedule()),
        ):
            pass
        raise ValueError("worker failed")
    assert [m["stage"] for m in record["marks"]] == [
        "executor_dispatch",
        "runner_setup",
        "end",
    ]
    assert record["scheduled_tokens"] == 4
    assert record["scheduled_requests"] == 1
    assert not record["overflow"]
    trace.mark("outside_scope")
    assert len(record["marks"]) == 3
    with trace.execution_scope(schedule()) as next_record:
        pass
    assert len(next_record["marks"]) == 2


def test_marks_are_bounded(monkeypatch):
    monkeypatch.setattr(trace, "ENABLED", True)
    with trace.execution_scope(schedule()) as record:
        for _ in range(110):
            trace.mark("many")
    assert record["overflow"]
    assert len(record["marks"]) == 96
