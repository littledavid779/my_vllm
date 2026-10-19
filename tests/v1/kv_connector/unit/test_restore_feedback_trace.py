# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.v1.kv_offload import restore_trace
from vllm.v1.outputs import AsyncModelRunnerOutput


def output():
    timing = {"worker_observed_s": 0.0}
    return SimpleNamespace(
        kv_connector_output=SimpleNamespace(
            finished_recving={"r"},
            kv_connector_stats=SimpleNamespace(
                data={
                    "CPU_to_GPU": [{"restore_timing": timing}],
                    "GPU_to_CPU": [{"restore_timing": None}],
                }
            ),
        )
    ), timing


class AsyncOutput(AsyncModelRunnerOutput):
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def get_output(self):
        self.calls += 1
        return self.value


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("mode", ["sync", "future", "async", "async_list"])
def test_executor_stamps_before_return(monkeypatch, enabled, mode):
    from vllm.v1.executor import uniproc_executor as mod

    monkeypatch.setattr(restore_trace, "OUTPUT_TRACE_ENABLED", enabled)
    result, timing = output()
    async_result = AsyncOutput(result)
    returned = async_result if mode.startswith("async") else result
    monkeypatch.setattr(mod, "run_method", lambda *a, **kw: returned)
    with ThreadPoolExecutor(max_workers=1) as pool:
        executor = SimpleNamespace(driver_worker=object(), async_output_thread=pool)
        actual = mod.UniProcExecutor.collective_rpc(
            executor,
            "sample_tokens",
            non_block=mode != "sync",
            single_value=mode != "async_list",
        )
        if mode != "sync":
            actual = actual.result(timeout=2)
        if mode == "async_list":
            actual = actual[0]
        assert actual is result
        assert async_result.calls == int(mode.startswith("async"))
    assert ("executor_output_ready_s" in timing) == enabled
    assert "duplicate_output_stamp" not in timing


@pytest.mark.parametrize("mode", ["step", "batch_queue"])
def test_engine_consumed_boundary(monkeypatch, mode):
    from vllm.v1.engine.core import EngineCore

    monkeypatch.setattr(restore_trace, "OUTPUT_TRACE_ENABLED", True)
    result, timing = output()
    restore_trace.stamp_output(result, "executor_output_ready_s")
    future = Future()
    future.set_result(result)
    engine = MagicMock()
    scheduled = SimpleNamespace(total_num_scheduled_tokens=1)
    engine.scheduler.schedule.return_value = scheduled
    engine.model_executor.execute_model.return_value = future
    if mode == "step":
        EngineCore.step(engine)
    else:
        engine.batch_queue_size = 2
        engine.batch_queue = deque([(future, scheduled, future)])
        engine.scheduler.has_requests.return_value = False
        EngineCore.step_with_batch_queue(engine)
    engine.scheduler.update_from_output.assert_called_once_with(scheduled, result)
    assert timing["executor_output_ready_s"] <= timing["engine_output_consumed_s"]
    assert timing["engine_mode"] == mode


def test_stamp_does_not_overwrite_and_skips_no_restore(monkeypatch):
    monkeypatch.setattr(restore_trace, "OUTPUT_TRACE_ENABLED", True)
    result, timing = output()
    restore_trace.stamp_output(result, "stage")
    original = timing["stage"]
    restore_trace.stamp_output(result, "stage")
    assert timing["stage"] == original
    assert timing["duplicate_output_stamp"]
    result.kv_connector_output.finished_recving = set()
    restore_trace.stamp_output(result, "another")
    assert "another" not in timing
    restore_trace.stamp_output(None, "another")
