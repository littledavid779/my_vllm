# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import deque
from concurrent.futures import Future
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.v1.kv_offload.restore_ready import ReadyRestoreDrain, RestoreReadinessTrace


def ready(restore=True):
    f = Future()
    f.set_result(
        SimpleNamespace(
            kv_connector_output=SimpleNamespace(
                finished_recving={"r"} if restore else set()
            )
        )
    )
    return f


@pytest.mark.parametrize("mode,expected", [("observe", False), ("on", True)])
def test_gate_only_oldest_nonblocking(mode, expected):
    gate = ReadyRestoreDrain(mode)
    pending = Future()
    q = deque([(ready(), None, None), (pending, None, None)])
    assert not gate.should_drain(q)
    assert len(q) == 2
    pending.set_result(SimpleNamespace(kv_connector_output=None))
    assert not gate.should_drain(q)
    q.pop()
    assert gate.should_drain(q) is expected
    assert len(q) == 1
    assert gate.counts["eligible"] == 1
    assert gate.counts["consumed"] == 0


def test_failed_cancelled_and_empty():
    g = ReadyRestoreDrain("on")
    assert not g.should_drain(deque())
    failed, cancelled = Future(), Future()
    failed.set_exception(RuntimeError("failed"))
    cancelled.cancel()
    for f in (failed, cancelled):
        assert not g.should_drain(deque([(f, None, None)]))
    assert g.snapshot(reset=True)["restore_ready_first_failed"] == 2
    assert not g.counts
    with pytest.raises(ValueError):
        ReadyRestoreDrain("invalid")


def engine(mode, restore=True):
    from vllm.v1.engine.core import EngineCore

    e = EngineCore.__new__(EngineCore)
    old = ready(restore)
    old_schedule = object()
    e.batch_queue = deque([(old, old_schedule, old)])
    e.batch_queue_size = 2
    e.restore_ready_first = None if mode == "off" else ReadyRestoreDrain(mode)
    e.scheduler = MagicMock()
    e.scheduler.has_requests.return_value = True
    e.scheduler.schedule.return_value = SimpleNamespace(
        total_num_scheduled_tokens=1, pending_structured_output_tokens=False
    )
    e.model_executor = MagicMock()
    e.model_executor.execute_model.return_value = ready(False)
    e.model_executor.sample_tokens.return_value = ready(False)
    e.log_error_detail = lambda *a: nullcontext()
    e.log_iteration_details = lambda *a: nullcontext()
    e._process_aborts_queue = MagicMock()
    e.is_ec_consumer = True
    e.is_pooling_model = False
    e.use_spec_decode = False
    return e, old, old_schedule


@pytest.mark.parametrize(
    "mode,restore", [("off", True), ("observe", True), ("on", True), ("on", False)]
)
def test_engine_reuses_fifo_consumer_and_abort_order(mode, restore):
    from vllm.v1.engine.core import EngineCore

    e, old, scheduled = engine(mode, restore)
    sequence = []
    e._process_aborts_queue.side_effect = lambda: sequence.append("abort")
    e.scheduler.update_from_output.side_effect = lambda *a: sequence.append("update")
    _, executed = EngineCore.step_with_batch_queue(e)
    drain = mode == "on" and restore
    assert executed is (not drain)
    assert e.scheduler.schedule.call_count == int(not drain)
    assert e.model_executor.execute_model.call_count == int(not drain)
    assert sequence == ["abort", "update"]
    e.scheduler.update_from_output.assert_called_once_with(scheduled, old.result())
    assert len(e.batch_queue) == int(not drain)
    if drain:
        assert e.restore_ready_first.counts["consumed"] == 1
        assert e._restore_ready_consumed_this_step
        EngineCore.step_with_batch_queue(e)
        assert e.restore_ready_first.counts["consumed"] == 1


def test_failed_future_still_propagates():
    from vllm.v1.engine.core import EngineCore

    e, _, _ = engine("on")
    failed = Future()
    failed.set_exception(RuntimeError("worker failed"))
    e.batch_queue = deque([(failed, object(), failed)])
    e.scheduler.has_requests.return_value = False
    with pytest.raises(RuntimeError, match="worker failed"):
        EngineCore.step_with_batch_queue(e)
    e.scheduler.update_from_output.assert_not_called()


@pytest.mark.parametrize("progress", [False, True])
def test_progress_does_not_add_idle_sleep(monkeypatch, progress):
    from vllm.v1.engine import core

    e = MagicMock()

    def step():
        e._restore_ready_consumed_this_step = progress
        return {}, False

    e.step_fn.side_effect = step
    e.scheduler.has_unfinished_requests.return_value = True
    sleep = MagicMock()
    monkeypatch.setattr(core.time, "sleep", sleep)
    assert core.EngineCoreProc._process_engine_step(e) is False
    assert sleep.call_count == int(not progress)
    e.post_step.assert_called_once_with(False)


@pytest.mark.parametrize("completion", ["schedule", "execute", "wait"])
def test_trace_old_future_becomes_ready_at_different_boundaries(completion):
    from vllm.v1.engine.core import EngineCore

    e, _, _ = engine("observe")
    pending = Future()
    old_schedule = object()
    e.batch_queue = deque([(pending, old_schedule, pending)])
    trace = e.restore_readiness_trace = RestoreReadinessTrace()
    # Register a Future submitted in the previous iteration.
    trace.begin(deque())
    trace.enqueued(pending)
    output = ready().result()
    scheduled = e.scheduler.schedule.return_value
    next_output = e.model_executor.execute_model.return_value

    def schedule():
        if completion == "schedule":
            pending.set_result(output)
        return scheduled

    def execute(*args, **kwargs):
        if completion == "execute":
            pending.set_result(output)
        return next_output

    original_result = pending.result
    calls = []

    def result(*args, **kwargs):
        calls.append(1)
        if completion == "wait":
            pending.set_result(output)
        return original_result(*args, **kwargs)

    pending.result = result
    e.scheduler.schedule.side_effect = schedule
    e.model_executor.execute_model.side_effect = execute
    EngineCore.step_with_batch_queue(e)
    event = trace.events[0]
    assert event["created_loop"] < event["consumed_loop"]
    first_done = next(p for p in event["observations"] if p["done"])
    assert (
        first_done["stage"]
        == {
            "schedule": "after_schedule",
            "execute": "after_execute",
            "wait": "after_result",
        }[completion]
    )
    assert event["request_ids"] == ["r"]
    assert calls == [1]  # Trace never adds a Future.result() call.
    e.scheduler.update_from_output.assert_called_once_with(old_schedule, output)
    assert e.restore_ready_first.counts["eligible"] == 0
    assert trace.counts["head_mismatch"] == 0


def test_trace_new_future_same_loop_and_non_restore_cleanup():
    from vllm.v1.engine.core import EngineCore

    e, _, _ = engine("observe")
    e.batch_queue.clear()
    trace = e.restore_readiness_trace = RestoreReadinessTrace()
    e.model_executor.sample_tokens.return_value = ready()
    EngineCore.step_with_batch_queue(e)
    event = trace.events[0]
    assert event["created_loop"] == event["consumed_loop"]
    assert event["observations"][0]["stage"] == "after_enqueue"
    assert all(p["done"] is None for p in event["submission"])
    assert not trace.active
    e.model_executor.sample_tokens.return_value = ready(False)
    EngineCore.step_with_batch_queue(e)
    assert len(trace.events) == 1
    assert not trace.active
    assert trace.counts["consumed_batches"] == 2


def test_trace_bounded_events_and_reset():
    trace = RestoreReadinessTrace(capacity=1)
    for _ in range(2):
        future = ready()
        trace.begin(deque())
        trace.enqueued(future)
        trace.before_result(future)
        trace.finish(future, future.result())
    stats = trace.snapshot(reset=True)
    assert len(stats["restore_readiness_trace_events"]) == 1
    assert stats["restore_readiness_trace_dropped"] == 1
    assert not trace.events
    assert not trace.active


def test_host_call_attached_without_changing_execution(monkeypatch):
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.kv_offload import restore_host_trace

    monkeypatch.setattr(restore_host_trace, "ENABLED", True)
    e, old, scheduled = engine("observe")
    e.scheduler.schedule.return_value.num_scheduled_tokens = {"new": 1}
    trace = e.restore_readiness_trace = RestoreReadinessTrace()
    trace.begin(deque())
    trace.enqueued(old)
    result = e.model_executor.execute_model.return_value

    def execute(*args, **kwargs):
        restore_host_trace.mark("runner_setup")
        return result

    e.model_executor.execute_model.side_effect = execute
    EngineCore.step_with_batch_queue(e)
    host = next(
        p for p in trace.events[0]["consumption"] if p["stage"] == "after_execute"
    )["host_call"]
    assert [m["stage"] for m in host["marks"]] == [
        "executor_dispatch",
        "runner_setup",
        "end",
    ]
    e.model_executor.execute_model.assert_called_once()
    e.scheduler.update_from_output.assert_called_once_with(scheduled, old.result())
    assert e.restore_ready_first.counts["consumed"] == 0
