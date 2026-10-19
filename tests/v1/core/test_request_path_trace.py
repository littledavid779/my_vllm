# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.v1.core.sched.request_trace import RequestPathTrace


def request(tokens=100):
    return SimpleNamespace(
        request_id="r",
        num_prompt_tokens=tokens,
        num_computed_tokens=0,
        num_preemptions=0,
        resumable=False,
        pooling_params=None,
        has_encoder_inputs=False,
    )


def test_restore_excluded_and_time_partition():
    clock = [0.0]
    trace = RequestPathTrace(clock=lambda: clock[0])
    req = request()
    trace.enqueue(req)
    trace.admission(req, 16, 64)
    # Async restore waits before first actual model computation.
    clock[0] = 0.010
    req.num_computed_tokens = 80
    batch = object()
    trace.schedule(batch, req, 20)
    trace.completed_step("r", trace.take_batch(batch)["r"])
    clock[0] = 0.014
    trace.first_output(req)
    event = trace.snapshot()["request_path_trace_events"][0]
    assert event["initial_gpu_hit_tokens"] == 16
    assert event["initial_external_match_tokens"] == 64
    assert event["completed_prefill_tokens"] == 20
    assert event["queue_to_first_compute_ms"] == pytest.approx(10)
    assert event["first_compute_to_output_ms"] == pytest.approx(4)
    assert event["scheduler_ttft_ms"] == pytest.approx(14)
    trace.finish("r")
    assert not trace.active and not trace.batches


def test_chunked_async_and_decode_excluded():
    trace = RequestPathTrace()
    req = request(64)
    trace.enqueue(req)
    trace.admission(req, 0, 0)
    first, second, decode = object(), object(), object()
    trace.schedule(first, req, 32)
    req.num_computed_tokens = 32
    trace.schedule(second, req, 32)
    req.num_computed_tokens = 64
    trace.schedule(decode, req, 1)
    assert trace.take_batch(decode) == {}
    trace.completed_step("r", trace.take_batch(first)["r"])
    trace.completed_step("r", trace.take_batch(second)["r"])
    trace.first_output(req)
    event = trace.snapshot()["request_path_trace_events"][0]
    assert event["scheduled_prefill_tokens"] == event["completed_prefill_tokens"] == 64
    assert event["prefill_steps"] == 2
    trace.first_output(req)
    assert len(trace.events) == 1


def test_preemption_counts_repeated_prompt_work():
    trace = RequestPathTrace()
    req = request(64)
    trace.enqueue(req)
    trace.admission(req, 0, 0)
    batch = object()
    trace.schedule(batch, req, 32)
    trace.completed_step("r", trace.take_batch(batch)["r"])
    req.num_preemptions = 1
    trace.admission(req, 0, 0)
    trace.schedule(batch, req, 64)
    trace.completed_step("r", trace.take_batch(batch)["r"])
    trace.first_output(req)
    assert trace.events[0]["completed_prefill_tokens"] == 96
    assert trace.events[0]["preemptions"] == 1


def test_abort_overflow_and_reset():
    trace = RequestPathTrace(capacity=1)
    req = request()
    trace.enqueue(req)
    other = request()
    other.request_id = "other"
    trace.enqueue(other)
    trace.finish("r")
    result = trace.snapshot(reset=True)
    assert result["request_path_trace_dropped"] == 1
    assert result["request_path_trace_finished_without_first_token"] == 1
    assert not trace.counters
    trace.enqueue(req)
    trace.snapshot(reset=True)
    assert "r" in trace.active  # live tracking must survive stats reset


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_scheduler_hooks_local_model(monkeypatch, async_scheduling):
    from pathlib import Path

    from tests.v1.core.utils import create_requests, create_scheduler
    from vllm.v1.outputs import ModelRunnerOutput

    model = "/public/model_zoo/Qwen3-0.6B"
    if not Path(model).exists():
        pytest.skip("Local model config unavailable; pure collector tests still run")
    monkeypatch.setenv("VLLM_EXPERIMENTAL_REQUEST_PATH_TRACE", "1")
    scheduler = create_scheduler(
        model=model,
        max_num_batched_tokens=32,
        max_model_len=128,
        async_scheduling=async_scheduling,
    )
    (req,) = create_requests(num_requests=1, num_tokens=64)
    scheduler.add_request(req)
    for tokens in ([], [100]):
        out = scheduler.schedule()
        assert out.num_scheduled_tokens[req.request_id] == 32
        model_out = ModelRunnerOutput(
            req_ids=[req.request_id],
            req_id_to_index={req.request_id: 0},
            sampled_token_ids=[tokens],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )
        scheduler.update_from_output(out, model_out)
    events = scheduler.request_path_trace.snapshot()["request_path_trace_events"]
    assert len(events) == 1
    assert events[0]["completed_prefill_tokens"] == 64
    assert events[0]["initial_gpu_hit_tokens"] == 0
    assert events[0]["first_compute_cached_tokens"] == 0
    assert events[0]["prefill_steps"] == 2


@pytest.mark.parametrize("async_load", [False, True])
def test_scheduler_external_cache_not_counted_as_compute(monkeypatch, async_load):
    from pathlib import Path

    from tests.v1.core.test_scheduler import _step_until_kv_transfer_finished
    from tests.v1.core.utils import create_requests, create_scheduler, mock_kv
    from vllm.v1.outputs import ModelRunnerOutput

    model = "/public/model_zoo/Qwen3-0.6B"
    if not Path(model).exists():
        pytest.skip("Local model config unavailable")
    monkeypatch.setenv("VLLM_EXPERIMENTAL_REQUEST_PATH_TRACE", "1")
    scheduler = create_scheduler(
        model=model,
        max_num_batched_tokens=128,
        enable_prefix_caching=True,
        use_kv_connector=mock_kv(32, async_load),
    )
    (req,) = create_requests(num_requests=1, num_tokens=64)
    scheduler.add_request(req)
    if async_load:
        _step_until_kv_transfer_finished(scheduler, [req.request_id])
    out = scheduler.schedule()
    assert out.num_scheduled_tokens[req.request_id] == 32
    scheduler.update_from_output(
        out,
        ModelRunnerOutput(
            req_ids=[req.request_id],
            req_id_to_index={req.request_id: 0},
            sampled_token_ids=[[100]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    e = scheduler.request_path_trace.events[0]
    assert e["initial_gpu_hit_tokens"] == 0
    assert e["initial_external_match_tokens"] == 32
    assert e["first_compute_cached_tokens"] == 32
    assert e["completed_prefill_tokens"] == 32


def test_engine_stats_exports_trace_without_connector():
    from vllm.v1.engine.core import EngineCore

    trace = RequestPathTrace()
    scheduler = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(
            get_value_policy_stats=lambda reset: {"base": 1}
        ),
        connector=None,
        request_path_trace=trace,
    )
    engine = SimpleNamespace(scheduler=scheduler)
    result = EngineCore.get_value_policy_stats(engine)
    assert result["base"] == 1
    assert result["request_path_trace_enabled"] == 1
    assert result["request_path_trace_events"] == []


def test_trace_default_off_preserves_schedule(monkeypatch):
    from pathlib import Path

    from tests.v1.core.utils import create_requests, create_scheduler

    model = "/public/model_zoo/Qwen3-0.6B"
    if not Path(model).exists():
        pytest.skip("Local model config unavailable")
    outputs = []
    for flag in ("0", "1"):
        monkeypatch.setenv("VLLM_EXPERIMENTAL_REQUEST_PATH_TRACE", flag)
        scheduler = create_scheduler(model=model, max_num_batched_tokens=128)
        for req in create_requests(num_requests=2, num_tokens=64):
            scheduler.add_request(req)
        outputs.append(scheduler.schedule().num_scheduled_tokens)
        assert (scheduler.request_path_trace is None) == (flag == "0")
    assert outputs[0] == outputs[1]
