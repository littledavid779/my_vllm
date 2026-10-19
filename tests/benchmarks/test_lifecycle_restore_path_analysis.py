# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
from pathlib import Path

import pytest


@pytest.fixture
def mod(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    return importlib.import_module("analyze_lifecycle_restore_path")


def sample():
    return dict(
        restore_path_trace_enabled=1,
        request_path_trace_events=[
            dict(
                request_id="r",
                enqueued_s=0.0,
                first_compute_s=0.009,
                queue_to_first_compute_ms=9,
                initial_external_match_tokens=16,
                preemptions=0,
            )
        ],
        restore_path_trace_events=[
            dict(
                request_id="r",
                host="local",
                decision_s=0.001,
                allocated_s=0.002,
                metadata_built_s=0.003,
                scheduler_observed_s=0.008,
                workers=[
                    dict(
                        host="local",
                        worker_received_s=0.004,
                        submit_start_s=0.005,
                        submit_return_s=0.006,
                        worker_observed_s=0.007,
                        dma_elapsed_s=0.0005,
                        bytes=1024,
                    )
                ],
            )
        ],
    )


def test_partition_excludes_overlapping_dma(mod):
    result = mod.analyze_restore_path(sample())
    assert result["count"] == 1
    assert result["means"]["dma_elapsed_ms"] == 0.5
    assert sum(result["means"][f] for _, _, f in mod.STAGES) == pytest.approx(9)


def test_feedback_partition_and_legacy(mod):
    s = sample()
    assert mod.analyze_restore_path(s)["feedback_count"] == 0
    with pytest.raises(ValueError, match="Incomplete"):
        mod.analyze_restore_path(s, require_feedback=True)
    w = s["restore_path_trace_events"][0]["workers"][0]
    w.update(
        executor_output_ready_s=0.0072,
        engine_output_consumed_s=0.0075,
        scheduler_update_entry_s=0.0076,
        executor="UniProcExecutor",
        engine_mode="batch_queue",
    )
    a = mod.analyze_restore_path(s, require_feedback=True)
    assert a["feedback_count"] == 1
    assert a["feedback_means"]["ready_to_engine_ms"] == pytest.approx(0.3)
    assert sum(a["feedback_means"].values()) == pytest.approx(1)


@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "order", "nan", "executor", "mode"]
)
def test_bad_feedback(mod, fault):
    s = sample()
    w = s["restore_path_trace_events"][0]["workers"][0]
    w.update(
        executor_output_ready_s=0.0072,
        engine_output_consumed_s=0.0075,
        scheduler_update_entry_s=0.0076,
        executor="UniProcExecutor",
        engine_mode="step",
    )
    if fault == "missing":
        del w["executor_output_ready_s"]
    elif fault == "duplicate":
        w["duplicate_output_stamp"] = True
    elif fault == "order":
        w["executor_output_ready_s"] = 0.009
    elif fault == "nan":
        w["engine_output_consumed_s"] = float("nan")
    elif fault == "executor":
        w["executor"] = "multi"
    else:
        w["engine_mode"] = None
    with pytest.raises(ValueError):
        mod.analyze_restore_path(s, require_feedback=True)


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "duplicate",
        "host",
        "workers",
        "order",
        "nan",
        "dma",
        "partition",
        "dropped",
        "preemption",
        "bytes",
    ],
)
def test_invalid_traces_fail_closed(mod, fault):
    s = sample()
    e = s["restore_path_trace_events"][0]
    w = e["workers"][0]
    if fault == "missing":
        s["restore_path_trace_events"] = []
    elif fault == "duplicate":
        s["restore_path_trace_events"].append(e)
    elif fault == "host":
        w["host"] = "other"
    elif fault == "workers":
        e["workers"].append(w)
    elif fault == "order":
        w["submit_start_s"] = 0
    elif fault == "nan":
        w["worker_observed_s"] = float("nan")
    elif fault == "dma":
        w["dma_elapsed_s"] = None
    elif fault == "partition":
        s["request_path_trace_events"][0]["queue_to_first_compute_ms"] = 10
    elif fault == "dropped":
        s["restore_path_trace_dropped"] = 1
    elif fault == "preemption":
        s["request_path_trace_events"][0]["preemptions"] = 1
    elif fault == "bytes":
        w["bytes"] = 0
    with pytest.raises(ValueError):
        mod.analyze_restore_path(s)
