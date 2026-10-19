# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.benchmarks.test_promotion_ttft_labels import rows
from tests.benchmarks.test_promotion_ttft_labels import sample as label_sample


@pytest.fixture
def mod(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    return importlib.import_module("analyze_promotion_restore_joint")


def sample():
    s = label_sample()
    s.update(
        restore_path_trace_enabled=1,
        restore_path_trace_events=[
            dict(
                request_id="chatcmpl-conv-turn3-abcdef12",
                host="local",
                decision_s=0.003,
                allocated_s=0.004,
                metadata_built_s=0.005,
                scheduler_observed_s=0.008,
                workers=[
                    dict(
                        host="local",
                        worker_received_s=0.006,
                        submit_start_s=0.0061,
                        submit_return_s=0.0062,
                        worker_observed_s=0.007,
                        executor_output_ready_s=0.0072,
                        engine_output_consumed_s=0.0075,
                        scheduler_update_entry_s=0.0076,
                        executor="UniProcExecutor",
                        engine_mode="batch_queue",
                        dma_elapsed_s=0.0005,
                        bytes=4096,
                    )
                ],
            )
        ],
    )
    return s


def test_offsets_and_feedback_not_double_counted(mod):
    a = mod.join_restore_labels(rows(), sample())
    e = a["events"][0]
    assert a["count"] == 1
    assert e["decision_clock_offset_ms"] == pytest.approx(1)
    assert sum(e[f] for f in mod.PHASES) == pytest.approx(8)
    assert e["before_schedule_ms"] + e["after_schedule_ms"] == pytest.approx(13)
    assert "worker_to_scheduler_ms" not in mod.PHASES
    assert "dma_elapsed_ms" not in mod.PHASES


@pytest.mark.parametrize(
    "fault", ["offset", "bytes", "retries", "missing", "feedback", "cache", "multi"]
)
def test_joint_audit_fails_closed(mod, fault):
    s = sample()
    if fault == "offset":
        s["restore_path_trace_events"][0]["decision_s"] = 0.001
    elif fault == "bytes":
        s["restore_path_trace_events"][0]["workers"][0]["bytes"] = 1
    elif fault == "retries":
        s["restore_path_trace_retried_decisions"] = 1
    elif fault == "missing":
        s["restore_path_trace_events"] = []
    elif fault == "feedback":
        del s["restore_path_trace_events"][0]["workers"][0]["executor_output_ready_s"]
    elif fault == "cache":
        p = s["request_path_trace_events"][0]
        p["first_compute_cached_tokens"] -= 1
        p["completed_prefill_tokens"] += 1
        p["scheduled_prefill_tokens"] += 1
    else:
        s["promotion_ttft_trace_events"].append(
            dict(s["promotion_ttft_trace_events"][0], event_id=1)
        )
        s["promotion_ttft_trace_recorded"] = s["offload_promotion_decision_count"] = 2
    with pytest.raises(ValueError):
        mod.join_restore_labels(rows(), s)


def test_diagnostic_requires_explicit_mixed_mode(mod):
    runner = importlib.import_module("run_promotion_ttft_calibration")
    with pytest.raises(ValueError, match="requires --mixed-seeds"):
        runner.main(SimpleNamespace(restore_diagnostic=True, mixed_seeds=None))


def test_aggregate_partition_and_constant_correlation(mod):
    summary = importlib.import_module("summarize_promotion_restore_joint")
    fields = [f for group in summary.GROUPS.values() for f in group]
    assert len(fields) == len(set(fields))
    assert set(fields) == set(mod.PHASES)
    e = mod.join_restore_labels(rows(), sample())["events"][0]
    e.update(conversation="conv", before_error_ms=1.0, total_error_ms=2.0)
    report = summary.summarize([e, e])
    assert sum(p["mean_ms"] for p in report["phases"].values()) == pytest.approx(8.0)
    assert all(p["corr_with_before_error"] is None for p in report["phases"].values())
    assert report["dma_mean_ms"] == pytest.approx(0.5)
