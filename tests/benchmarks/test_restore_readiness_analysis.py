# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "analyze_restore_readiness",
    Path(__file__).parents[2] / "scripts/analyze_restore_readiness.py",
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def fixture():
    points = [
        dict(loop=2, stage=stage, start_s=t, end_s=t + 0.001, done=done)
        for stage, t, done in (
            ("entry", 1.0, False),
            ("before_execute", 2.0, False),
            ("after_execute", 3.0, True),
            ("before_result", 4.0, True),
            ("after_result", 5.0, True),
        )
    ]
    stats = dict(
        restore_readiness_trace_enabled=True,
        restore_readiness_trace_restore_batches=1,
        restore_readiness_trace_events=[
            dict(
                batch_id=1,
                created_loop=1,
                consumed_loop=2,
                request_ids=["r"],
                observations=points,
                consumption=points,
            )
        ],
        restore_path_trace_events=[
            dict(
                request_id="r",
                workers=[
                    dict(
                        executor_output_ready_s=2.5,
                        engine_output_consumed_s=4.5,
                    )
                ],
            )
        ],
    )
    joined = dict(
        events=[
            dict(
                request_id="r",
                ready_to_engine_ms=2000.0,
                before_schedule_ms=6000.0,
            )
        ]
    )
    return stats, joined


def test_analysis_brackets_visibility_and_joins():
    stats, joined = fixture()
    result = module.analyze_readiness(stats, joined)
    assert result["first_done_stages"] == {"after_execute": 1}
    assert result["output_stamp_phases"] == {"by_after_execute": 1}
    assert result["same_loop_requests"] == 0
    assert result["events"][0]["visibility_bracket_ms"] == pytest.approx(1001.0)


def test_analysis_rejects_missing_and_duplicate_notifications():
    stats, joined = fixture()
    stats["restore_readiness_trace_events"][0]["request_ids"] = ["other"]
    with pytest.raises(ValueError, match="request set"):
        module.analyze_readiness(stats, joined)
    stats["restore_readiness_trace_events"][0]["request_ids"] = ["r", "r"]
    with pytest.raises(ValueError, match="Repeated"):
        module.analyze_readiness(stats, joined)


def test_analysis_rejects_overflow_and_invalid_order():
    stats, joined = fixture()
    stats["restore_readiness_trace_dropped"] = 1
    with pytest.raises(ValueError, match="Incomplete"):
        module.analyze_readiness(stats, joined)
    del stats["restore_readiness_trace_dropped"]
    stats["restore_path_trace_events"][0]["workers"][0]["executor_output_ready_s"] = 6.0
    with pytest.raises(ValueError, match="ordering"):
        module.analyze_readiness(stats, joined)
