# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "analyze_restore_host",
    Path(__file__).parents[2] / "scripts/analyze_restore_host.py",
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def call():
    return dict(
        overflow=False,
        scheduled_tokens=8,
        scheduled_requests=2,
        marks=[
            dict(stage=s, wall_s=w, cpu_s=c)
            for s, w, c in (
                ("executor_dispatch", 1.0, 0.0),
                ("runner_setup", 2.0, 0.1),
                ("input_prep_event_sync", 3.0, 0.2),
                ("end", 5.0, 0.21),
            )
        ],
    )


def test_overlap_partition_keeps_cpu_and_wall_separate():
    parts = module.partition_call(call(), 2.5, 5.1)
    assert sum(parts["full_ms"].values()) == pytest.approx(4000)
    assert sum(parts["overlap_ms"].values()) == pytest.approx(2600)
    assert parts["overlap_ms"] == pytest.approx(
        dict(
            executor_dispatch=0,
            runner_setup=500,
            input_prep_event_sync=2000,
            outside_scope=100,
        )
    )
    assert parts["thread_cpu_ms"]["input_prep_event_sync"] == pytest.approx(10)


@pytest.mark.parametrize("start,end", [(0.0, 0.5), (6.0, 7.0), (2.0, 2.0)])
def test_outside_and_zero_windows(start, end):
    parts = module.partition_call(call(), start, end)
    assert sum(parts["overlap_ms"].values()) == pytest.approx((end - start) * 1000)


@pytest.mark.parametrize("failure", ["overflow", "end", "clock"])
def test_rejects_incomplete_trace(failure):
    data = call()
    if failure == "overflow":
        data["overflow"] = True
    elif failure == "end":
        data["marks"].pop()
    else:
        data["marks"][2]["wall_s"] = 0.0
    with pytest.raises(ValueError):
        module.partition_call(data, 2.0, 4.0)


def test_host_analysis_joins_consuming_call_and_excludes_other_cases():
    stats = dict(
        restore_host_trace_enabled=True,
        restore_readiness_trace_events=[
            dict(
                request_ids=["r"],
                consumption=[
                    dict(stage="before_execute", start_s=1.0),
                    dict(stage="after_execute", end_s=5.1, host_call=call()),
                ],
            ),
            dict(request_ids=["other"], consumption=[]),
        ],
        restore_path_trace_events=[
            dict(
                request_id="r",
                workers=[dict(executor_output_ready_s=2.5)],
            )
        ],
    )
    readiness = dict(
        events=[
            dict(request_id="r", first_done_stage="after_execute"),
            dict(request_id="other", first_done_stage="after_result"),
        ]
    )
    result = module.analyze_host(stats, readiness)
    assert result["requests"] == 2
    assert result["target_requests"] == 1
    assert sum(result["target_mean_overlap_ms"].values()) == pytest.approx(2600)
    assert result["events"][1]["no_execute"]
    stats["restore_readiness_trace_events"][0]["consumption"][1].pop("host_call")
    with pytest.raises(ValueError, match="Missing model runner"):
        module.analyze_host(stats, readiness)
