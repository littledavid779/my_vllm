# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnectorMetadata,
    OffloadingConnectorStats,
    OffloadingConnectorWorker,
)
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.restore_trace import RestorePathTrace
from vllm.v1.kv_offload.worker.worker import TransferResult
from vllm.v1.outputs import KVConnectorOutput

from .test_offloading_connector import _make_promotion_scheduler


def test_scheduler_lookup_alloc_and_metadata_hooks(monkeypatch):
    monkeypatch.setenv("VLLM_EXPERIMENTAL_RESTORE_PATH_TRACE", "1")
    scheduler, manager = _make_promotion_scheduler("always_restore")
    request = SimpleNamespace(
        request_id="r",
        num_tokens=12,
        block_hashes=[BlockHash(str(i).encode()) for i in range(3)],
    )
    assert scheduler.get_num_new_matched_tokens(request, 0) == (12, True)
    blocks = SimpleNamespace(
        get_block_ids=lambda: ([1, 2, 3],),
        blocks=([SimpleNamespace(block_hash=None) for _ in range(3)],),
    )
    scheduler.update_state_after_alloc(request, blocks, 12)
    output = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[],
            new_block_ids=[],
            resumed_req_ids=set(),
        ),
        preempted_req_ids=[],
    )
    meta = scheduler.build_connector_meta(output)
    assert set(meta.reqs_to_load) == {"r"}
    event = scheduler.restore_path_trace.active["r"]
    assert event["decision_s"] <= event["allocated_s"] <= event["metadata_built_s"]
    assert meta.load_cost_events == {}
    manager.prepare_load.assert_called_once()


def test_bounded_trace_reset_and_retry():
    now = [1.0]
    trace = RestorePathTrace(capacity=1, clock=lambda: now[0])
    trace.decision("a")
    now[0] = 2
    trace.decision("a")
    assert trace.active["a"]["decision_s"] == 2
    trace.decision("b")
    trace.mark("a", "allocated_s")
    trace.decision("a")
    assert trace.counters["duplicate_active"] == 1
    trace.snapshot(reset=True)
    assert "a" in trace.active
    trace.complete("a")
    trace.decision("b")
    trace.complete("b")
    assert trace.snapshot()["restore_path_trace_dropped"] == 1
    assert trace.events[0]["request_id"] == "b"
    trace.decision("c")
    trace.cancel("c")
    assert not trace.active
    assert trace.counters["cancelled"] == 1


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("serialized", [False, True])
def test_restore_only_timing_roundtrip(monkeypatch, enabled, serialized):
    monkeypatch.setenv("VLLM_EXPERIMENTAL_RESTORE_PATH_TRACE", str(int(enabled)))
    worker = OffloadingConnectorWorker(SimpleNamespace(extra_config={}))
    worker.worker = MagicMock()
    worker.worker.transfer_async.return_value = True
    scheduler, _ = _make_promotion_scheduler("always_restore")
    if enabled:
        scheduler.restore_path_trace.decision("r")
        scheduler.restore_path_trace.mark("r", "allocated_s")
        scheduler.restore_path_trace.mark("r", "metadata_built_s")
    # A deferred store must not gain timing metadata even with restore trace on.
    metadata = OffloadingConnectorMetadata(
        reqs_to_load={"r": (MagicMock(), MagicMock())},
        reqs_to_store={"s": (MagicMock(), MagicMock())},
        load_cost_events={},
        store_cost_events={},
    )
    worker.prepare_store_kv(metadata)  # job 0; load is job 1
    worker.start_kv_transfers(metadata)
    worker.worker.get_finished.return_value = [
        TransferResult(
            job_id=i,
            success=True,
            transfer_size=1024,
            transfer_time=0.001,
            transfer_type=kind,
        )
        for i, kind in enumerate([("GPU", "CPU"), ("CPU", "GPU")])
    ]
    _, received = worker.get_finished(set())
    assert received == {"r"}
    stats = worker.get_kv_connector_stats()
    assert stats.data["GPU_to_CPU"][0].restore_timing is None
    load = stats.data["CPU_to_GPU"][0]
    assert (load.restore_timing is not None) == enabled
    if enabled:
        t = load.restore_timing
        assert (
            t["worker_received_s"]
            <= t["submit_start_s"]
            <= t["submit_return_s"]
            <= t["worker_observed_s"]
        )
        assert t["dma_elapsed_s"] == 0.001
    if serialized:
        stats = OffloadingConnectorStats(
            data={k: [asdict(op) for op in ops] for k, ops in stats.data.items()}
        )
    scheduler.update_connector_output(
        KVConnectorOutput(
            finished_recving=received,
            kv_connector_stats=stats,
        )
    )
    result = scheduler.get_value_policy_stats()
    assert result["offload_cost_events_enabled"] == 0
    assert "offload_cost_events" not in result
    if enabled:
        assert len(result["restore_path_trace_events"]) == 1
        assert len(result["restore_path_trace_events"][0]["workers"]) == 1
        assert result["restore_path_trace_active"] == 0
    else:
        assert "restore_path_trace_events" not in result
