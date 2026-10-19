# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import time
from collections import OrderedDict, defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import islice
from typing import Any

import torch

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.distributed.kv_events import BlockRemoved, BlockStored, KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.utils import yield_req_data
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.v1.attention.backend import AttentionBackend, AttentionMetadata
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.abstract import OffloadingManager, PrepareStoreOutput
from vllm.v1.kv_offload.factory import OffloadingSpecFactory
from vllm.v1.kv_offload.lifecycle_policy import (
    LifecycleOffloadingManager,
    LifecyclePolicyConfig,
)
from vllm.v1.kv_offload.mediums import GPULoadStoreSpec
from vllm.v1.kv_offload.promotion_calibration import CalibrationActions
from vllm.v1.kv_offload.promotion_trace import PromotionDecisionTrace
from vllm.v1.kv_offload.restore_trace import TRACE_HOST, RestorePathTrace
from vllm.v1.kv_offload.spec import OffloadingSpec
from vllm.v1.kv_offload.worker.worker import (
    OffloadingWorker,
    TransferSpec,
    TransferType,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

ReqId = str

logger = init_logger(__name__)


@dataclass
class OffloadingOperationMetrics:
    op_size: int
    op_time: float
    event_id: int | None = None
    request_id: str | None = None
    block_count: int | None = None
    submit_to_complete_time: float | None = None
    host_submit_time: float | None = None
    blocking_wait_time: float | None = None
    restore_timing: dict[str, Any] | None = None


@dataclass
class OffloadingPromotionStats:
    decision_count: int = 0
    candidate_blocks: int = 0
    candidate_tokens: int = 0
    restore_decision_count: int = 0
    restore_blocks: int = 0
    restore_tokens: int = 0
    recompute_decision_count: int = 0
    recompute_blocks: int = 0
    recompute_tokens: int = 0


@dataclass
class OffloadingWorkerJob:
    request_id: ReqId
    store: bool
    event_id: int | None = None
    block_count: int | None = None
    submitted_at: float | None = None
    host_submit_time: float | None = None
    blocking_wait_time: float = 0.0
    restore_timing: dict[str, Any] | None = None


@dataclass
class OffloadingConnectorStats(KVConnectorStats):
    def __post_init__(self):
        if not self.data:
            # Empty container init, no data is passed in.
            self.reset()

    def reset(self):
        self.data: dict[str, list[OffloadingOperationMetrics]] = {}

    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
        if not other.is_empty():
            for k, v in other.data.items():
                if k not in self.data:
                    self.data[k] = v
                else:
                    accumulator = self.data[k]
                    assert isinstance(accumulator, list)
                    accumulator.extend(v)
        return self

    def reduce(self) -> dict[str, int | float]:
        """
        Reduce the observations collected during a time interval to one or
        more representative values (eg avg/median/sum of the series).
        This is meant to be called by the logger to produce a summary of the
        stats for the last time interval.
        """
        return_dict: dict[str, int | float] = {}
        for transfer_type, ops_list in self.data.items():
            assert isinstance(ops_list, list)
            total_bytes = 0
            total_time = 0.0
            for op in ops_list:
                total_bytes += int(self._metric_value(op, "op_size"))
                total_time += float(self._metric_value(op, "op_time"))
            return_dict[f"{transfer_type}_total_bytes"] = total_bytes
            return_dict[f"{transfer_type}_total_time"] = total_time
        return return_dict

    def is_empty(self) -> bool:
        return not self.data

    @staticmethod
    def _metric_value(op: Any, name: str) -> Any:
        return op.get(name) if isinstance(op, dict) else getattr(op, name)

    def record_transfer(
        self,
        num_bytes: int,
        time: float,
        transfer_type: TransferType,
        *,
        event_id: int | None = None,
        request_id: str | None = None,
        block_count: int | None = None,
        submit_to_complete_time: float | None = None,
        host_submit_time: float | None = None,
        blocking_wait_time: float | None = None,
        restore_timing: dict[str, Any] | None = None,
    ):
        src, dst = transfer_type
        transfer_type_key = src + "_to_" + dst
        op = OffloadingOperationMetrics(
            num_bytes,
            time,
            event_id=event_id,
            request_id=request_id,
            block_count=block_count,
            submit_to_complete_time=submit_to_complete_time,
            host_submit_time=host_submit_time,
            blocking_wait_time=blocking_wait_time,
            restore_timing=restore_timing,
        )
        if transfer_type_key in self.data:
            self.data[transfer_type_key].append(op)
        else:
            self.data[transfer_type_key] = [op]


@dataclass
class OffloadingConnectorMetadata(KVConnectorMetadata):
    reqs_to_load: dict[ReqId, TransferSpec]
    reqs_to_store: dict[ReqId, TransferSpec]
    load_cost_events: dict[ReqId, tuple[int, int]]
    store_cost_events: dict[ReqId, tuple[int, int]]


class OffloadingConnector(KVConnectorBase_V1):
    @property
    def prefer_cross_layer_blocks(self) -> bool:
        return True

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        assert kv_cache_config is not None
        spec = OffloadingSpecFactory.create_spec(vllm_config, kv_cache_config)

        self.connector_scheduler: OffloadingConnectorScheduler | None = None
        self.connector_worker: OffloadingConnectorWorker | None = None
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = OffloadingConnectorScheduler(spec)
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = OffloadingConnectorWorker(spec)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def register_cross_layers_kv_cache(
        self, kv_cache: torch.Tensor, attn_backend: type[AttentionBackend]
    ):
        assert self.connector_worker is not None
        self.connector_worker.register_cross_layers_kv_cache(kv_cache, attn_backend)

    def handle_preemptions(self, preempted_req_ids: set[str]):
        assert self.connector_worker is not None
        self.connector_worker.handle_preemptions(preempted_req_ids)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, OffloadingConnectorMetadata)
        self.connector_worker.start_kv_transfers(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        pass

    def wait_for_save(self):
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, OffloadingConnectorMetadata)
        self.connector_worker.prepare_store_kv(self._connector_metadata)

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        assert self.connector_worker is not None
        return self.connector_worker.get_finished(finished_req_ids)

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens
        )

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens
        )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def update_connector_output(self, connector_output: KVConnectorOutput):
        assert self.connector_scheduler is not None
        self.connector_scheduler.update_connector_output(connector_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    def take_events(self) -> Iterable[KVCacheEvent]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.take_events()

    def get_value_policy_stats(self, reset: bool = False) -> dict[str, Any]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_value_policy_stats(reset=reset)

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        if self.connector_worker is None:
            return None  # We only emit stats from the worker-side
        return self.connector_worker.get_kv_connector_stats()

    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> KVConnectorStats | None:
        return (
            OffloadingConnectorStats(data=data)
            if data is not None
            else OffloadingConnectorStats()
        )

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> KVConnectorPromMetrics:
        return OffloadPromMetrics(
            vllm_config, metric_types, labelnames, per_engine_labelvalues
        )


class OffloadingConnectorScheduler:
    """Implementation of Scheduler side methods"""

    _SUPPORTED_PROMOTION_POLICIES = frozenset(
        {"always_restore", "always_recompute", "cost_aware", "calibration_mixed"}
    )

    def __init__(self, spec: OffloadingSpec):
        assert len(spec.gpu_block_size) == 1
        self.gpu_block_size = spec.gpu_block_size[0]
        self.offloaded_block_size = self.gpu_block_size * spec.block_size_factor
        self.block_size_factor = spec.block_size_factor
        self.manager: OffloadingManager = spec.get_manager()

        self.lifecycle_policy: LifecycleOffloadingManager | None = None
        if "lifecycle_policy" in spec.extra_config:
            if int(spec.extra_config.get("store_threshold", 0)) > 1:
                raise ValueError(
                    "lifecycle_policy cannot be combined with store_threshold > 1"
                )
            config = LifecyclePolicyConfig.from_dict(
                spec.extra_config["lifecycle_policy"]
            )
            self.lifecycle_policy = LifecycleOffloadingManager(
                self.manager, config, self.offloaded_block_size
            )
            self.manager = self.lifecycle_policy

        self.promotion_policy = str(
            spec.extra_config.get(
                "promotion_policy",
                "cost_aware" if self.lifecycle_policy else "always_restore",
            )
        )
        if self.promotion_policy == "cost_aware" and self.lifecycle_policy is None:
            raise ValueError(
                "cost_aware requires lifecycle_policy with calibrated cost priors"
            )
        if self.promotion_policy not in self._SUPPORTED_PROMOTION_POLICIES:
            supported = ", ".join(sorted(self._SUPPORTED_PROMOTION_POLICIES))
            raise ValueError(
                "Unsupported offloading promotion_policy "
                f"{self.promotion_policy!r}; expected one of: {supported}"
            )
        self._promotion_stats = OffloadingPromotionStats()

        self.promotion_ttft_trace = (
            PromotionDecisionTrace()
            if spec.extra_config.get("track_promotion_ttft", False)
            else None
        )
        self.calibration_actions = None
        if self.promotion_policy == "calibration_mixed":
            if (
                self.promotion_ttft_trace is None
                or (
                    self.lifecycle_policy is not None
                    and (
                        self.lifecycle_policy.config.admission_enabled
                        or self.lifecycle_policy.config.guard_enabled
                    )
                )
                or int(spec.extra_config.get("store_threshold", 0)) > 1
            ):
                raise ValueError(
                    "calibration_mixed requires decision tracing and fixed admission; "
                    "disable admission controller, guard and selective threshold"
                )
            self.calibration_actions = CalibrationActions(
                spec.extra_config.get("calibration_action_assignments")
            )
        self._trace_bytes_per_block = getattr(
            spec, "kv_bytes_per_offloaded_block", None
        )

        self.restore_path_trace = (
            RestorePathTrace()
            if os.environ.get("VLLM_EXPERIMENTAL_RESTORE_PATH_TRACE") == "1"
            else None
        )

        self.track_cost_events = bool(spec.extra_config.get("track_cost_events", False))
        self.trace_admission = bool(
            self.lifecycle_policy and self.lifecycle_policy.config.trace_admission
        )
        if self.trace_admission and not self.track_cost_events:
            raise ValueError("trace_admission requires track_cost_events")
        self.max_cost_events = int(spec.extra_config.get("max_cost_events", 4096))
        self.max_cost_event_blocks = int(
            spec.extra_config.get("max_cost_event_blocks", 64_000)
        )
        if self.max_cost_events < 1:
            raise ValueError("max_cost_events must be >= 1")
        if self.max_cost_event_blocks < 1:
            raise ValueError("max_cost_event_blocks must be >= 1")
        self._cost_events: deque[dict[str, Any]] = deque()
        self._cost_events_by_id: dict[int, dict[str, Any]] = {}
        self._cost_event_started_at: dict[int, float] = {}
        self._cost_event_counter = 0
        self._cost_event_dropped_count = 0
        self._cost_event_unmatched_transfer_count = 0
        self._cost_block_ids: OrderedDict[BlockHash, int] = OrderedDict()
        self._cost_block_id_counter = 0
        self._cost_block_id_eviction_count = 0
        self._pending_load_cost_event_ids: dict[ReqId, int] = {}
        self._active_load_cost_event_ids: dict[ReqId, int] = {}
        self._active_recompute_cost_events: dict[ReqId, tuple[int, int]] = {}
        self._reqs_to_load_cost_events: dict[ReqId, tuple[int, int]] = {}
        self._reqs_to_store_cost_events: dict[ReqId, tuple[int, int]] = {}

        self._requests: dict[ReqId, Request] = {}
        # list of GPU block IDs per request
        self._request_block_ids: dict[ReqId, list[int]] = {}
        # requests to load for the current scheduler step
        self._reqs_to_load: dict[ReqId, TransferSpec] = {}
        # request blocks are stored in order
        # index of next block (of size offloaded_block_size) to offload
        self._next_stored_block_idx: dict[ReqId, int] = {}
        # if GPU prefix caching is enabled,
        # track loaded blocks to avoid redundant loads
        self._blocks_being_loaded: set[BlockHash] | None = (
            set() if spec.vllm_config.cache_config.enable_prefix_caching else None
        )

        # request ID -> set(block hashes being stored/load)
        self._reqs_being_stored = defaultdict[ReqId, set[BlockHash]](set)
        self._reqs_being_loaded = defaultdict[ReqId, set[BlockHash]](set)

    def _get_cost_block_ids(self, block_hashes: Iterable[BlockHash]) -> list[int]:
        block_ids = []
        for block_hash in block_hashes:
            block_id = self._cost_block_ids.get(block_hash)
            if block_id is None:
                if len(self._cost_block_ids) >= self.max_cost_event_blocks:
                    self._cost_block_ids.popitem(last=False)
                    self._cost_block_id_eviction_count += 1
                block_id = self._cost_block_id_counter
                self._cost_block_id_counter += 1
                self._cost_block_ids[block_hash] = block_id
            else:
                self._cost_block_ids.move_to_end(block_hash)
            block_ids.append(block_id)
        return block_ids

    def _append_cost_event(self, event: dict[str, Any]) -> int | None:
        if not self.track_cost_events:
            return None
        event_id = self._cost_event_counter
        self._cost_event_counter += 1
        event["event_id"] = event_id
        if len(self._cost_events) >= self.max_cost_events:
            dropped = self._cost_events.popleft()
            dropped_id = int(dropped["event_id"])
            self._cost_events_by_id.pop(dropped_id, None)
            self._cost_event_started_at.pop(dropped_id, None)
            self._cost_event_dropped_count += 1
        self._cost_events.append(event)
        self._cost_events_by_id[event_id] = event
        self._cost_event_started_at[event_id] = time.perf_counter()
        return event_id

    def _record_admission_diagnostic(
        self,
        request_id: str,
        start_block_idx: int,
        hashes: list[BlockHash],
        output: PrepareStoreOutput | None,
    ) -> None:
        if not self.trace_admission:
            return
        assert self.lifecycle_policy is not None
        decision = self.lifecycle_policy.last_admission
        if decision is None:
            return
        self._append_cost_event(
            {
                **decision,
                "event_type": "admission",
                "request_id": request_id,
                "start_block_idx": start_block_idx,
                "candidate_block_ids": self._get_cost_block_ids(hashes),
                "allocation_failed": output is None,
                "stored_block_ids": self._get_cost_block_ids(
                    output.block_hashes_to_store if output is not None else []
                ),
            }
        )

    def _record_completed_transfer_metrics(
        self, connector_stats: KVConnectorStats | None
    ) -> None:
        if not self.track_cost_events or not isinstance(
            connector_stats, OffloadingConnectorStats
        ):
            return
        for transfer_type, operations in connector_stats.data.items():
            for operation in operations:
                event_id = OffloadingConnectorStats._metric_value(operation, "event_id")
                if event_id is None:
                    continue
                event = self._cost_events_by_id.get(int(event_id))
                if event is None:
                    self._cost_event_unmatched_transfer_count += 1
                    continue
                observation = {
                    "transfer_type": transfer_type,
                    "bytes": int(
                        OffloadingConnectorStats._metric_value(operation, "op_size")
                    ),
                    "dma_time_s": float(
                        OffloadingConnectorStats._metric_value(operation, "op_time")
                    ),
                    "blocks": OffloadingConnectorStats._metric_value(
                        operation, "block_count"
                    ),
                    "submit_to_complete_s": OffloadingConnectorStats._metric_value(
                        operation, "submit_to_complete_time"
                    ),
                    "host_submit_s": OffloadingConnectorStats._metric_value(
                        operation, "host_submit_time"
                    ),
                    "blocking_wait_s": OffloadingConnectorStats._metric_value(
                        operation, "blocking_wait_time"
                    ),
                }
                event.setdefault("transfer_observations", []).append(observation)
                observations = event["transfer_observations"]
                event["dma_bytes"] = sum(item["bytes"] for item in observations)
                event["dma_time_s"] = max(item["dma_time_s"] for item in observations)
                event["submit_to_complete_s"] = max(
                    item["submit_to_complete_s"] or 0.0 for item in observations
                )
                if event.get("event_type") == "store":
                    # Retain old completion latency as a separate diagnostic.
                    # Max across workers, not sum; excludes indirect I/O interference.
                    if all(
                        item["host_submit_s"] is not None
                        and item["blocking_wait_s"] is not None
                        for item in observations
                    ):
                        event["store_host_submit_s"] = max(
                            item["host_submit_s"] for item in observations
                        )
                        event["store_blocking_wait_s"] = max(
                            item["blocking_wait_s"] for item in observations
                        )
                        event["store_host_service_s"] = event.get(
                            "prepare_store_wall_time_s", 0.0
                        ) + max(
                            item["host_submit_s"] + item["blocking_wait_s"]
                            for item in observations
                        )
                    self._cost_event_started_at.pop(int(event_id), None)

    def _complete_recompute_cost_event(
        self, request: Request, *, request_finished: bool = False
    ) -> None:
        if self.lifecycle_policy is not None:
            pending = self.lifecycle_policy.pending.get(request.request_id)
            if pending is not None and pending.action == "recompute":
                self.lifecycle_policy.complete_promotion(
                    request.request_id,
                    computed_tokens=request.num_computed_tokens,
                    cancelled=request_finished
                    and request.num_computed_tokens < pending.target,
                )
        active = self._active_recompute_cost_events.get(request.request_id)
        if active is None:
            return
        event_id, target_computed_tokens = active
        completed = request.num_computed_tokens >= target_computed_tokens
        if not completed and not request_finished:
            return
        self._active_recompute_cost_events.pop(request.request_id, None)
        event = self._cost_events_by_id.get(event_id)
        started_at = self._cost_event_started_at.pop(event_id, None)
        if event is None:
            return
        event["observed_num_computed_tokens"] = request.num_computed_tokens
        if completed and started_at is not None:
            event["recompute_wait_s"] = time.perf_counter() - started_at
        else:
            event["recompute_incomplete"] = True

    def get_value_policy_stats(self, reset: bool = False) -> dict[str, Any]:
        promotion_stats = {
            "offload_promotion_policy": self.promotion_policy,
            **{
                f"offload_promotion_{name}": value
                for name, value in vars(self._promotion_stats).items()
            },
        }
        get_stats = getattr(self.manager, "get_value_policy_stats", None)
        manager_stats = get_stats(reset=reset) if get_stats is not None else {}
        manager_stats.update(promotion_stats)
        if self.promotion_ttft_trace is not None:
            manager_stats.update(self.promotion_ttft_trace.snapshot(reset=reset))
        if self.restore_path_trace is not None:
            manager_stats.update(self.restore_path_trace.snapshot(reset=reset))
        manager_stats["offload_cost_events_enabled"] = int(self.track_cost_events)
        manager_stats["offload_admission_trace_enabled"] = int(self.trace_admission)
        if self.track_cost_events:
            manager_stats.update(
                {
                    "offload_cost_events_max": self.max_cost_events,
                    "offload_cost_events_retained": len(self._cost_events),
                    "offload_cost_events_dropped": self._cost_event_dropped_count,
                    "offload_cost_event_block_ids_retained": len(self._cost_block_ids),
                    "offload_cost_event_block_id_evictions": (
                        self._cost_block_id_eviction_count
                    ),
                    "offload_cost_event_unmatched_transfers": (
                        self._cost_event_unmatched_transfer_count
                    ),
                    "offload_cost_events": [dict(event) for event in self._cost_events],
                }
            )
        if reset:
            self._promotion_stats = OffloadingPromotionStats()
            self._cost_events.clear()
            self._cost_events_by_id.clear()
            self._cost_event_started_at.clear()
            self._cost_event_dropped_count = 0
            self._cost_event_unmatched_transfer_count = 0
        return manager_stats

    def _record_promotion_decision(
        self,
        action: str,
        blocks: int,
        tokens: int,
        *,
        request_id: ReqId,
        num_computed_tokens: int,
        lookup_wall_time_s: float,
        block_hashes: Iterable[BlockHash],
        prompt_tokens: int | None = None,
    ) -> int | None:
        if self.promotion_ttft_trace is not None:
            load_blocks = sum(len(v) for v in self._reqs_being_loaded.values())
            store_blocks = sum(len(v) for v in self._reqs_being_stored.values())
            unit = self._trace_bytes_per_block
            self.promotion_ttft_trace.record(
                request_id=request_id,
                action=action,
                candidate_blocks=blocks,
                candidate_tokens=tokens,
                gpu_hit_tokens=num_computed_tokens,
                prompt_tokens=prompt_tokens,
                tracked_requests=len(self._requests),
                pending_load_requests=len(self._reqs_being_loaded),
                pending_store_requests=len(self._reqs_being_stored),
                pending_load_blocks=load_blocks,
                pending_store_blocks=store_blocks,
                bytes_per_offloaded_block=unit,
                candidate_bytes=blocks * unit if unit is not None else None,
                pending_load_bytes=load_blocks * unit if unit is not None else None,
                pending_store_bytes=store_blocks * unit if unit is not None else None,
            )
        self._promotion_stats.decision_count += 1
        self._promotion_stats.candidate_blocks += blocks
        self._promotion_stats.candidate_tokens += tokens
        if action == "restore":
            self._promotion_stats.restore_decision_count += 1
            self._promotion_stats.restore_blocks += blocks
            self._promotion_stats.restore_tokens += tokens
        else:
            assert action == "recompute"
            self._promotion_stats.recompute_decision_count += 1
            self._promotion_stats.recompute_blocks += blocks
            self._promotion_stats.recompute_tokens += tokens
        if not self.track_cost_events:
            return None
        return self._append_cost_event(
            {
                "event_type": "promotion",
                "request_id": request_id,
                "action": action,
                "candidate_blocks": blocks,
                "candidate_tokens": tokens,
                "num_computed_tokens": num_computed_tokens,
                "target_computed_tokens": num_computed_tokens + tokens,
                "lookup_wall_time_s": lookup_wall_time_s,
                "block_ids": self._get_cost_block_ids(block_hashes),
            }
        )

    def _get_block_hashes(
        self,
        req: Request,
        start_idx: int = 0,
        end_idx: int | None = None,
    ) -> Iterable[BlockHash]:
        return islice(
            req.block_hashes,
            self.block_size_factor * start_idx + self.block_size_factor - 1,
            self.block_size_factor * end_idx if end_idx else None,
            self.block_size_factor,
        )

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        """
        Get number of new tokens that can be loaded beyond the
        num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            A tuple with the following elements:
                - The number of tokens that can be loaded beyond what is
                  already computed.
                  If None, it means that the connector needs more time to
                  determine the number of matched tokens, and the scheduler
                  should query for this request again later.
                - `True` if tokens will be loaded asynchronously
                  (between scheduler steps).
        """
        num_blocks = request.num_tokens // self.offloaded_block_size

        assert len(request.block_hashes) // self.block_size_factor == num_blocks
        block_hashes = self._get_block_hashes(request)

        self.manager.touch(block_hashes)

        if self.lifecycle_policy is not None:
            self.lifecycle_policy.observe_request(
                request.request_id, self._get_block_hashes(request)
            )
            self.lifecycle_policy.context = (
                max(1, len(self._requests)),
                len(self._reqs_being_loaded) + len(self._reqs_being_stored),
            )

        full_block_tokens = self.offloaded_block_size * num_blocks
        if full_block_tokens - num_computed_tokens < self.offloaded_block_size:
            # we can load less than a block, skip
            return 0, False

        start_block_idx = num_computed_tokens // self.offloaded_block_size
        candidate_hashes = list(
            self._get_block_hashes(request, start_idx=start_block_idx)
        )
        lookup_started_at = time.perf_counter() if self.track_cost_events else None
        hits = self.manager.lookup(candidate_hashes)
        lookup_wall_time_s = (
            time.perf_counter() - lookup_started_at
            if lookup_started_at is not None
            else 0.0
        )
        if self.trace_admission:
            # Inspect the whole suffix without changing the prefix lookup result.
            # Ready blocks beyond the first gap are NOT actual CPU hits.
            self._append_cost_event(
                {
                    "event_type": "external_lookup",
                    "request_id": request.request_id,
                    "start_block_idx": start_block_idx,
                    "num_computed_tokens": num_computed_tokens,
                    "candidate_block_ids": self._get_cost_block_ids(candidate_hashes),
                    "candidate_states": self.manager.get_block_residency(
                        candidate_hashes
                    ),
                    "hit_blocks": hits,
                }
            )
        if hits is None:
            # indicates a lookup that should be tried later
            return None, False
        if hits == 0:
            return 0, False

        num_hit_tokens = (
            self.offloaded_block_size * (start_block_idx + hits) - num_computed_tokens
        )
        logger.debug(
            "Request %s hit %s offloaded tokens after %s GPU hit tokens",
            request.request_id,
            num_hit_tokens,
            num_computed_tokens,
        )
        if num_hit_tokens < self.offloaded_block_size:
            return 0, False

        action = (
            "recompute" if self.promotion_policy == "always_recompute" else "restore"
        )
        if self.calibration_actions is not None:
            action = self.calibration_actions.action(request.request_id)
        if self.lifecycle_policy is not None:
            action = self.lifecycle_policy.choose(
                request.request_id,
                hits,
                num_hit_tokens,
                num_computed_tokens + num_hit_tokens,
                forced_action=action if self.promotion_policy != "cost_aware" else None,
            )
        if action == "recompute":
            event_id = self._record_promotion_decision(
                "recompute",
                hits,
                num_hit_tokens,
                request_id=request.request_id,
                num_computed_tokens=num_computed_tokens,
                lookup_wall_time_s=lookup_wall_time_s,
                block_hashes=candidate_hashes[:hits],
                prompt_tokens=getattr(request, "num_prompt_tokens", request.num_tokens),
            )
            if event_id is not None:
                self._active_recompute_cost_events[request.request_id] = (
                    event_id,
                    num_computed_tokens + num_hit_tokens,
                )
            logger.debug(
                "Request %s bypassing %s offloaded tokens to recompute them",
                request.request_id,
                num_hit_tokens,
            )
            return 0, False

        if self._blocks_being_loaded and any(
            block_hash in self._blocks_being_loaded
            for block_hash in candidate_hashes[:hits]
        ):
            # hit blocks are being loaded, delay request
            logger.debug(
                "Delaying request %s since some of its blocks are already being loaded",
                request.request_id,
            )
            return None, False

        event_id = self._record_promotion_decision(
            "restore",
            hits,
            num_hit_tokens,
            request_id=request.request_id,
            num_computed_tokens=num_computed_tokens,
            lookup_wall_time_s=lookup_wall_time_s,
            block_hashes=candidate_hashes[:hits],
            prompt_tokens=getattr(request, "num_prompt_tokens", request.num_tokens),
        )
        if event_id is not None:
            self._pending_load_cost_event_ids[request.request_id] = event_id
        if self.restore_path_trace is not None:
            self.restore_path_trace.decision(request.request_id)
        return num_hit_tokens, True

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ):
        self._requests[request.request_id] = request
        # the block ids are updated in _get_reqs_to_store
        self._request_block_ids[request.request_id] = []

        if num_external_tokens == 0:
            return

        if self.restore_path_trace is not None:
            self.restore_path_trace.mark(request.request_id, "allocated_s")

        load_cost_event_id = self._pending_load_cost_event_ids.pop(
            request.request_id, None
        )

        block_groups = blocks.get_block_ids()
        block_ids = block_groups[0]

        num_computed_gpu_blocks = sum(
            block.block_hash is not None for block in blocks.blocks[0]
        )
        num_computed_tokens = num_computed_gpu_blocks * self.gpu_block_size
        full_block_tokens = num_computed_tokens + num_external_tokens
        assert full_block_tokens % self.offloaded_block_size == 0

        num_pending_gpu_blocks = len(block_ids) - num_computed_gpu_blocks
        assert num_external_tokens == num_pending_gpu_blocks * self.gpu_block_size

        start_block_idx = num_computed_tokens // self.offloaded_block_size
        num_blocks = full_block_tokens // self.offloaded_block_size

        assert len(request.block_hashes) // self.block_size_factor >= num_blocks
        block_hashes = self._get_block_hashes(
            request, start_idx=start_block_idx, end_idx=num_blocks
        )

        src_spec = self.manager.prepare_load(block_hashes)
        dst_spec = GPULoadStoreSpec(block_ids[num_computed_gpu_blocks:])

        block_hashes = self._get_block_hashes(
            request, start_idx=start_block_idx, end_idx=num_blocks
        )

        self._reqs_to_load[request.request_id] = (src_spec, dst_spec)
        if load_cost_event_id is not None:
            loaded_blocks = num_external_tokens // self.offloaded_block_size
            self._reqs_to_load_cost_events[request.request_id] = (
                load_cost_event_id,
                loaded_blocks,
            )
            self._active_load_cost_event_ids[request.request_id] = load_cost_event_id
        req_blocks_being_loaded = self._reqs_being_loaded[request.request_id]
        req_blocks_being_loaded.update(block_hashes)
        self._next_stored_block_idx[request.request_id] = num_blocks

        if self._blocks_being_loaded is not None:
            self._blocks_being_loaded.update(req_blocks_being_loaded)

    def _get_reqs_to_store(self, scheduler_output: SchedulerOutput):
        reqs_to_store: dict[ReqId, TransferSpec] = {}
        # iterate over both new and cached requests
        for req_id, new_block_id_groups, preempted in yield_req_data(scheduler_output):
            if preempted:
                self._request_block_ids[req_id] = []

            if new_block_id_groups:
                new_block_ids = new_block_id_groups[0]
                self._request_block_ids[req_id] += new_block_ids

            block_ids = self._request_block_ids[req_id]

            req = self._requests[req_id]
            self._complete_recompute_cost_event(req)
            new_tokens = scheduler_output.num_scheduled_tokens[req_id]
            expected_tokens = req.num_computed_tokens + new_tokens
            # with async scheduling, some tokens may be missing
            total_tokens = min(expected_tokens, req.num_tokens)
            num_blocks = total_tokens // self.offloaded_block_size
            start_block_idx = self._next_stored_block_idx.get(req_id, 0)
            num_new_blocks = num_blocks - start_block_idx

            if num_new_blocks <= 0:
                continue

            num_gpu_blocks = num_blocks * self.block_size_factor
            assert len(req.block_hashes) >= num_gpu_blocks

            new_block_hashes = list(
                self._get_block_hashes(
                    req, start_idx=start_block_idx, end_idx=num_blocks
                )
            )
            prepare_store_started_at = (
                time.perf_counter() if self.track_cost_events else None
            )
            if self.lifecycle_policy is not None:
                self.lifecycle_policy.context = (
                    max(1, len(self._requests)),
                    len(self._reqs_being_loaded) + len(self._reqs_being_stored),
                )
            store_output = self.manager.prepare_store(new_block_hashes)
            prepare_store_wall_time_s = (
                time.perf_counter() - prepare_store_started_at
                if prepare_store_started_at is not None
                else 0.0
            )
            self._record_admission_diagnostic(
                req_id, start_block_idx, new_block_hashes, store_output
            )
            if store_output is None:
                if self.track_cost_events:
                    self._append_cost_event(
                        {
                            "event_type": "store",
                            "request_id": req_id,
                            "action": "failed",
                            "start_block_idx": start_block_idx,
                            "requested_blocks": len(new_block_hashes),
                            "stored_blocks": 0,
                            "evicted_blocks": 0,
                            "prepare_store_wall_time_s": prepare_store_wall_time_s,
                            "requested_block_ids": self._get_cost_block_ids(
                                new_block_hashes
                            ),
                            "stored_block_ids": [],
                            "evicted_block_ids": [],
                        }
                    )
                logger.warning(
                    "Request %s: cannot store %s blocks", req_id, num_new_blocks
                )
                continue

            self._next_stored_block_idx[req_id] = num_blocks

            stored_hashes = list(store_output.block_hashes_to_store)
            event_id = None
            if self.track_cost_events:
                event_id = self._append_cost_event(
                    {
                        "event_type": "store",
                        "request_id": req_id,
                        "action": "store" if stored_hashes else "no_transfer",
                        "start_block_idx": start_block_idx,
                        "requested_blocks": len(new_block_hashes),
                        "stored_blocks": len(stored_hashes),
                        "evicted_blocks": len(store_output.block_hashes_evicted),
                        "prepare_store_wall_time_s": prepare_store_wall_time_s,
                        "requested_block_ids": self._get_cost_block_ids(
                            new_block_hashes
                        ),
                        "stored_block_ids": self._get_cost_block_ids(stored_hashes),
                        "evicted_block_ids": self._get_cost_block_ids(
                            store_output.block_hashes_evicted
                        ),
                    }
                )

            if not stored_hashes:
                continue
            block_hashes_to_store = set(stored_hashes)

            block_hashes = self._get_block_hashes(req, end_idx=num_blocks)
            self.manager.touch(block_hashes)

            new_block_hashes = self._get_block_hashes(
                req, start_idx=start_block_idx, end_idx=num_blocks
            )
            dst_spec = store_output.store_spec
            src_block_ids: list[int] = []
            for idx, blk_hash in enumerate(new_block_hashes):
                if blk_hash not in block_hashes_to_store:
                    continue
                offloaded_block_idx = start_block_idx + idx
                gpu_block_idx = offloaded_block_idx * self.block_size_factor
                for i in range(self.block_size_factor):
                    src_block_ids.append(block_ids[gpu_block_idx + i])
            src_spec = GPULoadStoreSpec(src_block_ids)

            reqs_to_store[req_id] = (src_spec, dst_spec)
            if event_id is not None:
                self._reqs_to_store_cost_events[req_id] = (
                    event_id,
                    len(stored_hashes),
                )
            self._reqs_being_stored[req_id] |= block_hashes_to_store

            logger.debug(
                "Request %s offloading %s blocks starting from block #%d",
                req_id,
                len(block_hashes_to_store),
                start_block_idx,
            )

        return reqs_to_store

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        meta = OffloadingConnectorMetadata(
            reqs_to_load=self._reqs_to_load,
            reqs_to_store=self._get_reqs_to_store(scheduler_output),
            load_cost_events=self._reqs_to_load_cost_events,
            store_cost_events=self._reqs_to_store_cost_events,
        )
        if self.restore_path_trace is not None:
            for req_id in meta.reqs_to_load:
                self.restore_path_trace.mark(req_id, "metadata_built_s")
        self._reqs_to_load = {}
        self._reqs_to_load_cost_events = {}
        self._reqs_to_store_cost_events = {}

        # NOTE (orozery): we should move this logic to update_connector_output
        # once KVConnectorOutput allows us to report completed transfers
        for req_id in scheduler_output.preempted_req_ids or ():
            if self.lifecycle_policy is not None:
                self.lifecycle_policy.complete_promotion(req_id, cancelled=True)
            block_hashes = self._reqs_being_stored.get(req_id)
            if block_hashes:
                self.manager.complete_store(block_hashes)
                block_hashes.clear()

        return meta

    def update_connector_output(self, connector_output: KVConnectorOutput):
        """
        Update KVConnector state from worker-side connectors output.

        Args:
            connector_output (KVConnectorOutput): the worker-side
                connectors output.
        """
        self._record_completed_transfer_metrics(connector_output.kv_connector_stats)
        if self.restore_path_trace is not None:
            stats = connector_output.kv_connector_stats
            if isinstance(stats, OffloadingConnectorStats):
                for operations in stats.data.values():
                    for operation in operations:
                        timing = (
                            operation.get("restore_timing")
                            if isinstance(operation, dict)
                            else operation.restore_timing
                        )
                        if timing is not None:
                            req_id = OffloadingConnectorStats._metric_value(
                                operation, "request_id"
                            )
                            self.restore_path_trace.observe(req_id, timing)

        for req_id in connector_output.finished_sending or []:
            block_hashes = self._reqs_being_stored.pop(req_id, None)
            if block_hashes:
                self.manager.complete_store(block_hashes)

        for req_id in connector_output.finished_recving or []:
            if self.restore_path_trace is not None:
                self.restore_path_trace.complete(req_id)
            if self.lifecycle_policy is not None:
                self.lifecycle_policy.complete_promotion(req_id)
            event_id = self._active_load_cost_event_ids.pop(req_id, None)
            if event_id is not None:
                event = self._cost_events_by_id.get(event_id)
                started_at = self._cost_event_started_at.pop(event_id, None)
                if event is not None and started_at is not None:
                    event["restore_wait_s"] = time.perf_counter() - started_at
            block_hashes = self._reqs_being_loaded.pop(req_id, None)
            if block_hashes:
                if self._blocks_being_loaded:
                    self._blocks_being_loaded.difference_update(block_hashes)
                self.manager.complete_load(block_hashes)

    def request_finished(
        self,
        request: Request,
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Called when a request has finished, before its blocks are freed.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """
        req_id = request.request_id
        if self.restore_path_trace is not None:
            self.restore_path_trace.cancel(req_id)
        self._complete_recompute_cost_event(request, request_finished=True)
        if self.lifecycle_policy is not None:
            self.lifecycle_policy.complete_promotion(req_id, cancelled=True)
        self._requests.pop(req_id, None)
        self._request_block_ids.pop(req_id, None)

        # TODO(orozery): possibly kickoff offload for last block
        # which may have been deferred due to async scheduling
        self._next_stored_block_idx.pop(req_id, None)

        request_being_stored = req_id in self._reqs_being_stored
        return request_being_stored, None

    def take_events(self) -> Iterable[KVCacheEvent]:
        """Take the KV cache events from the connector.

        Returns:
            A list of KV cache events.
        """
        for event in self.manager.take_events():
            if event.removed:
                yield BlockRemoved(block_hashes=event.block_hashes, medium=event.medium)
            else:
                yield BlockStored(
                    block_hashes=event.block_hashes,
                    parent_block_hash=None,
                    token_ids=[],
                    lora_id=None,
                    block_size=event.block_size,
                    medium=event.medium,
                    lora_name=None,
                )


class OffloadingConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(self, spec: OffloadingSpec):
        self.spec = spec
        self.worker = OffloadingWorker()
        self.track_restore_path = (
            os.environ.get("VLLM_EXPERIMENTAL_RESTORE_PATH_TRACE") == "1"
        )
        self.track_cost_events = bool(spec.extra_config.get("track_cost_events", False))

        self._job_counter = 0

        self.kv_connector_stats = OffloadingConnectorStats()
        # job_id -> request and optional cost-event metadata
        self._jobs: dict[int, OffloadingWorkerJob] = {}
        # req_id -> active job IDs
        self._load_job: dict[ReqId, int] = {}
        # req_id -> set(active job IDs)
        self._store_jobs = defaultdict[ReqId, set[int]](set)
        # list of store jobs pending submission (job_id, transfer_spec)
        self._unsubmitted_store_jobs: list[tuple[int, TransferSpec]] = []

        self._finished_reqs_waiting_for_store: set[ReqId] = set()

    def _generate_job_id(self) -> int:
        job_id = self._job_counter
        self._job_counter = job_id + 1
        return job_id

    def _submit_transfer(self, job_id: int, transfer_spec: TransferSpec) -> None:
        job = self._jobs[job_id]
        if job.restore_timing is not None:
            job.restore_timing["submit_start_s"] = time.perf_counter()
        if self.track_cost_events:
            job.submitted_at = time.perf_counter()
        success = self.worker.transfer_async(job_id, transfer_spec)
        if job.restore_timing is not None:
            job.restore_timing["submit_return_s"] = time.perf_counter()
        if self.track_cost_events and job.submitted_at is not None:
            job.host_submit_time = time.perf_counter() - job.submitted_at
        assert success

    def _register_handlers(
        self,
        kv_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
    ):
        for src_cls, dst_cls, handler in self.spec.get_handlers(
            kv_caches, attn_backends
        ):
            self.worker.register_handler(src_cls, dst_cls, handler)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        layer_names = list(kv_caches.keys())
        layers = get_layers_from_vllm_config(
            self.spec.vllm_config, Attention, layer_names
        )
        attn_backends = {
            layer_name: layers[layer_name].get_attn_backend()
            for layer_name in layer_names
        }
        self._register_handlers(kv_caches, attn_backends)

    def register_cross_layers_kv_cache(
        self, kv_cache: torch.Tensor, attn_backend: type[AttentionBackend]
    ):
        cross_layer_name = "ALL_LAYERS"
        kv_caches = {cross_layer_name: kv_cache}
        attn_backends = {cross_layer_name: attn_backend}
        self._register_handlers(kv_caches, attn_backends)

    def handle_preemptions(self, preempted_req_ids: set[str]):
        for job_id, transfer_spec in self._unsubmitted_store_jobs:
            self._submit_transfer(job_id, transfer_spec)
        self._unsubmitted_store_jobs.clear()

        for req_id in preempted_req_ids:
            job_ids = self._store_jobs.get(req_id)
            if job_ids:
                started = time.perf_counter() if self.track_cost_events else None
                self.worker.wait(job_ids)
                if started is not None:
                    elapsed = time.perf_counter() - started
                    # One measured wait for a group of jobs, apportioned once.
                    jobs = [self._jobs[j] for j in job_ids if j in self._jobs]
                    total_weight = sum(job.block_count or 1 for job in jobs)
                    for job in jobs:
                        job.blocking_wait_time += (
                            elapsed * (job.block_count or 1) / total_weight
                        )

    def start_kv_transfers(self, metadata: OffloadingConnectorMetadata):
        received = (
            time.perf_counter()
            if self.track_restore_path and metadata.reqs_to_load
            else None
        )
        for job_id, transfer_spec in self._unsubmitted_store_jobs:
            self._submit_transfer(job_id, transfer_spec)
        self._unsubmitted_store_jobs.clear()

        for req_id, transfer_spec in metadata.reqs_to_load.items():
            job_id = self._generate_job_id()
            cost_event = metadata.load_cost_events.get(req_id)
            event_id, block_count = (
                cost_event if cost_event is not None else (None, None)
            )
            self._jobs[job_id] = OffloadingWorkerJob(
                request_id=req_id,
                store=False,
                event_id=event_id,
                block_count=block_count,
                restore_timing=dict(host=TRACE_HOST, worker_received_s=received)
                if received is not None
                else None,
            )
            assert req_id not in self._load_job
            self._load_job[req_id] = job_id
            self._submit_transfer(job_id, transfer_spec)

    def prepare_store_kv(self, metadata: OffloadingConnectorMetadata):
        for req_id, transfer_spec in metadata.reqs_to_store.items():
            job_id = self._generate_job_id()
            cost_event = metadata.store_cost_events.get(req_id)
            event_id, block_count = (
                cost_event if cost_event is not None else (None, None)
            )
            self._jobs[job_id] = OffloadingWorkerJob(
                request_id=req_id,
                store=True,
                event_id=event_id,
                block_count=block_count,
            )
            self._store_jobs[req_id].add(job_id)
            # NOTE(orozery): defer the store to the beginning of the next engine step,
            # so that offloading starts AFTER transfers related to token sampling,
            # thereby avoiding delays to token generation due to offloading.
            self._unsubmitted_store_jobs.append((job_id, transfer_spec))

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens.
        Returns a list of request IDs that finished loading or storing.

        Returns:
            ids of requests that have finished asynchronous transfer
            tuple of (sending/saving ids, recving/loading ids).
        """
        finished_sending = set()
        finished_recving = set()
        for transfer_result in self.worker.get_finished():
            # we currently do not support job failures
            job_id = transfer_result.job_id
            assert transfer_result.success
            job = self._jobs.pop(job_id)
            req_id = job.request_id
            store = job.store
            if job.restore_timing is not None:
                job.restore_timing.update(
                    worker_observed_s=time.perf_counter(),
                    dma_elapsed_s=transfer_result.transfer_time,
                    bytes=transfer_result.transfer_size,
                )
            if (
                transfer_result.transfer_time is not None
                and transfer_result.transfer_size is not None
                and transfer_result.transfer_type is not None
            ):
                submit_to_complete_time = None
                if self.track_cost_events and job.submitted_at is not None:
                    submit_to_complete_time = time.perf_counter() - job.submitted_at
                self.kv_connector_stats.record_transfer(
                    num_bytes=transfer_result.transfer_size,
                    time=transfer_result.transfer_time,
                    transfer_type=transfer_result.transfer_type,
                    event_id=job.event_id if self.track_cost_events else None,
                    request_id=req_id
                    if self.track_cost_events or job.restore_timing is not None
                    else None,
                    block_count=job.block_count if self.track_cost_events else None,
                    submit_to_complete_time=submit_to_complete_time,
                    host_submit_time=job.host_submit_time
                    if self.track_cost_events
                    else None,
                    blocking_wait_time=job.blocking_wait_time
                    if self.track_cost_events
                    else None,
                    restore_timing=job.restore_timing,
                )
            if store:
                req_jobs = self._store_jobs[req_id]
                req_jobs.remove(job_id)
                if req_jobs:
                    continue

                if req_id in self._finished_reqs_waiting_for_store:
                    self._finished_reqs_waiting_for_store.remove(req_id)
                    finished_sending.add(req_id)
                    del self._store_jobs[req_id]
            else:
                req_job = self._load_job[req_id]
                assert job_id == req_job
                del self._load_job[req_id]
                finished_recving.add(req_id)

        for req_id in finished_req_ids:
            pending_req_jobs = self._store_jobs.get(req_id)
            if pending_req_jobs:
                self._finished_reqs_waiting_for_store.add(req_id)
            elif pending_req_jobs is not None:
                finished_sending.add(req_id)
                del self._store_jobs[req_id]

        return finished_sending, finished_recving

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        """
        Get the KV transfer stats for the connector.
        """

        if self.kv_connector_stats.is_empty():
            return None
        # Clear stats for next iteration
        kv_connector_stats = self.kv_connector_stats
        self.kv_connector_stats = OffloadingConnectorStats()
        return kv_connector_stats


class OffloadPromMetrics(KVConnectorPromMetrics):
    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)
        # (engine_idx, transfer_type) -> (metric with bounded labels)
        self.histogram_transfer_size: dict[tuple[int, str], PromMetricT] = {}
        self.counter_kv_bytes: dict[tuple[int, str], PromMetricT] = {}
        self.counter_kv_transfer_time: dict[tuple[int, str], PromMetricT] = {}
        buckets = [  # In bytes
            1e6,
            5e6,
            10e6,
            20e6,
            40e6,
            60e6,
            80e6,
            100e6,
            150e6,
            200e6,
        ]

        self._counter_kv_bytes = self._counter_cls(
            name="vllm:kv_offload_total_bytes",
            documentation="Number of bytes offloaded by KV connector",
            labelnames=labelnames + ["transfer_type"],
        )

        self._counter_kv_transfer_time = self._counter_cls(
            name="vllm:kv_offload_total_time",
            documentation="Total time measured by all KV offloading operations",
            labelnames=labelnames + ["transfer_type"],
        )

        self._histogram_transfer_size = self._histogram_cls(
            name="vllm:kv_offload_size",
            documentation="Histogram of KV offload transfer size, in bytes.",
            buckets=buckets[:],
            labelnames=labelnames + ["transfer_type"],
        )

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        """
        Observe transfer statistics from the new data structure.
        transfer_stats_data is expected to be a dict where:
        - keys are transfer type strings (e.g., "cpu_to_gpu", "gpu_to_cpu")
        - values are lists of OffloadingOperationMetrics objects
        """

        for transfer_type, ops in transfer_stats_data.items():
            # Cache:
            if (engine_idx, transfer_type) not in self.histogram_transfer_size:
                self.histogram_transfer_size[(engine_idx, transfer_type)] = (
                    self._histogram_transfer_size.labels(
                        *(self.per_engine_labelvalues[engine_idx] + [transfer_type])
                    )
                )
                self.counter_kv_bytes[(engine_idx, transfer_type)] = (
                    self._counter_kv_bytes.labels(
                        *(self.per_engine_labelvalues[engine_idx] + [transfer_type])
                    )
                )
                self.counter_kv_transfer_time[(engine_idx, transfer_type)] = (
                    self._counter_kv_transfer_time.labels(
                        *(self.per_engine_labelvalues[engine_idx] + [transfer_type])
                    )
                )

            # Process ops:
            assert isinstance(ops, list)
            for op in ops:  # ops is a list of serialized OffloadingOperationMetrics
                assert isinstance(op, dict)
                # Observe size histogram
                self.histogram_transfer_size[(engine_idx, transfer_type)].observe(
                    op["op_size"]
                )

                # Increment byte and time counters
                self.counter_kv_bytes[(engine_idx, transfer_type)].inc(op["op_size"])

                self.counter_kv_transfer_time[(engine_idx, transfer_type)].inc(
                    op["op_time"]
                )
