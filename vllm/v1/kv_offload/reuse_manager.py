# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Reuse-frequency gating for CPU KV-cache offload stores.

FilterReusedOffloadingManager — OffloadingManager decorator that skips
    storing blocks that have not yet been seen enough times.
"""

from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import (
    BlockResidency,
    LoadStoreSpec,
    OffloadingEvent,
    OffloadingManager,
    PrepareStoreOutput,
)


@dataclass
class FilterReusedOffloadingStats:
    lookup_query_count: int = 0
    lookup_observed_blocks: int = 0
    lookup_hit_query_count: int = 0
    lookup_hit_blocks: int = 0
    load_request_count: int = 0
    load_requested_blocks: int = 0
    load_completed_blocks: int = 0
    store_request_count: int = 0
    store_requested_blocks: int = 0
    store_eligible_blocks: int = 0
    store_dropped_blocks: int = 0
    store_actual_blocks: int = 0
    store_already_present_blocks: int = 0
    store_evicted_blocks: int = 0
    store_failed_count: int = 0
    store_completed_blocks: int = 0
    store_completion_failed_blocks: int = 0


@dataclass
class OffloadedBlockLifecycle:
    """Process-lifetime state for one block in the CPU offload tier."""

    store_count: int = 0
    cpu_eviction_count: int = 0
    load_count: int = 0
    resident: bool = False
    ever_looked_up_after_store: bool = False
    ever_cpu_hit_after_store: bool = False
    ever_looked_up_after_eviction: bool = False
    ever_evicted_before_first_lookup: bool = False


class FilterReusedOffloadingManager(OffloadingManager):
    """An :class:`OffloadingManager` decorator that skips storing blocks
    whose reuse frequency is below *store_threshold*.

    All methods are delegated to the *backing* manager.  Two methods are
    intercepted:

    * ``lookup`` — records each visited block hash in an internal LRU counter.
    * ``prepare_store`` — filters out block hashes that have not yet
      crossed the threshold *before* calling the backing
      ``prepare_store``.

    Args:
        backing: The underlying ``OffloadingManager`` to delegate to.
        store_threshold: A block must be seen at least this many times in
            ``lookup()`` before it is eligible for offloading. Values <= 1
            disable filtering while still collecting admission stats.
        max_tracker_size: Maximum entries in the internal tracker's LRU table.
        count_store_attempts: When set, a dropped store attempt also counts as
            an observation for future admission decisions. Eligibility for the
            current store is still evaluated before recording the attempt.
    """

    def __init__(
        self,
        backing: OffloadingManager,
        store_threshold: int = 2,
        max_tracker_size: int = 64_000,
        count_store_attempts: bool = False,
        track_lifecycle_stats: bool = False,
    ):
        if store_threshold < 0:
            raise ValueError(
                "FilterReusedOffloadingManager store_threshold must be >= 0, "
                f"got {store_threshold}"
            )
        if max_tracker_size < 1:
            raise ValueError(
                "FilterReusedOffloadingManager max_tracker_size must be >= 1, "
                f"got {max_tracker_size}"
            )
        self._backing = backing
        self.store_threshold = store_threshold
        self.max_tracker_size = max_tracker_size
        self.count_store_attempts = count_store_attempts
        self.track_lifecycle_stats = track_lifecycle_stats
        # Ordered so we can evict the LRU entry in O(1).
        self.counts: OrderedDict[BlockHash, int] = OrderedDict()
        # This opt-in tracker deliberately stores no token IDs, request IDs, or
        # serialized block hashes. It is bounded independently by the same
        # limit as the reuse counter and only exposes aggregate statistics.
        self.lifecycles: OrderedDict[BlockHash, OffloadedBlockLifecycle] = OrderedDict()
        self.lifecycle_tracker_evictions = 0
        self.stats = FilterReusedOffloadingStats()

    def _get_lifecycle(self, block_hash: BlockHash) -> OffloadedBlockLifecycle:
        lifecycle = self.lifecycles.get(block_hash)
        if lifecycle is not None:
            self.lifecycles.move_to_end(block_hash)
            return lifecycle
        if len(self.lifecycles) >= self.max_tracker_size:
            self.lifecycles.popitem(last=False)
            self.lifecycle_tracker_evictions += 1
        lifecycle = OffloadedBlockLifecycle()
        self.lifecycles[block_hash] = lifecycle
        return lifecycle

    def _record_lookup_lifecycles(
        self, block_hashes: list[BlockHash], hit_blocks: int
    ) -> None:
        if not self.track_lifecycle_stats:
            return
        # OffloadingManager.lookup() stops at its first miss. Only the hit
        # prefix and (when present) the first miss were actually inspected.
        inspected = min(len(block_hashes), hit_blocks + 1)
        if hit_blocks == len(block_hashes):
            inspected = hit_blocks
        for idx, block_hash in enumerate(block_hashes[:inspected]):
            lifecycle = self._get_lifecycle(block_hash)
            if lifecycle.store_count == 0:
                continue
            lifecycle.ever_looked_up_after_store = True
            if idx < hit_blocks:
                lifecycle.ever_cpu_hit_after_store = True
            elif not lifecycle.resident and lifecycle.cpu_eviction_count > 0:
                lifecycle.ever_looked_up_after_eviction = True

    def _record_cpu_evictions(self, block_hashes: Iterable[BlockHash]) -> None:
        if not self.track_lifecycle_stats:
            return
        for block_hash in block_hashes:
            lifecycle = self._get_lifecycle(block_hash)
            lifecycle.cpu_eviction_count += 1
            lifecycle.resident = False
            if not lifecycle.ever_looked_up_after_store:
                lifecycle.ever_evicted_before_first_lookup = True

    def _record_observed_blocks(self, block_hashes: Iterable[BlockHash]) -> None:
        for block_hash in block_hashes:
            if block_hash in self.counts:
                self.counts.move_to_end(block_hash)
                self.counts[block_hash] += 1
            else:
                if len(self.counts) >= self.max_tracker_size:
                    self.counts.popitem(last=False)  # evict LRU
                self.counts[block_hash] = 1

    # ------------------------------------------------------------------
    # Intercepted methods
    # ------------------------------------------------------------------

    def get_block_residency(
        self, block_hashes: Iterable[BlockHash]
    ) -> list[BlockResidency]:
        return self._backing.get_block_residency(block_hashes)

    def lookup(self, block_hashes: Iterable[BlockHash]) -> int | None:
        """Record each hash, then delegate lookup to backing manager."""
        block_hashes = list(block_hashes)
        self.stats.lookup_query_count += 1
        self.stats.lookup_observed_blocks += len(block_hashes)
        self._record_observed_blocks(block_hashes)
        hit_blocks = self._backing.lookup(block_hashes)
        if hit_blocks is not None:
            self._record_lookup_lifecycles(block_hashes, hit_blocks)
            if hit_blocks > 0:
                self.stats.lookup_hit_query_count += 1
                self.stats.lookup_hit_blocks += hit_blocks
        return hit_blocks

    def _select_store_blocks(self, block_hashes: list[BlockHash]) -> list[BlockHash]:
        if self.store_threshold <= 1:
            return block_hashes
        return [
            bh for bh in block_hashes if self.counts.get(bh, 0) >= self.store_threshold
        ]

    def prepare_store(
        self, block_hashes: Iterable[BlockHash]
    ) -> PrepareStoreOutput | None:
        """Filter out blocks below threshold, then delegate to backing.

        Filtering is evaluated *before* calling the backing manager's
        ``prepare_store`` so that blocks that would be skipped do not
        consume any CPU offload capacity.
        """
        block_hashes = list(block_hashes)
        eligible = self._select_store_blocks(block_hashes)
        self.stats.store_request_count += 1
        self.stats.store_requested_blocks += len(block_hashes)
        self.stats.store_eligible_blocks += len(eligible)
        self.stats.store_dropped_blocks += len(block_hashes) - len(eligible)
        if self.count_store_attempts and self.store_threshold > 1:
            self._record_observed_blocks(block_hashes)

        # Delegate to the backing manager with only the eligible hashes.
        # Passing an empty list is intentional and safe — both
        # LRUOffloadingManager and ARCOffloadingManager handle it correctly,
        # returning a PrepareStoreOutput with empty lists.
        store_output = self._backing.prepare_store(eligible)
        if store_output is None:
            self.stats.store_failed_count += 1
            return None
        self.stats.store_actual_blocks += len(store_output.block_hashes_to_store)
        self.stats.store_already_present_blocks += len(eligible) - len(
            store_output.block_hashes_to_store
        )
        self.stats.store_evicted_blocks += len(store_output.block_hashes_evicted)
        self._record_cpu_evictions(store_output.block_hashes_evicted)
        return store_output

    # ------------------------------------------------------------------
    # Delegated methods
    # ------------------------------------------------------------------

    def prepare_load(self, block_hashes: Iterable[BlockHash]) -> LoadStoreSpec:
        block_hashes = list(block_hashes)
        load_spec = self._backing.prepare_load(block_hashes)
        self.stats.load_request_count += 1
        self.stats.load_requested_blocks += len(block_hashes)
        return load_spec

    def touch(self, block_hashes: Iterable[BlockHash]) -> None:
        return self._backing.touch(block_hashes)

    def complete_load(self, block_hashes: Iterable[BlockHash]) -> None:
        block_hashes = list(block_hashes)
        self._backing.complete_load(block_hashes)
        self.stats.load_completed_blocks += len(block_hashes)
        if self.track_lifecycle_stats:
            for block_hash in block_hashes:
                lifecycle = self._get_lifecycle(block_hash)
                lifecycle.load_count += 1

    def complete_store(
        self, block_hashes: Iterable[BlockHash], success: bool = True
    ) -> None:
        block_hashes = list(block_hashes)
        self._backing.complete_store(block_hashes, success)
        if success:
            self.stats.store_completed_blocks += len(block_hashes)
            if self.track_lifecycle_stats:
                for block_hash in block_hashes:
                    lifecycle = self._get_lifecycle(block_hash)
                    lifecycle.store_count += 1
                    lifecycle.resident = True
        else:
            self.stats.store_completion_failed_blocks += len(block_hashes)

    def take_events(self) -> Iterable[OffloadingEvent]:
        return self._backing.take_events()

    def get_value_policy_stats(self, reset: bool = False) -> dict[str, int | float]:
        lifecycle_stats = self._get_lifecycle_stats()
        snapshot = {
            "offload_store_threshold": self.store_threshold,
            "offload_max_tracker_size": self.max_tracker_size,
            "offload_count_store_attempts": int(self.count_store_attempts),
            "offload_tracker_size": len(self.counts),
            "offload_lookup_query_count": self.stats.lookup_query_count,
            "offload_lookup_observed_blocks": self.stats.lookup_observed_blocks,
            "offload_lookup_hit_query_count": self.stats.lookup_hit_query_count,
            "offload_lookup_hit_blocks": self.stats.lookup_hit_blocks,
            "offload_load_request_count": self.stats.load_request_count,
            "offload_load_requested_blocks": self.stats.load_requested_blocks,
            "offload_load_completed_blocks": self.stats.load_completed_blocks,
            "offload_store_request_count": self.stats.store_request_count,
            "offload_store_requested_blocks": self.stats.store_requested_blocks,
            "offload_store_eligible_blocks": self.stats.store_eligible_blocks,
            "offload_store_dropped_blocks": self.stats.store_dropped_blocks,
            "offload_store_actual_blocks": self.stats.store_actual_blocks,
            "offload_store_already_present_blocks": (
                self.stats.store_already_present_blocks
            ),
            "offload_store_evicted_blocks": self.stats.store_evicted_blocks,
            "offload_store_failed_count": self.stats.store_failed_count,
            "offload_store_completed_blocks": self.stats.store_completed_blocks,
            "offload_store_completion_failed_blocks": (
                self.stats.store_completion_failed_blocks
            ),
            **lifecycle_stats,
        }
        if reset:
            self.stats = FilterReusedOffloadingStats()
        return snapshot

    def _get_lifecycle_stats(self) -> dict[str, int]:
        if not self.track_lifecycle_stats:
            return {"offload_lifecycle_stats_enabled": 0}

        stored = [
            lifecycle
            for lifecycle in self.lifecycles.values()
            if lifecycle.store_count > 0
        ]
        return {
            "offload_lifecycle_stats_enabled": 1,
            "offload_lifecycle_tracked_blocks": len(self.lifecycles),
            "offload_lifecycle_tracker_evictions": (self.lifecycle_tracker_evictions),
            "offload_lifecycle_ever_stored_blocks": len(stored),
            "offload_lifecycle_resident_blocks": sum(
                lifecycle.resident for lifecycle in stored
            ),
            "offload_lifecycle_never_looked_up_after_store_blocks": sum(
                not lifecycle.ever_looked_up_after_store for lifecycle in stored
            ),
            "offload_lifecycle_looked_up_after_store_blocks": sum(
                lifecycle.ever_looked_up_after_store for lifecycle in stored
            ),
            "offload_lifecycle_cpu_hit_after_store_blocks": sum(
                lifecycle.ever_cpu_hit_after_store for lifecycle in stored
            ),
            "offload_lifecycle_looked_up_after_eviction_blocks": sum(
                lifecycle.ever_looked_up_after_eviction for lifecycle in stored
            ),
            "offload_lifecycle_evicted_before_first_lookup_blocks": sum(
                lifecycle.ever_evicted_before_first_lookup for lifecycle in stored
            ),
            "offload_lifecycle_loaded_blocks": sum(
                lifecycle.load_count > 0 for lifecycle in stored
            ),
        }
