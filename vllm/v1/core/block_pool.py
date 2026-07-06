# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import math
import os
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from vllm.distributed.kv_events import (
    MEDIUM_GPU,
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVCacheEvent,
)
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashList,
    BlockHashListWithBlockSize,
    BlockHashWithGroupId,
    ExternalBlockHash,
    FreeKVCacheBlockQueue,
    KVCacheBlock,
    generate_block_hash_extra_keys,
    get_block_hash,
    make_block_hash_with_group_id,
    maybe_convert_block_hash,
)
from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class BlockValueStats:
    request_ids: set[str] = field(default_factory=set)
    store_count: int = 0
    apc_hit_count: int = 0
    access_count: int = 0
    last_access_s: float = 0.0
    last_apc_hit_s: float = 0.0
    restore_count: int = 0
    restore_bytes: int = 0
    min_prefix_index: int | None = None
    max_prefix_chain_len: int = 0


@dataclass
class BlockValuePolicyStats:
    prefix_cache_query_count: int = 0
    prefix_cache_zero_hit_count: int = 0
    effective_prefix_hit_blocks_sum: int = 0
    effective_prefix_hit_tokens_sum: int = 0
    reorder_attempt_count: int = 0
    reorder_count: int = 0
    reorder_time_s: float = 0.0
    reorder_scored_blocks: int = 0
    reorder_skipped_low_usage_count: int = 0
    reorder_skipped_no_cached_evict_count: int = 0
    evicted_blocks: int = 0
    evicted_value_score_sum: float = 0.0
    evicted_owner_count_sum: int = 0
    evicted_access_count_sum: int = 0
    evicted_apc_hit_count_sum: int = 0
    evicted_restore_count_sum: int = 0
    evicted_prefix_index_sum: int = 0
    evicted_prefix_index_count: int = 0
    evicted_chain_downstream_blocks_sum: int = 0
    evicted_chain_impact_sum: float = 0.0
    evicted_seconds_since_last_apc_hit_sum: float = 0.0
    evicted_seconds_since_last_apc_hit_count: int = 0


_VALUE_EVICTION_POLICIES = {
    "lru",
    "prefix_only",
    "hotness_only",
    "hybrid",
    "chain_hybrid",
}


class BlockHashToBlockMap:
    """
    Cache of blocks that are used for prefix caching. It caches blocks
    from hash directly to a block or multiple blocks
    (i.e. {block_hash: KVCacheBlocks})
    - Mostly block_hash maps to a single KVCacheBlock, and KVCacheBlocks
        would simply be a KVCacheBlock.
    - Otherwise, KVCacheBlocks is a dict from {block_id: KVCacheBlock}

    A cached block is a full block with a block hash that can be used
    for prefix caching.
    The cached block may be used by running requests or in the
    free_block_queue that could potentially be evicted.

    NOTE #1: We currently don't de-duplicate the blocks in the cache,
    meaning that if a block becomes full and is cached, we don't check
    if there is already an identical block in the cache. This is because
    we want to make sure the allocated block IDs won't change so that
    block tables are append-only.
    NOTE #2: The union type is introduced in order to reduce GC costs
    from the inner dict.
    """

    def __init__(self):
        self._cache: dict[
            BlockHashWithGroupId, KVCacheBlock | dict[int, KVCacheBlock]
        ] = {}

    def get_one_block(self, key: BlockHashWithGroupId) -> KVCacheBlock | None:
        """
        Gets any block with the given block hash key.
        """
        blocks = self._cache.get(key)
        if blocks is not None:
            if isinstance(blocks, KVCacheBlock):
                return blocks
            if isinstance(blocks, dict):
                return next(iter(blocks.values()))
            self._unexpected_blocks_type(blocks)
        return None

    def insert(self, key: BlockHashWithGroupId, block: KVCacheBlock) -> None:
        """
        Inserts the KVCacheBlock to the cache
        """
        blocks = self._cache.get(key)
        if blocks is None:
            # When key is not found, attach a single block to the key
            self._cache[key] = block
        elif isinstance(blocks, KVCacheBlock):
            # If there's a block with the same key, merge the original block
            # and the new block into a dict
            self._cache[key] = {blocks.block_id: blocks, block.block_id: block}
        elif isinstance(blocks, dict):
            # If it's already a dict, simply insert the block
            blocks[block.block_id] = block
        else:
            self._unexpected_blocks_type(blocks)

    def pop(self, key: BlockHashWithGroupId, block_id: int) -> KVCacheBlock | None:
        """
        Checks if block_hash exists and pop block_id from the cache
        """
        blocks = self._cache.pop(key, None)
        if blocks is None:
            # block_hash not found in the cache
            return None
        # TODO(Jialin): If key is found, block_id should always present
        # in blocks. We currently keep the original behaviour for safety.
        #
        # Will add block_id == blocks.block_id assertion and
        # use del blocks[block_id] instead as followup.
        if isinstance(blocks, KVCacheBlock):
            if blocks.block_id == block_id:
                return blocks
            # If the single block ID doesn't match, we should put the
            # block back (it should happen rarely)
            self._cache[key] = blocks
            return None
        if isinstance(blocks, dict):
            # Try to pop block_id from the block dict, and if dict still
            # contain blocks, put back to the cache.
            block = blocks.pop(block_id, None)
            if len(blocks) > 0:
                self._cache[key] = blocks
            return block
        self._unexpected_blocks_type(blocks)
        return None

    def contains(self, key: BlockHashWithGroupId, block_id: int) -> bool:
        blocks = self._cache.get(key)
        if blocks is None:
            return False
        if isinstance(blocks, KVCacheBlock):
            return blocks.block_id == block_id
        if isinstance(blocks, dict):
            return block_id in blocks
        self._unexpected_blocks_type(blocks)
        return False

    def __len__(self) -> int:
        return len(self._cache)

    def _unexpected_blocks_type(self, blocks: Any) -> None:
        raise AssertionError(f"Invalid KV cache block type {type(blocks)}")


class BlockPool:
    """BlockPool that manages KVCacheBlocks.
    It provides methods to allocate, free and cache the kv cache blocks. The
    free_block_queue stores the free blocks in eviction order to enable
    allocation, free, and cache eviction. The cached_block_hash_to_block
    maps between block hash and cached block to support finding cached blocks
    by their block hash.

    Args:
        num_gpu_blocks: The number of blocks in the pool.
        enable_caching: Whether to enable prefix caching.
        hash_block_size: The block size of which the block hashes are computed.
            The actual block size usually equals hash_block_size, but in cases
            where different KV cache groups have different block sizes, the
            actual block size can be a multiple of hash_block_size.
        enable_kv_cache_events: Whether to enable kv cache events.
        metrics_collector: Optional metrics collector for tracking block residency.
    """

    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool,
        hash_block_size: int,
        enable_kv_cache_events: bool = False,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ):
        assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching
        self.hash_block_size = hash_block_size
        # All kv-cache blocks.
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_gpu_blocks)
        ]
        # Free block queue that constructs and manipulates a doubly linked
        # list of free blocks (including eviction candidates when caching is
        # enabled).
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)

        # Cache for block lookup
        self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()

        # To represent a placeholder block with block_id=0.
        # The ref_cnt of null_block is not maintained, needs special care to
        # avoid freeing it.
        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True

        self.enable_kv_cache_events = enable_kv_cache_events
        self.kv_event_queue: list[KVCacheEvent] = []

        policy = os.getenv("VLLM_EXPERIMENTAL_GPU_EVICTION_POLICY", "lru")
        self.value_eviction_policy = policy.strip().lower()
        if self.value_eviction_policy not in _VALUE_EVICTION_POLICIES:
            logger.warning(
                "Unknown VLLM_EXPERIMENTAL_GPU_EVICTION_POLICY=%s; using lru",
                policy,
            )
            self.value_eviction_policy = "lru"
        self._value_eviction_enabled = (
            self.enable_caching and self.value_eviction_policy != "lru"
        )
        self._block_value_stats: dict[BlockHashWithGroupId, BlockValueStats] = {}
        self._value_share_weight = self._read_float_env(
            "VLLM_EXPERIMENTAL_GPU_EVICTION_SHARE_WEIGHT", 1.0
        )
        self._value_hotness_weight = self._read_float_env(
            "VLLM_EXPERIMENTAL_GPU_EVICTION_HOTNESS_WEIGHT", 1.0
        )
        self._value_restore_weight = self._read_float_env(
            "VLLM_EXPERIMENTAL_GPU_EVICTION_RESTORE_WEIGHT", 1.0
        )
        self._value_chain_weight = self._read_float_env(
            "VLLM_EXPERIMENTAL_GPU_EVICTION_CHAIN_WEIGHT", 1.0
        )
        self._value_hotness_half_life_s = max(
            self._read_float_env(
                "VLLM_EXPERIMENTAL_GPU_EVICTION_HOTNESS_HALF_LIFE_S", 30.0
            ),
            1e-6,
        )
        self._value_min_usage = min(
            max(
                self._read_float_env(
                    "VLLM_EXPERIMENTAL_GPU_EVICTION_MIN_USAGE", 0.20
                ),
                0.0,
            ),
            1.0,
        )
        self._value_candidate_window = max(
            1,
            int(
                self._read_float_env(
                    "VLLM_EXPERIMENTAL_GPU_EVICTION_CANDIDATE_WINDOW",
                    64.0,
                )
            ),
        )
        self._value_request_owner_map = self._read_owner_map_env()
        self._value_policy_stats = BlockValuePolicyStats()
        if self._value_eviction_enabled:
            logger.info(
                "Experimental GPU value eviction enabled: "
                "policy=%s min_usage=%.2f share_weight=%.2f "
                "hotness_weight=%.2f restore_weight=%.2f "
                "chain_weight=%.2f candidate_window=%d",
                self.value_eviction_policy,
                self._value_min_usage,
                self._value_share_weight,
                self._value_hotness_weight,
                self._value_restore_weight,
                self._value_chain_weight,
                self._value_candidate_window,
            )

        self.metrics_collector = metrics_collector

    @staticmethod
    def _read_float_env(name: str, default: float) -> float:
        value = os.getenv(name)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError:
            logger.warning("Invalid %s=%s; using %s", name, value, default)
            return default

    @staticmethod
    def _read_owner_map_env() -> dict[str, str]:
        value = os.getenv("VLLM_EXPERIMENTAL_GPU_EVICTION_REQUEST_OWNER_MAP")
        if not value:
            return {}
        try:
            data = json.loads(value)
        except json.JSONDecodeError:
            logger.warning(
                "Invalid VLLM_EXPERIMENTAL_GPU_EVICTION_REQUEST_OWNER_MAP; "
                "falling back to request_id ownership"
            )
            return {}
        if not isinstance(data, dict):
            logger.warning(
                "VLLM_EXPERIMENTAL_GPU_EVICTION_REQUEST_OWNER_MAP must be a JSON "
                "object; falling back to request_id ownership"
            )
            return {}
        return {
            str(request_id): str(owner_id)
            for request_id, owner_id in data.items()
            if owner_id is not None
        }

    def _value_owner_id(self, request_id: str | None) -> str | None:
        if not request_id:
            return None
        owner_id = self._value_request_owner_map.get(request_id)
        if owner_id is not None:
            return owner_id

        external_request_id, sep, suffix = request_id.rpartition("-")
        if sep and len(suffix) == 8 and all(c in "0123456789abcdef" for c in suffix):
            owner_id = self._value_request_owner_map.get(external_request_id)
            if owner_id is not None:
                return owner_id
            return external_request_id
        return request_id

    def _get_block_value_stats(
        self, block_hash: BlockHashWithGroupId
    ) -> BlockValueStats:
        return self._block_value_stats.setdefault(block_hash, BlockValueStats())

    def _record_block_store(
        self,
        block_hash: BlockHashWithGroupId | None,
        request_id: str | None,
        prefix_index: int,
        prefix_chain_len: int,
    ) -> None:
        if block_hash is None:
            return
        stats = self._get_block_value_stats(block_hash)
        owner_id = self._value_owner_id(request_id)
        if owner_id:
            stats.request_ids.add(owner_id)
        stats.store_count += 1
        if stats.min_prefix_index is None:
            stats.min_prefix_index = prefix_index
        else:
            stats.min_prefix_index = min(stats.min_prefix_index, prefix_index)
        stats.max_prefix_chain_len = max(
            stats.max_prefix_chain_len, prefix_chain_len
        )

    def _record_block_access(
        self, block_hash: BlockHashWithGroupId | None, request_id: str | None
    ) -> None:
        if block_hash is None:
            return
        stats = self._get_block_value_stats(block_hash)
        owner_id = self._value_owner_id(request_id)
        if owner_id:
            stats.request_ids.add(owner_id)
        stats.access_count += 1
        stats.apc_hit_count += 1
        now_s = time.monotonic()
        stats.last_access_s = now_s
        stats.last_apc_hit_s = now_s

    def _record_block_restore(
        self,
        block_hash: BlockHashWithGroupId | None,
        request_id: str | None,
        restore_bytes: int = 0,
    ) -> None:
        if block_hash is None:
            return
        stats = self._get_block_value_stats(block_hash)
        owner_id = self._value_owner_id(request_id)
        if owner_id:
            stats.request_ids.add(owner_id)
        stats.restore_count += 1
        stats.restore_bytes += max(restore_bytes, 0)
        stats.last_access_s = time.monotonic()

    @staticmethod
    def _block_chain_downstream_blocks(stats: BlockValueStats) -> int:
        if stats.min_prefix_index is None:
            return 0
        return max(
            0, stats.max_prefix_chain_len - stats.min_prefix_index - 1
        )

    @classmethod
    def _block_chain_impact_score(cls, stats: BlockValueStats) -> float:
        downstream_blocks = cls._block_chain_downstream_blocks(stats)
        return math.log1p(downstream_blocks)

    def _block_value_score(self, block: KVCacheBlock, now_s: float) -> float:
        block_hash = block.block_hash
        if block_hash is None:
            return 0.0

        stats = self._block_value_stats.get(block_hash)
        if stats is None:
            return 0.0

        share_score = math.log1p(len(stats.request_ids))
        if stats.last_apc_hit_s > 0.0:
            age_s = max(0.0, now_s - stats.last_apc_hit_s)
            hotness_score = stats.apc_hit_count * math.pow(
                0.5, age_s / self._value_hotness_half_life_s
            )
        else:
            hotness_score = 0.0
        chain_score = self._block_chain_impact_score(stats)
        restore_score = math.log1p(stats.restore_count) + math.log1p(
            stats.restore_bytes
        )

        if self.value_eviction_policy == "prefix_only":
            return share_score
        if self.value_eviction_policy == "hotness_only":
            return hotness_score

        hotness_boost = 1.0 + self._value_hotness_weight * math.log1p(
            hotness_score
        )
        restore_boost = 1.0 + self._value_restore_weight * math.log1p(
            restore_score
        )
        if self.value_eviction_policy == "chain_hybrid":
            chain_boost = 1.0 + self._value_chain_weight * chain_score
            return (
                self._value_share_weight
                * share_score
                * hotness_boost
                * chain_boost
                * restore_boost
            )
        return self._value_share_weight * share_score * hotness_boost * restore_boost

    def _usage_after_allocate(self, num_blocks_to_allocate: int) -> float:
        total_gpu_blocks = self.num_gpu_blocks - 1
        if total_gpu_blocks <= 0:
            return 0.0
        used_blocks_after_allocate = (
            total_gpu_blocks - self.free_block_queue.num_free_blocks
        ) + num_blocks_to_allocate
        return used_blocks_after_allocate / total_gpu_blocks

    def _should_reorder_free_queue_for_value_policy(
        self, num_blocks_to_allocate: int
    ) -> bool:
        if (
            not self._value_eviction_enabled
            or self.free_block_queue.num_free_blocks <= 1
        ):
            return False
        return (
            self._usage_after_allocate(num_blocks_to_allocate)
            >= self._value_min_usage
        )

    def _is_cached_eviction_candidate(self, block: KVCacheBlock) -> bool:
        block_hash = block.block_hash
        return (
            block_hash is not None
            and self.cached_block_hash_to_block.contains(
                block_hash, block.block_id
            )
        )

    def _maybe_reorder_free_queue_for_value_policy(
        self, num_blocks_to_allocate: int
    ) -> None:
        if (
            not self._value_eviction_enabled
            or self.free_block_queue.num_free_blocks <= 1
        ):
            return
        stats = self._value_policy_stats
        if not self._should_reorder_free_queue_for_value_policy(
            num_blocks_to_allocate
        ):
            stats.reorder_skipped_low_usage_count += 1
            return

        stats.reorder_attempt_count += 1
        start_s = time.perf_counter()
        candidate_window = min(
            self.free_block_queue.num_free_blocks,
            max(num_blocks_to_allocate, self._value_candidate_window),
        )
        free_blocks = self.free_block_queue.get_first_n_free_blocks(candidate_window)
        eviction_candidate_count = min(num_blocks_to_allocate, len(free_blocks))
        eviction_candidates = free_blocks[:eviction_candidate_count]
        if not any(
            self._is_cached_eviction_candidate(block)
            for block in eviction_candidates
        ):
            stats.reorder_skipped_no_cached_evict_count += 1
            return

        now_s = time.monotonic()
        scored_blocks = list(enumerate(free_blocks))
        scored_blocks.sort(
            key=lambda item: (self._block_value_score(item[1], now_s), item[0])
        )
        ordered_blocks = [block for _, block in scored_blocks]
        stats.reorder_scored_blocks += len(scored_blocks)
        stats.reorder_time_s += time.perf_counter() - start_s
        if ordered_blocks != free_blocks:
            self.free_block_queue.reorder_prefix(ordered_blocks)
            stats.reorder_count += 1

    def _record_cached_block_eviction(
        self, block: KVCacheBlock, block_hash: BlockHashWithGroupId
    ) -> None:
        if not self._value_eviction_enabled:
            return
        stats = self._block_value_stats.get(block_hash)
        policy_stats = self._value_policy_stats
        now_s = time.monotonic()
        policy_stats.evicted_blocks += 1
        policy_stats.evicted_value_score_sum += self._block_value_score(
            block, now_s
        )
        if stats is not None:
            policy_stats.evicted_owner_count_sum += len(stats.request_ids)
            policy_stats.evicted_access_count_sum += stats.access_count
            policy_stats.evicted_apc_hit_count_sum += stats.apc_hit_count
            policy_stats.evicted_restore_count_sum += stats.restore_count
            if stats.min_prefix_index is not None:
                policy_stats.evicted_prefix_index_sum += stats.min_prefix_index
                policy_stats.evicted_prefix_index_count += 1
            policy_stats.evicted_chain_downstream_blocks_sum += (
                self._block_chain_downstream_blocks(stats)
            )
            policy_stats.evicted_chain_impact_sum += (
                self._block_chain_impact_score(stats)
            )
            if stats.last_apc_hit_s > 0.0:
                policy_stats.evicted_seconds_since_last_apc_hit_sum += max(
                    0.0, now_s - stats.last_apc_hit_s
                )
                policy_stats.evicted_seconds_since_last_apc_hit_count += 1

    def record_prefix_cache_lookup(self, num_hit_tokens: int) -> None:
        stats = self._value_policy_stats
        stats.prefix_cache_query_count += 1
        stats.effective_prefix_hit_tokens_sum += num_hit_tokens
        stats.effective_prefix_hit_blocks_sum += (
            num_hit_tokens // self.hash_block_size
        )
        if num_hit_tokens == 0:
            stats.prefix_cache_zero_hit_count += 1

    def get_cached_block(
        self, block_hash: BlockHash, kv_cache_group_ids: list[int]
    ) -> list[KVCacheBlock] | None:
        """Get the cached block by the block hash for each group in
        `kv_cache_group_ids`, or None if cache miss for any group.
        If there are duplicated blocks, we return the first block in the cache.

        Args:
            block_hash: The hash value of the block.
            kv_cache_group_ids: The ids of the KV cache groups.

        Returns:
            The cached blocks if exists, or None.
        """
        # 开题阶段代码阅读：BlockPool 用 block_hash 查找 GPU 中
        # 已缓存的 KV block，这是 prefix hit 最底层的数据结构入口。
        cached_blocks = []
        for group_id in kv_cache_group_ids:
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, group_id
            )
            block = self.cached_block_hash_to_block.get_one_block(
                block_hash_with_group_id
            )
            if not block:
                return None
            cached_blocks.append(block)
        return cached_blocks

    def cache_full_blocks(
        self,
        request: Request,
        blocks: list[KVCacheBlock],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
        restored_block_ids: set[int] | None = None,
    ) -> None:
        """Cache a list of full blocks for prefix caching.
        This function takes a list of blocks that will have their block hash
        metadata to be updated and cached. Given a request, it updates the
        metadata for each block and caching it in the
        `cached_block_hash_to_block`.
        The block hashes values are computed by the Request object immediately
        when it is created and when new tokens are appended.

        Args:
            request: The request to cache the blocks.
            blocks: All blocks in the request.
            num_cached_blocks: The number of blocks that are already cached.
            num_full_blocks: The number of blocks that are full and should
                be cached after this function.
            block_size: Number of tokens in each block.
            kv_cache_group_id: The id of the KV cache group.
        """
        # 开题阶段代码阅读：这个函数把已经算满的 blocks 注册进
        # prefix cache 的 hash table，之后其他请求才能通过 hash 命中。
        if num_cached_blocks >= num_full_blocks:
            return
        new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
        assert len(request.block_hashes) >= num_full_blocks
        if block_size == self.hash_block_size:
            # Common case.
            block_hashes: BlockHashList = request.block_hashes
        else:
            # block_size is a multiple of hash_block_size. This happens when
            # different KV cache groups have different block sizes.
            assert block_size % self.hash_block_size == 0
            # Recalculate block_hashes at the granularity of block_size, using
            # the original block_hashes (at the granularity of hash_block_size).
            block_hashes = BlockHashListWithBlockSize(
                request.block_hashes, self.hash_block_size, block_size
            )

        new_block_hashes = block_hashes[num_cached_blocks:]
        new_hashes: list[ExternalBlockHash] | None = (
            [] if self.enable_kv_cache_events else None
        )
        for i, blk in enumerate(new_full_blocks):
            # Some blocks may be null blocks when enabling sparse attention like
            # sliding window attention, or Mamba models with prefix-caching in
            # align mode. We skip null blocks here.
            if blk.is_null:
                continue
            assert blk.block_hash is None
            block_hash = new_block_hashes[i]

            # 开题阶段代码阅读：这里给 full block 绑定 block_hash，
            # 让它从普通 GPU block 变成可被 prefix cache 复用的 block。
            # Update and added the full block to the cache.
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, kv_cache_group_id
            )
            blk.block_hash = block_hash_with_group_id
            self.cached_block_hash_to_block.insert(block_hash_with_group_id, blk)
            self._record_block_store(
                block_hash_with_group_id,
                request.request_id,
                prefix_index=num_cached_blocks + i,
                prefix_chain_len=num_full_blocks,
            )
            if restored_block_ids and blk.block_id in restored_block_ids:
                self._record_block_restore(
                    block_hash_with_group_id, request.request_id
                )
            if new_hashes is not None:
                new_hashes.append(maybe_convert_block_hash(block_hash))

        if self.enable_kv_cache_events:
            if num_cached_blocks == 0:
                parent_block_hash: ExternalBlockHash | None = None
            else:
                parent_block_hash = maybe_convert_block_hash(
                    block_hashes[num_cached_blocks - 1]
                )

            # Calculate token range for the blocks being cached
            start_token_idx = num_cached_blocks * block_size
            end_token_idx = num_full_blocks * block_size

            # Generate extra keys for each block individually.
            # Each block may have different extra_keys (e.g., different MM
            # features, or cache_salt only for the first block).
            # Skip null blocks to match the length of new_hashes.
            extra_keys_list: list[tuple[Any, ...] | None] = []
            curr_mm_idx = 0
            for i in range(num_cached_blocks, num_full_blocks):
                if blocks[i].is_null:
                    continue
                block_start = i * block_size
                block_end = block_start + block_size
                extra_keys, curr_mm_idx = generate_block_hash_extra_keys(
                    request, block_start, block_end, curr_mm_idx
                )
                extra_keys_list.append(extra_keys)

            self.kv_event_queue.append(
                BlockStored(
                    block_hashes=new_hashes,
                    parent_block_hash=parent_block_hash,
                    token_ids=request.all_token_ids[start_token_idx:end_token_idx],
                    block_size=block_size,
                    lora_id=request.lora_request.adapter_id
                    if request.lora_request
                    else None,
                    medium=MEDIUM_GPU,
                    lora_name=request.lora_request.name
                    if request.lora_request
                    else None,
                    extra_keys=extra_keys_list if extra_keys_list else None,
                )
            )

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        """Get new blocks from the free block pool.

        Note that we do not check block cache in this function.

        Args:
            num_blocks: The number of blocks to allocate.

        Returns:
            A list of new block.
        """
        if num_blocks > self.get_num_free_blocks():
            raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")

        # 开题阶段代码阅读：新 block 从 free_block_queue 头部取出；
        # 因此 free queue 的排序实际决定了哪些 cached blocks 会优先被挤掉。
        self._maybe_reorder_free_queue_for_value_policy(num_blocks)
        ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)

        # In order to only iterate the list once, we duplicated code a bit
        if self.enable_caching:
            for block in ret:
                self._maybe_evict_cached_block(block)
                assert block.ref_cnt == 0
                block.ref_cnt += 1
                if self.metrics_collector:
                    self.metrics_collector.on_block_allocated(block)
        else:
            for block in ret:
                assert block.ref_cnt == 0
                block.ref_cnt += 1
                if self.metrics_collector:
                    self.metrics_collector.on_block_allocated(block)
        return ret

    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
        """
        If a block is cached in `cached_block_hash_to_block`, we reset its hash
        metadata and evict it from the cache.

        Args:
            block: The block to evict.

        Returns:
            True if the block is evicted, False otherwise.
        """
        # 开题阶段代码阅读：当 free queue 中的 cached block 被重新
        # 分配时，会在这里清除 hash 元数据，相当于从 prefix cache 淘汰。
        # Clean up metrics tracking first to prevent leaks
        if self.metrics_collector:
            self.metrics_collector.on_block_evicted(block)

        block_hash = block.block_hash
        if block_hash is None:
            # The block doesn't have hash, eviction is not needed
            return False

        if self.cached_block_hash_to_block.pop(block_hash, block.block_id) is None:
            # block not found in cached_block_hash_to_block,
            # eviction is not needed
            return False

        self._record_cached_block_eviction(block, block_hash)
        block.reset_hash()

        if self.enable_kv_cache_events:
            # FIXME (Chen): Not sure whether we should return `hash_value`
            # or `(hash_value, group_id)` here. But it's fine now because
            # we disable hybrid kv cache manager when kv cache event is
            # enabled, so there is only one group.
            self.kv_event_queue.append(
                BlockRemoved(
                    block_hashes=[maybe_convert_block_hash(get_block_hash(block_hash))],
                    medium=MEDIUM_GPU,
                )
            )
        return True

    def touch(
        self, blocks: Sequence[KVCacheBlock], request_id: str | None = None
    ) -> None:
        """Touch a block increases its reference count by 1, and may remove
        the block from the free queue. This is used when a block is hit by
        another request with the same prefix.

        Args:
            blocks: A list of blocks to touch.
        """
        # 开题阶段代码阅读：prefix hit 后会 touch block，提高引用计数
        # 并把 ref_cnt=0 的块移出 free queue；这里可对应 hotness/access 信号。
        for block in blocks:
            self._record_block_access(block.block_hash, request_id)
            # ref_cnt=0 means this block is in the free list (i.e. eviction
            # candidate), so remove it.
            if block.ref_cnt == 0 and not block.is_null:
                self.free_block_queue.remove(block)
            block.ref_cnt += 1
            if self.metrics_collector:
                self.metrics_collector.on_block_accessed(block)

    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        """Free a list of blocks. The blocks should be ordered by their
        eviction priority, where the first block will be evicted first.

        Args:
            ordered_blocks: A list of blocks to free ordered by their eviction
                priority.
        """
        # 开题阶段代码阅读：请求释放 blocks 时，ref_cnt 归零的块会
        # 进入 free queue 成为 eviction candidates；这是 retention policy
        # 最直接的可观察位置。
        # Materialize the iterable to allow multiple passes.
        blocks_list = list(ordered_blocks)
        for block in blocks_list:
            block.ref_cnt -= 1
        self.free_block_queue.append_n(
            [block for block in blocks_list if block.ref_cnt == 0 and not block.is_null]
        )

    def evict_blocks(self, block_ids: set[int]) -> None:
        """evict blocks from the prefix cache by their block IDs.

        only evicts blocks that are currently cached (have a hash). blocks
        with ref_cnt > 0 are not freed from the block pool, only evicted
        from the prefix cache hash table.

        Args:
            block_ids: Set of block IDs to evict from cache.
        """
        for block_id in block_ids:
            assert block_id < len(self.blocks), (
                f"Invalid block_id {block_id} >= {len(self.blocks)}. "
                f"This indicates a bug in the KV connector - workers should "
                f"only report block IDs that were allocated by the scheduler."
            )
            block = self.blocks[block_id]
            self._maybe_evict_cached_block(block)

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalid prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        num_used_blocks = self.num_gpu_blocks - self.get_num_free_blocks()
        if num_used_blocks != 1:  # The null block is always marked as used
            logger.warning(
                "Failed to reset prefix cache because some "
                "blocks (%d) are not freed yet",
                num_used_blocks - 1,
            )
            return False

        # Remove all hashes so that no new blocks will hit.
        self.cached_block_hash_to_block = BlockHashToBlockMap()
        self._block_value_stats.clear()

        # Remove all hashes from all blocks.
        for block in self.blocks:
            block.reset_hash()

        if self.metrics_collector:
            self.metrics_collector.reset()

        logger.info("Successfully reset prefix cache")

        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())

        return True

    def get_value_policy_stats(
        self, reset: bool = False
    ) -> dict[str, int | float | str]:
        stats = self._value_policy_stats
        snapshot: dict[str, int | float | str] = {
            "policy": self.value_eviction_policy,
            "min_usage": self._value_min_usage,
            "candidate_window": self._value_candidate_window,
            "chain_weight": self._value_chain_weight,
            "prefix_cache_query_count": stats.prefix_cache_query_count,
            "prefix_cache_zero_hit_count": stats.prefix_cache_zero_hit_count,
            "effective_prefix_hit_blocks_sum": (
                stats.effective_prefix_hit_blocks_sum
            ),
            "effective_prefix_hit_tokens_sum": (
                stats.effective_prefix_hit_tokens_sum
            ),
            "reorder_attempt_count": stats.reorder_attempt_count,
            "reorder_count": stats.reorder_count,
            "reorder_time_s": stats.reorder_time_s,
            "reorder_scored_blocks": stats.reorder_scored_blocks,
            "reorder_skipped_low_usage_count": (
                stats.reorder_skipped_low_usage_count
            ),
            "reorder_skipped_no_cached_evict_count": (
                stats.reorder_skipped_no_cached_evict_count
            ),
            "evicted_blocks": stats.evicted_blocks,
            "evicted_value_score_sum": stats.evicted_value_score_sum,
            "evicted_owner_count_sum": stats.evicted_owner_count_sum,
            "evicted_access_count_sum": stats.evicted_access_count_sum,
            "evicted_apc_hit_count_sum": stats.evicted_apc_hit_count_sum,
            "evicted_restore_count_sum": stats.evicted_restore_count_sum,
            "evicted_prefix_index_sum": stats.evicted_prefix_index_sum,
            "evicted_prefix_index_count": stats.evicted_prefix_index_count,
            "evicted_chain_downstream_blocks_sum": (
                stats.evicted_chain_downstream_blocks_sum
            ),
            "evicted_chain_impact_sum": stats.evicted_chain_impact_sum,
            "evicted_seconds_since_last_apc_hit_sum": (
                stats.evicted_seconds_since_last_apc_hit_sum
            ),
            "evicted_seconds_since_last_apc_hit_count": (
                stats.evicted_seconds_since_last_apc_hit_count
            ),
        }
        if reset:
            self._value_policy_stats = BlockValuePolicyStats()
        return snapshot

    def get_num_free_blocks(self) -> int:
        """Get the number of free blocks in the pool.

        Returns:
            The number of free blocks.
        """
        return self.free_block_queue.num_free_blocks

    def get_usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """

        # Subtract 1 to account for null block.
        total_gpu_blocks = self.num_gpu_blocks - 1
        if not total_gpu_blocks:
            return 0
        return 1.0 - (self.get_num_free_blocks() / total_gpu_blocks)

    def take_events(self) -> list[KVCacheEvent]:
        """Atomically takes all events and clears the queue.

        Returns:
            A list of KV cache events.
        """
        if not self.enable_kv_cache_events:
            return []
        events = self.kv_event_queue
        self.kv_event_queue = []
        return events
