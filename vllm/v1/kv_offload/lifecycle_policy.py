# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental, opt-in lifecycle utility policy; all costs are seconds.

Counterfactual savings and store charges are estimates, not causal measurements.
No CUDA calls are made on the scheduler thread. Context uses scheduler proxies.
"""

import math
import time
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, fields
from typing import Any

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import OffloadingManager, PrepareStoreOutput
from vllm.v1.kv_offload.reuse_manager import FilterReusedOffloadingManager


@dataclass(frozen=True)
class LifecyclePolicyConfig:
    # Required calibration priors. Never silently assume a model/hardware scale.
    store_s_per_block: float
    restore_s_per_block: float
    recompute_s_per_token: float
    restore_fixed_s: float = 0.0
    capacity_s_per_block: float = 0.0
    demand_prior: float = 0.25
    survival_discount: float = 0.5
    reuse_horizon_s: float = 60.0
    admission_margin_s: float = 0.0
    promotion_margin_s: float = 0.0
    ewma_alpha: float = 0.2
    min_cost_samples: int = 3
    max_tracker_size: int = 64000
    max_cost_buckets: int = 512
    max_decision_events: int = 256
    admission_enabled: bool = True
    # Independent ablations; defaults preserve the existing policy.
    new_write_store_cost: bool = False
    trace_admission: bool = False
    guard_enabled: bool = True
    guard_window_s: float = 30.0
    guard_min_samples: int = 8
    guard_epsilon_s: float = 0.001
    throttle_every: int = 4
    probe_every: int = 64
    probe_max_blocks: int = 4
    # Optional finite warmup budget, independent of OFF-state recovery probes.
    cold_probe_every: int = 0
    cold_probe_min_blocks: int = 16
    cold_probe_max_blocks: int = 64
    cold_probe_budget_blocks: int = 128
    cold_probe_budget_s: float = 0.1

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "LifecyclePolicyConfig":
        if not isinstance(values, dict):
            raise ValueError("lifecycle_policy must be a JSON object")
        known = {f.name for f in fields(cls)}
        if unknown := set(values) - known:
            raise ValueError(f"Unknown lifecycle_policy fields: {sorted(unknown)}")
        required = {"store_s_per_block", "restore_s_per_block", "recompute_s_per_token"}
        if missing := required - set(values):
            raise ValueError(f"Missing lifecycle cost priors: {sorted(missing)}")
        return cls(**values)

    def __post_init__(self):
        positive_ints = {
            "min_cost_samples",
            "max_tracker_size",
            "max_cost_buckets",
            "max_decision_events",
            "guard_min_samples",
            "throttle_every",
            "probe_every",
            "probe_max_blocks",
            "cold_probe_min_blocks",
            "cold_probe_max_blocks",
            "cold_probe_budget_blocks",
        }
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name in {
                "admission_enabled",
                "guard_enabled",
                "new_write_store_cost",
                "trace_admission",
            }:
                if type(value) is not bool:
                    raise ValueError(f"{f.name} must be boolean")
            elif f.name == "cold_probe_every":
                if type(value) is not int or value < 0:
                    raise ValueError("cold_probe_every must be a nonnegative integer")
            elif f.name in positive_ints:
                if type(value) is not int or value < 1:
                    raise ValueError(f"{f.name} must be a positive integer")
            elif (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{f.name} must be finite and nonnegative")
        for name in ("reuse_horizon_s", "guard_window_s", "recompute_s_per_token"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.restore_fixed_s + self.restore_s_per_block <= 0:
            raise ValueError("At least one restore cost prior must be positive")
        for name in ("demand_prior", "survival_discount", "ewma_alpha"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.cold_probe_min_blocks > self.cold_probe_max_blocks:
            raise ValueError("cold_probe_min_blocks exceeds cold_probe_max_blocks")


@dataclass
class PendingPromotion:
    action: str
    blocks: int
    tokens: int
    target: int
    context: tuple[int, int]
    restore_s: float
    recompute_s: float
    started: float


class LifecycleOffloadingManager(FilterReusedOffloadingManager):
    """Batch-prefix admission plus shared online promotion/guard state.

    Store batches arrive during execution, not exclusively at GPU eviction.
    Costs for a candidate prefix are joint estimates (not summed singleton
    prefill costs). The longest profitable prefix is eligible for storage.
    This does not guarantee retention of earlier chunks or CPU ancestors.
    """

    def __init__(
        self,
        backing: OffloadingManager,
        config: LifecyclePolicyConfig,
        block_tokens: int,
        *,
        clock=time.perf_counter,
    ):
        if type(block_tokens) is not int or block_tokens < 1:
            raise ValueError("block_tokens must be a positive integer")
        super().__init__(
            backing,
            store_threshold=0,
            max_tracker_size=config.max_tracker_size,
            track_lifecycle_stats=True,
        )
        self.config = config
        self.block_tokens = block_tokens
        self.clock = clock
        self.context = (1, 0)  # active requests, in-flight transfer requests
        self.history: OrderedDict[BlockHash, tuple[int, float]] = OrderedDict()
        self.seen_requests: OrderedDict[str, None] = OrderedDict()
        self.costs: OrderedDict[tuple, tuple[float, int]] = OrderedDict()
        self.pending: dict[str, PendingPromotion] = {}
        self.events: deque[dict[str, Any]] = deque(maxlen=config.max_decision_events)
        self.counters: Counter = Counter()
        self.guard_state = "NORMAL"
        self.window_start = self.clock()
        self.window_utility_s = 0.0
        self.window_samples = 0
        self.guard_ewma_s: float | None = None
        self.admission_index = 0
        self.cold_probe_candidates = 0
        self.cold_probe_blocks_spent = 0
        self.cold_probe_cost_spent_s = 0.0
        self._selected_cold_probe = False
        self.last_admission: dict[str, Any] | None = None

    def observe_request(self, request_id: str, hashes) -> None:
        if request_id in self.seen_requests:
            return
        self.seen_requests[request_id] = None
        if len(self.seen_requests) > self.config.max_tracker_size:
            self.seen_requests.popitem(last=False)
            self.counters["request_tracker_evictions"] += 1
        now = self.clock()
        for bh in hashes:
            count, _ = self.history.pop(bh, (0, now))
            self.history[bh] = (count + 1, now)
            if len(self.history) > self.config.max_tracker_size:
                self.history.popitem(last=False)
                self.counters["reuse_tracker_evictions"] += 1

    def probability(self, bh: BlockHash) -> float:
        count, last = self.history.get(bh, (0, self.clock()))
        c = self.config
        repeat = 1.0 - math.exp(-max(0, count - 1))
        recent = math.exp(-max(0.0, self.clock() - last) / c.reuse_horizon_s)
        return c.survival_discount * (
            c.demand_prior + (1 - c.demand_prior) * repeat * recent
        )

    @staticmethod
    def _bucket(value: int) -> int:
        return max(0, value).bit_length()

    def _key(self, action: str, blocks: int, tokens: int, context: tuple[int, int]):
        return (
            action,
            self._bucket(blocks),
            self._bucket(tokens),
            self._bucket(context[0]),
            self._bucket(context[1]),
        )

    def _prior(self, action: str, blocks: int, tokens: int) -> float:
        c = self.config
        if action == "restore":
            return c.restore_fixed_s + c.restore_s_per_block * blocks
        return c.recompute_s_per_token * tokens

    def estimate(
        self,
        action: str,
        blocks: int,
        tokens: int,
        context: tuple[int, int] | None = None,
    ) -> float:
        key = self._key(action, blocks, tokens, context or self.context)
        factor, samples = self.costs.get(key, (1.0, 0))
        if samples < self.config.min_cost_samples:
            factor = 1.0
        return self._prior(action, blocks, tokens) * factor

    def _record(self, **event) -> None:
        if len(self.events) == self.events.maxlen:
            self.counters["decision_events_dropped"] += 1
        self.events.append(event)

    def _guard_tick(self):
        c = self.config
        if self.clock() - self.window_start < c.guard_window_s:
            return
        if c.guard_enabled and self.window_samples >= c.guard_min_samples:
            old = self.guard_state
            value = self.window_utility_s
            self.guard_ewma_s = (
                value
                if self.guard_ewma_s is None
                else (c.ewma_alpha * value + (1 - c.ewma_alpha) * self.guard_ewma_s)
            )
            states = ["NORMAL", "THROTTLE", "OFF"]
            idx = states.index(old)
            if self.guard_ewma_s < -c.guard_epsilon_s:
                idx = min(2, idx + 1)
            elif self.guard_ewma_s > c.guard_epsilon_s:
                idx = max(0, idx - 1)
            self.guard_state = states[idx]
            if old != self.guard_state:
                self.counters["guard_transitions"] += 1
                self._record(
                    kind="guard",
                    old=old,
                    new=self.guard_state,
                    estimated_utility_s=self.guard_ewma_s,
                )
        self.window_start = self.clock()
        self.window_samples = 0
        self.window_utility_s = 0.0

    def _utility(self, delta: float):
        self.window_utility_s += delta
        self.window_samples += 1
        self.counters["estimated_net_utility_s"] += delta

    def _select_store_blocks(self, block_hashes: list[BlockHash]) -> list[BlockHash]:
        self._selected_cold_probe = False
        self.last_admission = None
        started = self.clock()
        self._guard_tick()
        if not block_hashes:
            return []
        c = self.config
        states = (
            self.get_block_residency(block_hashes)
            if c.new_write_store_cost or c.trace_admission
            else None
        )
        # Unknown residency is charged conservatively. Capacity remains the
        # original prefix-retention proxy; this ablation changes STORE only.
        new_counts = [0]
        if states is not None:
            for state in states:
                new_counts.append(new_counts[-1] + int(state in {"missing", "unknown"}))

        def charge(n: int) -> float:
            if not c.new_write_store_cost:
                return n * (c.store_s_per_block + c.capacity_s_per_block)
            return new_counts[n] * c.store_s_per_block + n * c.capacity_s_per_block

        self.counters["admission_batches"] += 1
        self.admission_index += 1
        index = self.admission_index
        every = c.probe_every if self.guard_state == "OFF" else c.throttle_every
        allowed = self.guard_state == "NORMAL" or index % every == 0
        limit = len(block_hashes)
        if self.guard_state == "OFF":
            limit = min(limit, c.probe_max_blocks)
        best, best_gain, probability_sum = 0, 0.0, 0.0
        max_candidate_gain = None
        reason = "guard_drop"
        if allowed:
            for n, bh in enumerate(block_hashes[:limit], 1):
                probability_sum += self.probability(bh)
                tokens = n * self.block_tokens
                cr = self.estimate("restore", n, tokens)
                cc = self.estimate("recompute", n, tokens)
                gain = probability_sum / n * max(0.0, cc - cr) - charge(n)
                max_candidate_gain = (
                    gain
                    if max_candidate_gain is None
                    else max(max_candidate_gain, gain)
                )
                if not c.admission_enabled or gain > c.admission_margin_s:
                    best, best_gain = n, gain
            reason = (
                ("admission_disabled" if not c.admission_enabled else "profit")
                if best
                else "cost_drop"
            )
            # A bounded probe can refresh negative/cold models even when G<=0.
            if self.guard_state == "OFF" and best == 0:
                best = min(limit, c.probe_max_blocks)
            if self.guard_state == "OFF":
                self.counters["probe_batches"] += 1
                reason = "off_probe"
            if (
                best == 0
                and self.guard_state == "NORMAL"
                and c.admission_enabled
                and c.cold_probe_every
                and len(block_hashes) >= c.cold_probe_min_blocks
            ):
                self.cold_probe_candidates += 1
                if self.cold_probe_candidates % c.cold_probe_every == 0:
                    unit_cost = c.store_s_per_block + c.capacity_s_per_block
                    remaining_cost = max(
                        0.0, c.cold_probe_budget_s - self.cold_probe_cost_spent_s
                    )
                    budget_blocks = (
                        c.cold_probe_budget_blocks - self.cold_probe_blocks_spent
                    )
                    if unit_cost > 0:
                        budget_blocks = min(
                            budget_blocks, int(remaining_cost / unit_cost)
                        )
                    probe_size = min(
                        len(block_hashes), c.cold_probe_max_blocks, budget_blocks
                    )
                    if probe_size >= c.cold_probe_min_blocks:
                        best = probe_size
                        self._selected_cold_probe = True
                        reason = "cold_probe"
                        self.counters["cold_probe_batches"] += 1
                    else:
                        self.counters["cold_probe_budget_denials"] += 1
            if reason in {"cold_probe", "off_probe"} and best:
                tokens = best * self.block_tokens
                best_gain = sum(
                    self.probability(bh) for bh in block_hashes[:best]
                ) / best * max(
                    0.0,
                    self.estimate("recompute", best, tokens)
                    - self.estimate("restore", best, tokens),
                ) - charge(best)
        event = dict(
            kind="admission",
            candidates=len(block_hashes),
            selected=best,
            action="offload" if best else "drop",
            gain_s=best_gain if best else None,
            max_candidate_gain_s=max_candidate_gain,
            reason=reason,
            guard=self.guard_state,
            context=self.context,
            store_cost_basis="new_write" if c.new_write_store_cost else "candidate",
        )
        if states is not None:
            event.update(
                candidate_new_blocks=new_counts[-1],
                selected_new_blocks=new_counts[best],
            )
        self._record(**event)
        if c.trace_admission:
            self.last_admission = {**event, "candidate_states": states}
        self.counters["decision_time_s"] += self.clock() - started
        return block_hashes[:best]

    def prepare_store(self, block_hashes) -> PrepareStoreOutput | None:
        output = super().prepare_store(block_hashes)
        if output is not None and output.block_hashes_to_store:
            # Charge only newly reserved blocks, not already-present candidates.
            n = len(output.block_hashes_to_store)
            charge = n * (
                self.config.store_s_per_block + self.config.capacity_s_per_block
            )
            self.counters["estimated_store_charge_s"] += charge
            self._utility(-charge)
            if self._selected_cold_probe:
                self.cold_probe_blocks_spent += n
                self.cold_probe_cost_spent_s += charge
                self.counters["cold_probe_stored_blocks"] += n
        return output

    def choose(
        self,
        request_id: str,
        blocks: int,
        tokens: int,
        target: int,
        forced_action: str | None = None,
    ) -> str:
        started = self.clock()
        self._guard_tick()
        existing = self.pending.get(request_id)
        if existing is not None and existing.target == target:
            return existing.action
        if existing is not None:
            self.complete_promotion(request_id, cancelled=True)
        cr = self.estimate("restore", blocks, tokens)
        cc = self.estimate("recompute", blocks, tokens)
        action = forced_action or (
            "restore" if cr + self.config.promotion_margin_s < cc else "recompute"
        )
        self.pending[request_id] = PendingPromotion(
            action, blocks, tokens, target, self.context, cr, cc, started
        )
        self.counters[f"{action}_decisions"] += 1
        self._record(
            kind="promotion",
            request_id=request_id,
            target=target,
            action=action,
            blocks=blocks,
            tokens=tokens,
            restore_s=cr,
            recompute_s=cc,
            context=self.context,
        )
        self.counters["decision_time_s"] += self.clock() - started
        return action

    def complete_promotion(
        self,
        request_id: str,
        *,
        computed_tokens: int | None = None,
        cancelled: bool = False,
    ) -> None:
        pending = self.pending.get(request_id)
        if pending is None:
            return
        if (
            pending.action == "recompute"
            and not cancelled
            and (computed_tokens is None or computed_tokens < pending.target)
        ):
            return
        self.pending.pop(request_id)
        if cancelled:
            self.counters["incomplete_promotions"] += 1
            return
        elapsed = max(0.0, self.clock() - pending.started)
        key = self._key(pending.action, pending.blocks, pending.tokens, pending.context)
        prior = self._prior(pending.action, pending.blocks, pending.tokens)
        if prior > 0:
            ratio = elapsed / prior
            old, samples = self.costs.pop(key, (ratio, 0))
            alpha = self.config.ewma_alpha
            self.costs[key] = ((1 - alpha) * old + alpha * ratio, samples + 1)
            if len(self.costs) > self.config.max_cost_buckets:
                self.costs.popitem(last=False)
                self.counters["cost_bucket_evictions"] += 1
        credit = pending.recompute_s - elapsed if pending.action == "restore" else 0.0
        self._utility(credit)
        self.counters[f"{pending.action}_observations"] += 1
        self._record(
            kind="completion",
            request_id=request_id,
            target=pending.target,
            action=pending.action,
            observed_wait_s=elapsed,
            estimated_saved_s=credit,
        )

    def get_value_policy_stats(self, reset=False) -> dict[str, Any]:
        snapshot = super().get_value_policy_stats(reset)
        snapshot.update({f"lcuc_{k}": v for k, v in self.counters.items()})
        snapshot.update(
            lcuc_enabled=1,
            lcuc_guard_state=self.guard_state,
            lcuc_cost_buckets=len(self.costs),
            lcuc_pending=len(self.pending),
            lcuc_history_size=len(self.history),
            lcuc_cold_probe_blocks_spent=self.cold_probe_blocks_spent,
            lcuc_cold_probe_cost_spent_s=self.cold_probe_cost_spent_s,
            lcuc_events=list(self.events),
            lcuc_utility_semantics="estimated_counterfactual_minus_observed_restore",
        )
        if reset:
            # A reporting reset must not reopen the guard or discard in-flight work.
            self.counters.clear()
            self.events.clear()
        return snapshot
