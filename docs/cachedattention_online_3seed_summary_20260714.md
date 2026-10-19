# CachedAttention Online Value Policy Summary

Date: 2026-07-14

This note summarizes the current online multi-turn value-policy results after
fixing two measurement issues:

1. LRU cached-block eviction counters are now recorded even when the
   experimental value policy is disabled.
2. Cached prefix blocks now update their observed chain length when later
   blocks from the same prefix chain are cached.

The effective comparison uses `MIN_USAGE=0`, because the previous default
`MIN_USAGE=0.20` skipped nearly all value-policy reorder attempts.

## Workload

- Model: `/public/model_zoo/Qwen3-0.6B`
- Trace: `results/trace_driven_sharegpt_multiturn_20260623/sharegpt_burstgpt_trace_128.json`
- Seeds: 0, 1, 2
- Requests per seed: 128
- Policies:
  - `lru`
  - `prefix_only`
  - `chain_hybrid`
  - `hybrid_hot1`
- Result root:
  `results/trace_driven_sharegpt_burstgpt_policy_compare_online_minusage0_chainfix`

## Aggregate Results

| policy | avg TTFT | vs LRU | p90 TTFT | vs LRU | avg evicted downstream blocks |
| --- | ---: | ---: | ---: | ---: | ---: |
| `chain_hybrid` | 21.6022 ms | -7.89% | 32.2354 ms | -4.13% | 0.0000 |
| `prefix_only` | 22.5056 ms | -4.04% | 32.5041 ms | -3.33% | 0.4472 |
| `lru` | 23.4526 ms | 0.00% | 33.6252 ms | 0.00% | 0.4450 |
| `hybrid_hot1` | 24.1197 ms | +2.84% | 35.1484 ms | +4.53% | 0.4455 |

## Mechanism Signals

With `MIN_USAGE=0`, the value policies actually reorder candidates:

| policy | reorder attempts | reorder count | scored blocks | skipped low usage |
| --- | ---: | ---: | ---: | ---: |
| `prefix_only` | 374.33 | 4.67 | 1088.00 | 0 |
| `hybrid_hot1` | 374.00 | 12.00 | 917.33 | 0 |
| `chain_hybrid` | 374.33 | 11.67 | 1194.67 | 0 |

The key mechanism result is that `chain_hybrid` avoids evicting blocks with
observed downstream chain impact:

```text
avg_evicted_chain_downstream_blocks
chain_hybrid  0.0000
prefix_only   0.4472
lru           0.4450
hybrid_hot1   0.4455
```

At the same time, the observed hotness signal remains weak in this workload:

```text
avg_evicted_apc_hit_count = 0.0000
```

This means the current evidence supports chain-aware prefix retention more
strongly than hotness-aware eviction.

## Current Interpretation

The best-supported research claim is:

> Chain-aware prefix block retention reduces eviction of blocks with downstream
> prefix-chain impact and improves average and tail TTFT on a trace-driven
> multi-turn online workload.

The current evidence does not support making short-term hotness the main
contribution. `hybrid_hot1` performs worse than LRU in the aggregate result,
while the evicted APC hit count stays at zero.

## 512-Request Smoke Result

The first higher-pressure smoke was run on 2026-07-17:

```bash
MIN_USAGE=0 MAX_NUM_REQUESTS=512 GPU_ID=1 SEED=0 PORT_BASE=8132 \
RESULT_ROOT=results/trace_driven_sharegpt_burstgpt_policy_compare_online_minusage0_chainfix_512 \
scripts/run_trace_driven_value_policy_online_sweep.sh \
lru prefix_only chain_hybrid
```

Result root:
`results/trace_driven_sharegpt_burstgpt_policy_compare_online_minusage0_chainfix_512/20260717_004748`

| policy | avg TTFT | p90 TTFT | avg evicted downstream blocks | reorder time |
| --- | ---: | ---: | ---: | ---: |
| `prefix_only` | 18.0303 ms | 22.7542 ms | 28.8581 | 0.1166 s |
| `lru` | 19.2691 ms | 27.7646 ms | 28.8441 | 0.0000 s |
| `chain_hybrid` | 20.0912 ms | 30.9630 ms | 28.6064 | 0.1341 s |

This smoke does not support immediately scaling the same configuration to
seeds 1 and 2. At 512 requests, `prefix_only` is best, `chain_hybrid` is slower
than LRU, and the downstream-eviction reduction is small
(`28.8441 -> 28.6064`).

The run appears valid: the benchmark completed 512 total samples and
`--max-num-requests` is split across clients, not applied once per client. The
likely issue is that this pressure setting changes the eviction regime: evicted
blocks increase from tens of blocks in the 128-request run to more than 4400
blocks, and the current chain boost no longer gives a clear retention advantage.

## Shared-Chain Refinement

A follow-up 512-request ablation showed that plain chain boosting is unstable.
With hotness and restore disabled, `chain_share2` can beat LRU on seed 0, but
still trails `prefix_only`; lower chain weights are not consistently useful.
Per-request diff analysis showed the pattern: chain boosting helps longer cached
requests, but hurts cold or short requests. This suggests that protecting private
long chains is too broad.

A narrower policy, `shared_chain_hybrid`, was added. It applies the chain boost
only when a block has been observed in more than one request; private chains use
the base sharing score without chain amplification. The 512-request 3-seed run
used `shared_chain1`, i.e. `shared_chain_hybrid` with hotness and restore disabled
and `chain_weight=1.0`.

Result root:
`results/trace_driven_sharegpt_burstgpt_policy_compare_online_minusage0_shared_chain_512/combined_3seed`

| policy | avg TTFT | vs LRU | p90 TTFT | vs LRU | avg evicted downstream blocks |
| --- | ---: | ---: | ---: | ---: | ---: |
| `shared_chain1` | 19.4386 ms | -3.71% | 28.7701 ms | +1.26% | 29.0545 |
| `lru` | 20.1877 ms | 0.00% | 28.4128 ms | 0.00% | 28.9081 |
| `prefix_only` | 20.8998 ms | +3.53% | 30.4790 ms | +7.27% | 29.8595 |

This is a better direction than the original `chain_hybrid`: it improves average
TTFT at 512 requests and avoids the large slowdown from protecting private
long-chain blocks. However, it is not yet a complete win because p90 TTFT is
still slightly worse than LRU. The next policy should combine shared-chain
retention with an explicit tail guard, or move to restore-aware selective
admission where long private chains can be offloaded or dropped based on reuse
probability instead of being retained in GPU memory.

## Next Step

Do not continue the original `chain_hybrid` configuration. It is too broad at
512 requests because it also protects private long chains. The current viable
line is `shared_chain_hybrid`, which applies chain retention only to blocks
observed in multiple requests.

The next concrete step is to add a tail-latency guard for `shared_chain_hybrid`:

1. keep shared-chain boosting for average TTFT;
2. avoid excessive reorder churn on cold or short requests;
3. preserve p90 TTFT at least on par with LRU;
4. then evaluate whether those blocks should become `KEEP`, `OFFLOAD`, or `DROP`
   in the restore-aware selective admission design.

If these hold, the next research phase should upgrade the policy from GPU-only
eviction scoring to restore-aware selective admission:

- `KEEP`: preserve high-value blocks in GPU memory.
- `OFFLOAD`: evict from GPU but admit into CPU/cache storage.
- `DROP`: evict without preserving low-value one-off blocks.
- `RESTORE` or `RECOMPUTE`: decide on reuse based on estimated restore cost
  versus recomputation cost.

## Tail-Guard Follow-up (2026-07-19)

A prefix-hit guard was added for value-policy reordering:

```text
VLLM_EXPERIMENTAL_GPU_EVICTION_MIN_REORDER_HIT_BLOCKS
```

When this threshold is greater than zero, `BlockPool` only reorders the free
queue for allocations belonging to requests with at least that many local prefix
hit blocks. The goal is to keep shared-chain retention for high-reuse requests
while leaving cold or low-hit requests closer to LRU behavior.

The first seed-0 smoke showed the guard moved in the intended direction:

| policy | avg TTFT | p90 TTFT | reorder count | skipped low-hit reorders | reorder time |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lru` | 18.9912 ms | 26.4066 ms | 0 | 0 | 0.0000 s |
| `shared_chain1` | 20.9902 ms | 30.7385 ms | 552 | 0 | 0.1512 s |
| `shared_chain1_hit8` | 20.2287 ms | 29.8833 ms | 327 | 725 | 0.0871 s |

A stricter threshold was then tested. `shared_chain1_hit24` reduced reorder churn
further and gave a small 3-seed aggregate win over LRU on this 512-request
workload:

Result root:
`results/trace_driven_sharegpt_burstgpt_policy_compare_online_minusage0_shared_chain_guard_hit24_512/combined_3seed`

| policy | avg TTFT | vs LRU | p90 TTFT | vs LRU | reorder count | skipped low-hit reorders | avg evicted downstream blocks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `shared_chain1_hit24` | 28.7691 ms | -0.67% | 41.2085 ms | -0.35% | 156.0000 | 1164.6667 | 29.0750 |
| `lru` | 28.9631 ms | 0.00% | 41.3536 ms | 0.00% | 0.0000 | 0.0000 | 29.2579 |

Per-seed results are mixed: seed 0 and seed 2 improve both average and p90 TTFT,
while seed 1 regresses. This means the guard is a plausible direction, but the
current evidence is still weak. The next useful step is request-level diffing for
seed 1 and a small threshold sweep around 20-32 hit blocks before treating
`hit24` as the default policy.

