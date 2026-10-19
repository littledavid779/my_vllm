# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare CPU KV-cache admission policies on one controlled request trace.

The server must be freshly started. The trace is A -> A -> A -> P -> Q -> A:

* A is a reusable target prefix.
* Two repeated As establish cross-request reuse.
* P is a long, one-time polluter.
* Q is shorter than one KV block and gives pending stores another engine step.
* The final A reveals whether the policy retained useful external KV.

The workload never resets either the local or external prefix cache.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from external_restore_sanity import (
    SanityError,
    _check,
    _delta,
    _fetch_transfer_metrics,
    _fetch_value_policy_stats,
    _make_prompt,
    _request,
    _send_completion,
    _stat,
    _wait_for_cpu_to_gpu_bytes,
    _wait_for_stat,
)


def _token_count(base_url: str, model: str, prompt: str) -> int:
    response = _request(
        base_url,
        "/tokenize",
        method="POST",
        payload={"model": model, "prompt": prompt},
    )
    count = response.get("count")
    if not isinstance(count, int):
        raise SanityError(f"/tokenize response has no integer count: {response}")
    return count


def _phase(
    args: argparse.Namespace,
    prompt: str,
    request_id: str,
) -> dict[str, Any]:
    response = _send_completion(
        args.url,
        args.model,
        prompt,
        request_id=request_id,
        timeout=args.request_timeout_seconds,
    )
    return {
        "response": response,
        "stats": _fetch_value_policy_stats(args.url),
        "transfer_metrics": _fetch_transfer_metrics(args.url),
    }


def _stats_delta(
    after_phase: dict[str, Any],
    before_phase: dict[str, Any],
    name: str,
) -> float:
    return _delta(after_phase["stats"], before_phase["stats"], name)


def run(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    report: dict[str, Any] = {
        "status": "ERROR",
        "server": {"url": args.url, "model": args.model},
        "configuration": {
            "scenario": "admission-comparison",
            "trace": [
                "target_cold",
                "target_repeat_1",
                "target_repeat_2",
                "polluter",
                "drain",
                "target_final",
            ],
            "mode": args.mode,
            "expected_final_path": args.expected_final_path,
            "target_prompt_tokens": args.target_prompt_tokens,
            "polluter_target_prompt_tokens": args.polluter_target_prompt_tokens,
            "block_size": args.block_size,
            "stat_wait_seconds": args.stat_wait_seconds,
            "metric_wait_seconds": args.metric_wait_seconds,
        },
        "phases": {},
        "checks": {},
    }

    _request(args.url, "/v1/models", timeout=args.request_timeout_seconds)
    # The runner provides a fresh server for every arm. Reset counters only;
    # this trace never calls the local prefix-cache reset endpoint.
    _fetch_value_policy_stats(args.url, reset=True)

    target_prompt, target_token_count = _make_prompt(
        args.url,
        args.model,
        args.target_prompt_tokens,
    )
    polluter_body, _ = _make_prompt(
        args.url,
        args.model,
        args.polluter_target_prompt_tokens,
    )
    polluter_prompt = "One-time admission polluter; no shared prefix.\n" + polluter_body
    polluter_token_count = _token_count(args.url, args.model, polluter_prompt)
    drain_prompt = "Q"
    drain_token_count = _token_count(args.url, args.model, drain_prompt)

    prompts = {
        "target": {
            "tokenized_count": target_token_count,
            "full_blocks": target_token_count // args.block_size,
            "character_count": len(target_prompt),
            "sha256": hashlib.sha256(target_prompt.encode()).hexdigest(),
        },
        "polluter": {
            "tokenized_count": polluter_token_count,
            "full_blocks": polluter_token_count // args.block_size,
            "character_count": len(polluter_prompt),
            "sha256": hashlib.sha256(polluter_prompt.encode()).hexdigest(),
        },
        "drain": {
            "tokenized_count": drain_token_count,
            "full_blocks": drain_token_count // args.block_size,
            "character_count": len(drain_prompt),
            "sha256": hashlib.sha256(drain_prompt.encode()).hexdigest(),
        },
    }
    report["prompts"] = prompts

    baseline_stats = _fetch_value_policy_stats(args.url)
    baseline_metrics = _fetch_transfer_metrics(args.url)
    promotion_policy = baseline_stats.get("offload_promotion_policy")
    report["configuration"]["promotion_policy"] = promotion_policy

    target_cold = _phase(
        args,
        target_prompt,
        "external-admission-target-cold",
    )
    report["phases"]["target_cold"] = target_cold

    target_repeat_1 = _phase(
        args,
        target_prompt,
        "external-admission-target-repeat-1",
    )
    report["phases"]["target_repeat_1"] = target_repeat_1

    target_repeat_2 = _phase(
        args,
        target_prompt,
        "external-admission-target-repeat-2",
    )
    report["phases"]["target_repeat_2"] = target_repeat_2

    polluter = _phase(
        args,
        polluter_prompt,
        "external-admission-polluter",
    )
    report["phases"]["polluter"] = polluter

    # Q has no full block to offload. Its engine step lets any pending store
    # work settle without introducing another cache-sized prefix.
    drain = _phase(
        args,
        drain_prompt,
        "external-admission-store-drain",
    )

    # Admission counters are updated synchronously in prepare_store, while D2H
    # completion can lag. Stabilize all admitted stores before the final A.
    pre_final_stats = drain["stats"]
    total_store_actual = _stat(pre_final_stats, "offload_store_actual_blocks")
    if total_store_actual > 0:
        pre_final_stats = _wait_for_stat(
            args.url,
            "offload_store_completed_blocks",
            total_store_actual,
            args.stat_wait_seconds,
        )
    drain["stats_after_store_wait"] = pre_final_stats
    drain["transfer_metrics_after_store_wait"] = _fetch_transfer_metrics(args.url)
    report["phases"]["drain"] = drain

    pre_final_metrics = drain["transfer_metrics_after_store_wait"]
    pre_final_phase = {
        "stats": pre_final_stats,
        "transfer_metrics": pre_final_metrics,
    }
    target_final = _phase(
        args,
        target_prompt,
        "external-admission-target-final",
    )
    final_stats = target_final["stats"]
    final_load_requested = _delta(
        final_stats,
        pre_final_stats,
        "offload_load_requested_blocks",
    )
    if final_load_requested > 0:
        final_stats = _wait_for_stat(
            args.url,
            "offload_load_completed_blocks",
            _stat(pre_final_stats, "offload_load_completed_blocks")
            + final_load_requested,
            args.stat_wait_seconds,
        )
        final_metrics = _wait_for_cpu_to_gpu_bytes(
            args.url,
            pre_final_metrics["cpu_to_gpu_bytes"],
            args.metric_wait_seconds,
        )
    else:
        final_metrics = _fetch_transfer_metrics(args.url)
    target_final["stats_after_load_wait"] = final_stats
    target_final["transfer_metrics_after_load_wait"] = final_metrics
    report["phases"]["target_final"] = target_final

    target_store_actual = _delta(
        target_repeat_2["stats"],
        baseline_stats,
        "offload_store_actual_blocks",
    )
    target_store_dropped = _delta(
        target_repeat_2["stats"],
        baseline_stats,
        "offload_store_dropped_blocks",
    )
    polluter_store_actual = _delta(
        pre_final_stats,
        target_repeat_2["stats"],
        "offload_store_actual_blocks",
    )
    polluter_store_dropped = _delta(
        pre_final_stats,
        target_repeat_2["stats"],
        "offload_store_dropped_blocks",
    )
    pre_final_store_requested = _delta(
        pre_final_stats,
        baseline_stats,
        "offload_store_requested_blocks",
    )
    pre_final_store_eligible = _delta(
        pre_final_stats,
        baseline_stats,
        "offload_store_eligible_blocks",
    )
    pre_final_store_dropped = _delta(
        pre_final_stats,
        baseline_stats,
        "offload_store_dropped_blocks",
    )
    pre_final_store_actual = _delta(
        pre_final_stats,
        baseline_stats,
        "offload_store_actual_blocks",
    )
    pre_final_store_completed = _delta(
        pre_final_stats,
        baseline_stats,
        "offload_store_completed_blocks",
    )
    natural_evicted_blocks = _stats_delta(
        pre_final_phase,
        target_repeat_2,
        "evicted_blocks",
    )
    final_effective_hit_blocks = _delta(
        final_stats,
        pre_final_stats,
        "effective_prefix_hit_blocks_sum",
    )
    final_external_lookup_hit_blocks = _delta(
        final_stats,
        pre_final_stats,
        "offload_lookup_hit_blocks",
    )
    final_load_completed = _delta(
        final_stats,
        pre_final_stats,
        "offload_load_completed_blocks",
    )
    final_promotion_candidate_blocks = _delta(
        final_stats,
        pre_final_stats,
        "offload_promotion_candidate_blocks",
    )
    final_promotion_candidate_tokens = _delta(
        final_stats,
        pre_final_stats,
        "offload_promotion_candidate_tokens",
    )
    final_promotion_restore_decisions = _delta(
        final_stats,
        pre_final_stats,
        "offload_promotion_restore_decision_count",
    )
    final_promotion_recompute_decisions = _delta(
        final_stats,
        pre_final_stats,
        "offload_promotion_recompute_decision_count",
    )
    final_cpu_to_gpu_bytes = (
        final_metrics["cpu_to_gpu_bytes"] - pre_final_metrics["cpu_to_gpu_bytes"]
    )
    final_cpu_to_gpu_time_s = (
        final_metrics["cpu_to_gpu_time_s"] - pre_final_metrics["cpu_to_gpu_time_s"]
    )
    pre_final_gpu_to_cpu_bytes = (
        pre_final_metrics["gpu_to_cpu_bytes"] - baseline_metrics["gpu_to_cpu_bytes"]
    )
    pre_final_gpu_to_cpu_time_s = (
        pre_final_metrics["gpu_to_cpu_time_s"] - baseline_metrics["gpu_to_cpu_time_s"]
    )

    target_full_blocks = prompts["target"]["full_blocks"]
    estimated_recomputed_target_blocks = max(
        target_full_blocks - final_effective_hit_blocks - final_load_completed,
        0,
    )
    request_latency_sum_ms = sum(
        report["phases"][name]["response"]["latency_ms"]
        for name in (
            "target_cold",
            "target_repeat_1",
            "target_repeat_2",
            "polluter",
            "drain",
            "target_final",
        )
    )
    loaded_blocks_per_completed_store = (
        final_load_completed / pre_final_store_completed
        if pre_final_store_completed > 0
        else 0.0
    )

    deltas = {
        "target_store_actual_blocks": target_store_actual,
        "target_store_dropped_blocks": target_store_dropped,
        "polluter_store_actual_blocks": polluter_store_actual,
        "polluter_store_dropped_blocks": polluter_store_dropped,
        "pre_final_store_requested_blocks": pre_final_store_requested,
        "pre_final_store_eligible_blocks": pre_final_store_eligible,
        "pre_final_store_dropped_blocks": pre_final_store_dropped,
        "pre_final_store_actual_blocks": pre_final_store_actual,
        "pre_final_store_completed_blocks": pre_final_store_completed,
        "natural_evicted_blocks": natural_evicted_blocks,
        "final_effective_prefix_hit_blocks": final_effective_hit_blocks,
        "final_external_lookup_hit_blocks": final_external_lookup_hit_blocks,
        "final_load_requested_blocks": final_load_requested,
        "final_load_completed_blocks": final_load_completed,
        "final_promotion_candidate_blocks": final_promotion_candidate_blocks,
        "final_promotion_candidate_tokens": final_promotion_candidate_tokens,
        "final_promotion_restore_decisions": final_promotion_restore_decisions,
        "final_promotion_recompute_decisions": (final_promotion_recompute_decisions),
        "estimated_recomputed_target_blocks": estimated_recomputed_target_blocks,
        "pre_final_gpu_to_cpu_bytes": pre_final_gpu_to_cpu_bytes,
        "pre_final_gpu_to_cpu_time_s": pre_final_gpu_to_cpu_time_s,
        "final_cpu_to_gpu_bytes": final_cpu_to_gpu_bytes,
        "final_cpu_to_gpu_time_s": final_cpu_to_gpu_time_s,
        "loaded_blocks_per_completed_store": loaded_blocks_per_completed_store,
        "request_latency_sum_ms": request_latency_sum_ms,
        "final_target_latency_ms": target_final["response"]["latency_ms"],
    }
    report["deltas"] = deltas

    checks = {
        "distinct_target_and_polluter": _check(
            prompts["target"]["sha256"] != prompts["polluter"]["sha256"],
            {
                "target": prompts["target"]["sha256"],
                "polluter": prompts["polluter"]["sha256"],
            },
            "different SHA-256 hashes",
        ),
        "polluter_larger_than_target": _check(
            polluter_token_count > target_token_count,
            {
                "target_tokens": target_token_count,
                "polluter_tokens": polluter_token_count,
            },
            "polluter_tokens > target_tokens",
        ),
        "drain_has_no_full_block": _check(
            drain_token_count < args.block_size,
            drain_token_count,
            f"< {args.block_size} tokens",
        ),
        "natural_gpu_eviction": _check(
            natural_evicted_blocks > 0,
            natural_evicted_blocks,
            "> 0 blocks",
        ),
        "stores_drained_before_final_target": _check(
            pre_final_store_completed >= pre_final_store_actual,
            {
                "actual": pre_final_store_actual,
                "completed": pre_final_store_completed,
            },
            "completed >= actual",
        ),
        "same_target_output": _check(
            len(
                {
                    target_cold["response"]["text"],
                    target_repeat_1["response"]["text"],
                    target_repeat_2["response"]["text"],
                    target_final["response"]["text"],
                }
            )
            == 1,
            {
                "cold": target_cold["response"]["text"],
                "repeat_1": target_repeat_1["response"]["text"],
                "repeat_2": target_repeat_2["response"]["text"],
                "final": target_final["response"]["text"],
            },
            "all target responses identical",
        ),
    }

    if args.mode == "no-offload":
        checks.update(
            {
                "no_cpu_store": _check(
                    pre_final_store_actual == 0,
                    pre_final_store_actual,
                    "0 blocks",
                ),
                "no_external_load": _check(
                    final_load_completed == 0,
                    final_load_completed,
                    "0 blocks",
                ),
                "no_kv_transfer": _check(
                    pre_final_gpu_to_cpu_bytes == 0 and final_cpu_to_gpu_bytes == 0,
                    {
                        "gpu_to_cpu_bytes": pre_final_gpu_to_cpu_bytes,
                        "cpu_to_gpu_bytes": final_cpu_to_gpu_bytes,
                    },
                    "both directions == 0 bytes",
                ),
            }
        )
    elif args.mode == "all-block":
        checks.update(
            {
                "polluter_admitted": _check(
                    polluter_store_actual > 0,
                    polluter_store_actual,
                    "> 0 blocks",
                ),
                "no_admission_drop": _check(
                    pre_final_store_dropped == 0,
                    pre_final_store_dropped,
                    "0 blocks",
                ),
            }
        )
    elif args.mode == "selective":
        checks.update(
            {
                "reused_target_admitted": _check(
                    target_store_actual > 0,
                    {
                        "actual": target_store_actual,
                        "dropped_before_admission": target_store_dropped,
                    },
                    "> 0 target blocks admitted",
                ),
                "one_time_polluter_dropped": _check(
                    polluter_store_actual == 0 and polluter_store_dropped > 0,
                    {
                        "actual": polluter_store_actual,
                        "dropped": polluter_store_dropped,
                    },
                    "actual == 0 and dropped > 0",
                ),
            }
        )

    if args.expected_final_path == "restore":
        checks["expected_final_restore"] = _check(
            final_load_completed > 0 and final_cpu_to_gpu_bytes > 0,
            {
                "load_completed_blocks": final_load_completed,
                "cpu_to_gpu_bytes": final_cpu_to_gpu_bytes,
            },
            "load_completed_blocks > 0 and CPU-to-GPU bytes > 0",
        )
        if promotion_policy == "always_restore":
            checks["expected_restore_promotion_decision"] = _check(
                final_promotion_restore_decisions > 0
                and final_promotion_recompute_decisions == 0,
                {
                    "restore_decisions": final_promotion_restore_decisions,
                    "recompute_decisions": final_promotion_recompute_decisions,
                },
                "restore_decisions > 0 and recompute_decisions == 0",
            )
    else:
        checks["expected_final_recompute"] = _check(
            final_load_completed == 0 and final_cpu_to_gpu_bytes == 0,
            {
                "load_completed_blocks": final_load_completed,
                "cpu_to_gpu_bytes": final_cpu_to_gpu_bytes,
            },
            "load_completed_blocks == 0 and CPU-to-GPU bytes == 0",
        )
        if promotion_policy == "always_recompute":
            checks["expected_recompute_promotion_decision"] = _check(
                final_external_lookup_hit_blocks > 0
                and final_promotion_recompute_decisions > 0
                and final_promotion_restore_decisions == 0,
                {
                    "external_lookup_hit_blocks": final_external_lookup_hit_blocks,
                    "restore_decisions": final_promotion_restore_decisions,
                    "recompute_decisions": final_promotion_recompute_decisions,
                },
                "CPU hit and recompute_decisions > 0 and restore_decisions == 0",
            )

    report["checks"] = checks
    passed = all(check["passed"] for check in checks.values())
    report["status"] = "PASS" if passed else "FAIL"
    return (0 if passed else 1), report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare CPU KV admission on a controlled A-A-A-P-Q-A trace."
    )
    parser.add_argument("--url", default="http://127.0.0.1:8140")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("no-offload", "all-block", "selective"),
        required=True,
    )
    parser.add_argument(
        "--expected-final-path",
        choices=("restore", "recompute"),
        required=True,
    )
    parser.add_argument("--target-prompt-tokens", type=int, default=512)
    parser.add_argument("--polluter-target-prompt-tokens", type=int, default=2000)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--stat-wait-seconds", type=float, default=10.0)
    parser.add_argument("--metric-wait-seconds", type=float, default=20.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=120.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report: dict[str, Any]
    try:
        exit_code, report = run(args)
    except Exception as exc:
        exit_code = 2
        report = {
            "status": "ERROR",
            "error": f"{type(exc).__name__}: {exc}",
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"external admission comparison: {report['status']}")
    print(f"report: {args.output}")
    if report.get("checks"):
        for name, check in report["checks"].items():
            state = "PASS" if check["passed"] else "FAIL"
            print(f"  {state:4} {name}: {check['actual']}")
    elif report.get("error"):
        print(report["error"], file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
