# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify external KV restore after natural GPU cache eviction.

The server must be freshly started. The workload is target A, distinct polluter
P, P again to drain deferred stores, then A. It never resets the prefix cache.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from external_restore_sanity import (
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
        raise RuntimeError(f"/tokenize response has no integer count: {response}")
    return count


def run(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    report: dict[str, Any] = {
        "status": "ERROR",
        "server": {"url": args.url, "model": args.model},
        "configuration": {
            "scenario": "natural-eviction",
            "target_prompt_tokens": args.target_prompt_tokens,
            "polluter_target_prompt_tokens": args.polluter_target_prompt_tokens,
            "stat_wait_seconds": args.stat_wait_seconds,
            "metric_wait_seconds": args.metric_wait_seconds,
        },
        "phases": {},
        "checks": {},
    }

    _request(args.url, "/v1/models", timeout=args.request_timeout_seconds)
    # A fresh server already has empty local and external caches. Reset only
    # debug counters; never clear the local GPU prefix cache in this workload.
    _fetch_value_policy_stats(args.url, reset=True)

    prompt, tokenized_count = _make_prompt(
        args.url,
        args.model,
        args.target_prompt_tokens,
    )
    polluter_body, _ = _make_prompt(
        args.url,
        args.model,
        args.polluter_target_prompt_tokens,
    )
    polluter_prompt = "Natural eviction polluter; no shared prefix.\n" + polluter_body
    polluter_tokenized_count = _token_count(
        args.url,
        args.model,
        polluter_prompt,
    )
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    polluter_hash = hashlib.sha256(polluter_prompt.encode("utf-8")).hexdigest()
    report["prompt"] = {
        "tokenized_count": tokenized_count,
        "character_count": len(prompt),
        "sha256": prompt_hash,
    }
    report["polluter_prompt"] = {
        "tokenized_count": polluter_tokenized_count,
        "character_count": len(polluter_prompt),
        "sha256": polluter_hash,
    }

    baseline_stats = _fetch_value_policy_stats(args.url)
    baseline_metrics = _fetch_transfer_metrics(args.url)
    cold_response = _send_completion(
        args.url,
        args.model,
        prompt,
        request_id="external-restore-natural-cold",
        timeout=args.request_timeout_seconds,
    )
    cold_stats = _fetch_value_policy_stats(args.url)
    report["phases"]["cold"] = {
        "response": cold_response,
        "stats": cold_stats,
        "transfer_metrics": _fetch_transfer_metrics(args.url),
    }

    # P advances A's deferred D2H store and puts ordinary pressure on the GPU
    # cache. No reset or direct cache mutation occurs between A and P.
    polluter_response = _send_completion(
        args.url,
        args.model,
        polluter_prompt,
        request_id="external-restore-natural-polluter",
        timeout=args.request_timeout_seconds,
    )
    polluter_scheduled_stats = _fetch_value_policy_stats(args.url)
    report["phases"]["polluter"] = {
        "response": polluter_response,
        "stats": polluter_scheduled_stats,
        "transfer_metrics": _fetch_transfer_metrics(args.url),
    }

    pre_restore_store_requested = _delta(
        polluter_scheduled_stats,
        baseline_stats,
        "offload_store_requested_blocks",
    )

    # Stores created at the end of P may remain deferred until the next engine
    # step. Repeating P is a GPU hit that drains those stores without adding a
    # new prefix, so CPU-capacity experiments start from a stable A+P state.
    polluter_flush_response = _send_completion(
        args.url,
        args.model,
        polluter_prompt,
        request_id="external-restore-natural-polluter-store-flush",
        timeout=args.request_timeout_seconds,
    )
    polluted_stats = _wait_for_stat(
        args.url,
        "offload_store_completed_blocks",
        _stat(baseline_stats, "offload_store_completed_blocks")
        + pre_restore_store_requested,
        args.stat_wait_seconds,
    )
    report["phases"]["polluter_store_flush"] = {
        "response": polluter_flush_response,
        "stats": polluted_stats,
        "transfer_metrics": _fetch_transfer_metrics(args.url),
    }

    before_restore_stats = polluted_stats
    restore_response = _send_completion(
        args.url,
        args.model,
        prompt,
        request_id="external-restore-natural-restore",
        timeout=args.request_timeout_seconds,
    )
    restore_stats = _wait_for_stat(
        args.url,
        "offload_load_completed_blocks",
        _stat(before_restore_stats, "offload_load_completed_blocks") + 1,
        args.stat_wait_seconds,
    )
    restore_metrics = _wait_for_cpu_to_gpu_bytes(
        args.url,
        baseline_metrics["cpu_to_gpu_bytes"],
        args.metric_wait_seconds,
    )
    report["phases"]["restore"] = {
        "response": restore_response,
        "stats": restore_stats,
        "transfer_metrics": restore_metrics,
    }

    pre_restore_store_completed = _delta(
        polluted_stats,
        baseline_stats,
        "offload_store_completed_blocks",
    )
    polluter_evicted_blocks = _delta(
        polluted_stats,
        cold_stats,
        "evicted_blocks",
    )
    polluter_flush_gpu_hit_blocks = _delta(
        polluted_stats,
        polluter_scheduled_stats,
        "effective_prefix_hit_blocks_sum",
    )
    restore_gpu_hit_blocks = _delta(
        restore_stats,
        before_restore_stats,
        "effective_prefix_hit_blocks_sum",
    )
    lookup_hit_blocks = _delta(
        restore_stats,
        before_restore_stats,
        "offload_lookup_hit_blocks",
    )
    load_requested_blocks = _delta(
        restore_stats,
        before_restore_stats,
        "offload_load_requested_blocks",
    )
    load_completed_blocks = _delta(
        restore_stats,
        before_restore_stats,
        "offload_load_completed_blocks",
    )
    cpu_to_gpu_bytes = (
        restore_metrics["cpu_to_gpu_bytes"] - baseline_metrics["cpu_to_gpu_bytes"]
    )
    cpu_to_gpu_time_s = (
        restore_metrics["cpu_to_gpu_time_s"] - baseline_metrics["cpu_to_gpu_time_s"]
    )

    checks = {
        "pre_restore_stores_scheduled": _check(
            pre_restore_store_requested > 0,
            pre_restore_store_requested,
            "> 0 blocks",
        ),
        "pre_restore_stores_drained": _check(
            pre_restore_store_completed >= pre_restore_store_requested,
            {
                "requested": pre_restore_store_requested,
                "completed": pre_restore_store_completed,
            },
            "completed >= requested",
        ),
        "polluter_flush_gpu_hit": _check(
            polluter_flush_gpu_hit_blocks > 0,
            polluter_flush_gpu_hit_blocks,
            "> 0 blocks",
        ),
        "distinct_polluter": _check(
            prompt_hash != polluter_hash,
            {"target": prompt_hash, "polluter": polluter_hash},
            "different SHA-256 hashes",
        ),
        "larger_polluter": _check(
            polluter_tokenized_count > tokenized_count,
            {
                "target_tokens": tokenized_count,
                "polluter_tokens": polluter_tokenized_count,
            },
            "polluter_tokens > target_tokens",
        ),
        "natural_gpu_eviction": _check(
            polluter_evicted_blocks > 0,
            polluter_evicted_blocks,
            "> 0 blocks evicted while processing the polluter",
        ),
        "external_lookup_hit": _check(
            lookup_hit_blocks > 0,
            lookup_hit_blocks,
            "> 0 blocks",
        ),
        "external_load_requested": _check(
            load_requested_blocks > 0,
            load_requested_blocks,
            "> 0 blocks",
        ),
        "external_load_completed": _check(
            load_completed_blocks > 0,
            load_completed_blocks,
            "> 0 blocks",
        ),
        "cpu_to_gpu_bytes": _check(
            cpu_to_gpu_bytes > 0,
            cpu_to_gpu_bytes,
            "> 0 bytes",
        ),
        "cpu_to_gpu_time": _check(
            cpu_to_gpu_time_s > 0,
            cpu_to_gpu_time_s,
            "> 0 seconds",
        ),
        "same_target_prompt_token_count": _check(
            cold_response["prompt_tokens"] == restore_response["prompt_tokens"],
            {
                "cold": cold_response["prompt_tokens"],
                "restore": restore_response["prompt_tokens"],
            },
            "cold == restore",
        ),
        "same_target_output": _check(
            cold_response["text"] == restore_response["text"],
            {
                "cold": cold_response["text"],
                "restore": restore_response["text"],
            },
            "cold == restore",
        ),
    }
    report["checks"] = checks
    report["deltas"] = {
        "pre_restore_store_requested_blocks": pre_restore_store_requested,
        "pre_restore_store_completed_blocks": pre_restore_store_completed,
        "polluter_evicted_blocks": polluter_evicted_blocks,
        "polluter_flush_gpu_hit_blocks": polluter_flush_gpu_hit_blocks,
        "restore_gpu_hit_blocks": restore_gpu_hit_blocks,
        "restore_lookup_hit_blocks": lookup_hit_blocks,
        "restore_load_requested_blocks": load_requested_blocks,
        "restore_load_completed_blocks": load_completed_blocks,
        "cpu_to_gpu_bytes": cpu_to_gpu_bytes,
        "cpu_to_gpu_time_s": cpu_to_gpu_time_s,
    }

    passed = all(check["passed"] for check in checks.values())
    report["status"] = "PASS" if passed else "FAIL"
    return (0 if passed else 1), report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Verify external CPU KV restore after natural GPU cache eviction.")
    )
    parser.add_argument("--url", default="http://127.0.0.1:8140")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-prompt-tokens", type=int, default=512)
    parser.add_argument("--polluter-target-prompt-tokens", type=int, default=2000)
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
    print(f"natural external restore: {report['status']}")
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
