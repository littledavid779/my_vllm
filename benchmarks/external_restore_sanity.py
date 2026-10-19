# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deterministic end-to-end sanity check for external KV-cache restore.

The check sends a cold completion request followed by an identical GPU-hit
request. The second request advances the engine so that the deferred CPU store
from the cold request can complete. It then clears only the local GPU prefix
cache and sends the exact same request a third time. A passing run proves that
the third request missed the GPU cache and restored KV blocks from the external
CPU cache.

This script expects a freshly started server. The OffloadingConnector does not
currently implement an external-cache reset operation.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class SanityError(RuntimeError):
    """Raised when the server cannot execute the sanity workload."""


def _request(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    query: dict[str, str] | None = None,
    timeout: float = 30.0,
    expect_json: bool = True,
    headers: dict[str, str] | None = None,
) -> Any:
    url = f"{base_url.rstrip('/')}{path}"
    if query:
        url = f"{url}?{urlencode(query)}"

    body = None
    request_headers = dict(headers or {})
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        request_headers["Content-Type"] = "application/json"

    request = Request(
        url,
        data=body,
        headers=request_headers,
        method=method,
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8")
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise SanityError(
            f"{method} {url} returned HTTP {exc.code}: {error_body}"
        ) from exc
    except URLError as exc:
        raise SanityError(f"{method} {url} failed: {exc.reason}") from exc

    if not expect_json:
        return response_body
    if not response_body:
        return {}
    try:
        return json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise SanityError(f"{method} {url} returned invalid JSON") from exc


def _make_prompt(base_url: str, model: str, target_tokens: int) -> tuple[str, int]:
    if target_tokens < 1:
        raise ValueError("target_tokens must be positive")

    def build_prompt(num_segments: int) -> str:
        return "".join(
            "External KV restore sanity segment "
            f"{index:05d}: alpha beta gamma delta epsilon zeta eta theta.\n"
            for index in range(num_segments)
        )

    def token_count(num_segments: int) -> int:
        response = _request(
            base_url,
            "/tokenize",
            method="POST",
            payload={"model": model, "prompt": build_prompt(num_segments)},
        )
        count = response.get("count")
        if not isinstance(count, int):
            raise SanityError(f"/tokenize response has no integer count: {response}")
        return count

    lower = 1
    upper = 1
    while token_count(upper) < target_tokens:
        lower = upper + 1
        upper *= 2

    while lower < upper:
        middle = (lower + upper) // 2
        if token_count(middle) < target_tokens:
            lower = middle + 1
        else:
            upper = middle

    prompt = build_prompt(lower)
    return prompt, token_count(lower)


def _fetch_value_policy_stats(base_url: str, *, reset: bool = False) -> dict[str, Any]:
    response = _request(
        base_url,
        "/debug/value_policy_stats",
        query={"reset": str(reset).lower()},
    )
    stats = response.get("stats")
    if not isinstance(stats, dict):
        raise SanityError(
            f"/debug/value_policy_stats response has no stats object: {response}"
        )
    return stats


def _stat(stats: dict[str, Any], name: str) -> float:
    value = stats.get(name, 0)
    if not isinstance(value, int | float):
        raise SanityError(f"stat {name!r} is not numeric: {value!r}")
    return float(value)


def _wait_for_stat(
    base_url: str,
    name: str,
    minimum: float,
    timeout: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    latest: dict[str, Any] = {}
    while True:
        latest = _fetch_value_policy_stats(base_url)
        if _stat(latest, name) >= minimum:
            return latest
        if time.monotonic() >= deadline:
            return latest
        time.sleep(0.2)


def _parse_transfer_metrics(metrics_text: str) -> dict[str, float]:
    values = {
        "cpu_to_gpu_bytes": 0.0,
        "cpu_to_gpu_time_s": 0.0,
        "gpu_to_cpu_bytes": 0.0,
        "gpu_to_cpu_time_s": 0.0,
    }
    metric_names = {
        "vllm:kv_offload_total_bytes": {
            "CPU_to_GPU": "cpu_to_gpu_bytes",
            "GPU_to_CPU": "gpu_to_cpu_bytes",
        },
        "vllm:kv_offload_total_time": {
            "CPU_to_GPU": "cpu_to_gpu_time_s",
            "GPU_to_CPU": "gpu_to_cpu_time_s",
        },
    }

    for line in metrics_text.splitlines():
        if not line or line.startswith("#"):
            continue
        try:
            sample, raw_value = line.rsplit(maxsplit=1)
            metric_name = sample.split("{", maxsplit=1)[0]
            value = float(raw_value)
        except (ValueError, IndexError):
            continue

        for expected_name, transfer_keys in metric_names.items():
            if metric_name not in (expected_name, f"{expected_name}_total"):
                continue
            for transfer_type, output_key in transfer_keys.items():
                if f'transfer_type="{transfer_type}"' in sample:
                    values[output_key] += value
    return values


def _fetch_transfer_metrics(base_url: str) -> dict[str, float]:
    metrics_text = _request(
        base_url,
        "/metrics",
        expect_json=False,
    )
    return _parse_transfer_metrics(metrics_text)


def _wait_for_cpu_to_gpu_bytes(
    base_url: str,
    baseline_bytes: float,
    timeout: float,
) -> dict[str, float]:
    deadline = time.monotonic() + timeout
    latest: dict[str, float] = {}
    while True:
        latest = _fetch_transfer_metrics(base_url)
        if latest["cpu_to_gpu_bytes"] > baseline_bytes:
            return latest
        if time.monotonic() >= deadline:
            return latest
        time.sleep(0.5)


def _send_completion(
    base_url: str,
    model: str,
    prompt: str,
    request_id: str,
    timeout: float,
) -> dict[str, Any]:
    start = time.perf_counter()
    response = _request(
        base_url,
        "/v1/completions",
        method="POST",
        payload={
            "model": model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0.0,
            "seed": 0,
            "stream": False,
        },
        headers={"X-Request-Id": request_id},
        timeout=timeout,
    )
    latency_ms = (time.perf_counter() - start) * 1000

    usage = response.get("usage")
    choices = response.get("choices")
    if not isinstance(usage, dict) or not isinstance(choices, list) or not choices:
        raise SanityError(f"invalid completion response: {response}")

    choice = choices[0]
    if not isinstance(choice, dict):
        raise SanityError(f"invalid completion choice: {choice!r}")
    return {
        "latency_ms": latency_ms,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "text": choice.get("text"),
        "finish_reason": choice.get("finish_reason"),
    }


def _delta(
    after: dict[str, Any],
    before: dict[str, Any],
    name: str,
) -> float:
    return _stat(after, name) - _stat(before, name)


def _check(passed: bool, actual: Any, expected: str) -> dict[str, Any]:
    return {
        "passed": passed,
        "actual": actual,
        "expected": expected,
    }


def run(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    report: dict[str, Any] = {
        "status": "ERROR",
        "server": {"url": args.url, "model": args.model},
        "configuration": {
            "target_prompt_tokens": args.target_prompt_tokens,
            "stat_wait_seconds": args.stat_wait_seconds,
            "metric_wait_seconds": args.metric_wait_seconds,
        },
        "phases": {},
        "checks": {},
    }

    _request(args.url, "/v1/models", timeout=args.request_timeout_seconds)
    # The runner starts a fresh server, so the external cache is already empty.
    # Clear any local blocks created by readiness or manual probes, then reset
    # counters without claiming to reset the connector-managed CPU cache.
    _request(args.url, "/reset_prefix_cache", method="POST")
    _fetch_value_policy_stats(args.url, reset=True)

    prompt, tokenized_count = _make_prompt(
        args.url,
        args.model,
        args.target_prompt_tokens,
    )
    report["prompt"] = {
        "tokenized_count": tokenized_count,
        "character_count": len(prompt),
        "sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    }

    baseline_stats = _fetch_value_policy_stats(args.url)
    baseline_metrics = _fetch_transfer_metrics(args.url)
    cold_response = _send_completion(
        args.url,
        args.model,
        prompt,
        request_id="external-restore-sanity-cold",
        timeout=args.request_timeout_seconds,
    )
    cold_stats = _fetch_value_policy_stats(args.url)
    report["phases"]["cold"] = {
        "response": cold_response,
        "stats": cold_stats,
        "transfer_metrics": _fetch_transfer_metrics(args.url),
    }

    # Stores are submitted at the beginning of the next engine step. Repeating
    # the request should hit the GPU cache while advancing the deferred store.
    flush_response = _send_completion(
        args.url,
        args.model,
        prompt,
        request_id="external-restore-sanity-store-flush",
        timeout=args.request_timeout_seconds,
    )
    flushed_stats = _wait_for_stat(
        args.url,
        "offload_store_completed_blocks",
        _stat(baseline_stats, "offload_store_completed_blocks") + 1,
        args.stat_wait_seconds,
    )
    report["phases"]["store_flush"] = {
        "response": flush_response,
        "stats": flushed_stats,
        "transfer_metrics": _fetch_transfer_metrics(args.url),
    }

    _request(args.url, "/reset_prefix_cache", method="POST")
    before_restore_stats = _fetch_value_policy_stats(args.url)

    restore_response = _send_completion(
        args.url,
        args.model,
        prompt,
        request_id="external-restore-sanity-restore",
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

    cold_store_completed = _delta(
        flushed_stats,
        baseline_stats,
        "offload_store_completed_blocks",
    )
    flush_gpu_hit_blocks = _delta(
        flushed_stats,
        cold_stats,
        "effective_prefix_hit_blocks_sum",
    )
    local_zero_hits = _delta(
        restore_stats,
        before_restore_stats,
        "prefix_cache_zero_hit_count",
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
        "cold_store_completed": _check(
            cold_store_completed > 0,
            cold_store_completed,
            "> 0 blocks",
        ),
        "store_flush_gpu_hit": _check(
            flush_gpu_hit_blocks > 0,
            flush_gpu_hit_blocks,
            "> 0 blocks",
        ),
        "restore_gpu_zero_hit": _check(
            local_zero_hits > 0,
            local_zero_hits,
            "> 0 additional zero-hit queries",
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
        "same_prompt_token_count": _check(
            cold_response["prompt_tokens"]
            == flush_response["prompt_tokens"]
            == restore_response["prompt_tokens"],
            {
                "cold": cold_response["prompt_tokens"],
                "store_flush": flush_response["prompt_tokens"],
                "restore": restore_response["prompt_tokens"],
            },
            "cold == store_flush == restore",
        ),
        "same_output": _check(
            cold_response["text"] == flush_response["text"] == restore_response["text"],
            {
                "cold": cold_response["text"],
                "store_flush": flush_response["text"],
                "restore": restore_response["text"],
            },
            "cold == store_flush == restore",
        ),
    }
    report["checks"] = checks
    report["deltas"] = {
        "cold_store_completed_blocks": cold_store_completed,
        "restore_local_zero_hit_queries": local_zero_hits,
        "store_flush_gpu_hit_blocks": flush_gpu_hit_blocks,
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
        description=(
            "Verify GPU miss -> external CPU hit -> CPU-to-GPU KV restore "
            "against a freshly started vLLM server."
        )
    )
    parser.add_argument("--url", default="http://127.0.0.1:8140")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-prompt-tokens", type=int, default=512)
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
    print(f"external restore sanity: {report['status']}")
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
