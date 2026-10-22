"""Compare native full/prefix and official CacheBlend through one HTTP interface."""

import argparse
import asyncio
import hashlib
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx
from run_smoke import gpu_state

PROJECT = Path(__file__).resolve().parents[1]
ARMS = ("full", "prefix", "cacheblend15")


def score(text, expected):
    if isinstance(expected, list):
        actual = sorted(set(re.findall(r"\b[A-Z]{5}\b", text)))
        return actual == sorted(expected)
    return text.strip().lower().strip(" .\n\t\"'") == expected


async def request(client, model, ids, max_tokens=64):
    started = time.perf_counter()
    first, chunks, rid, usage = None, [], None, None
    async with client.stream(
        "POST",
        "/v1/completions",
        json={
            "model": model,
            "prompt": ids,
            "temperature": 0,
            "seed": 0,
            "max_tokens": max_tokens,
            "stream": True,
            "logprobs": 1,
            "stream_options": {"include_usage": True},
        },
    ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            rid = event.get("id", rid)
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                token_logprobs = choice.get("logprobs") or {}
                if token_logprobs.get("tokens") and first is None:
                    first = time.perf_counter()
                chunks.append(choice.get("text", ""))
    if first is None:
        raise RuntimeError("stream did not provide a model token")
    return {
        "client_ttft_ms": (first - started) * 1000,
        "client_total_ms": (time.perf_counter() - started) * 1000,
        "generated_text": "".join(chunks),
        "request_id": rid,
        "usage": usage,
        "first_token_monotonic": first,
    }


async def measure(args, data, root, arm, server, alias):
    rows = []
    async with httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{args.port}", timeout=180, trust_env=False
    ) as client:
        for _ in range(180):
            if server.poll() is not None:
                raise RuntimeError(f"{arm} server exited; inspect log")
            try:
                health = await client.get("/health")
                if health.is_success:
                    models = (await client.get("/v1/models")).json()
                    if alias not in [item["id"] for item in models["data"]]:
                        raise RuntimeError("port belongs to another service")
                    break
            except httpx.HTTPError:
                pass
            await asyncio.sleep(1)
        else:
            raise RuntimeError("server readiness timeout")
        # Use disjoint synthetic warmup chunks: never warm a measured edit.
        await request(client, alias, data["warmup_seed_ids"], max_tokens=1)
        await request(client, alias, data["warmup_edit_ids"], max_tokens=1)
        for session in data["sessions"]:
            await request(client, alias, session["versions"][0]["token_ids"])
            for version, item in enumerate(session["versions"][1:], 1):
                pipeline_started = time.perf_counter()
                preparation = None
                if arm == "cacheblend15":
                    preparation = await request(
                        client, alias, item["new_chunk_ids"], max_tokens=1
                    )
                row = await request(client, alias, item["token_ids"])
                row["main_request_ttft_ms"] = row["client_ttft_ms"]
                row["new_chunk_prepare_ms"] = (
                    preparation["client_total_ms"] if preparation else 0.0
                )
                row["client_ttft_ms"] = (
                    row["first_token_monotonic"] - pipeline_started
                ) * 1000
                row["client_total_ms"] = (time.perf_counter() - pipeline_started) * 1000
                row.update(
                    arm=arm,
                    session_id=session["session_id"],
                    family=session["family"],
                    version=version,
                    suffix_start=(
                        session["editable_token_span"][1]
                        + 2 * len(data["pilot_provenance"]["separator_ids"])
                    ),
                    expected_value=item["expected_value"],
                    task_correct=score(row["generated_text"], item["expected_value"]),
                    exact_prefix_tokens=next(
                        i
                        for i, (a, b) in enumerate(
                            zip(
                                session["versions"][version - 1]["token_ids"],
                                item["token_ids"],
                                strict=True,
                            )
                        )
                        if a != b
                    ),
                )
                rows.append(row)
                with (root / "requests.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                print(
                    f"{arm} {session['session_id']} edit={version} "
                    f"correct={row['task_correct']}",
                    flush=True,
                )
    return rows


def report(root, rows):
    references = {
        (r["session_id"], r["version"]): r for r in rows if r["arm"] == "full"
    }
    summary = []
    for family in ("all", *sorted({r["family"] for r in rows})):
        for arm in ARMS:
            group = [
                r
                for r in rows
                if r["arm"] == arm and (family == "all" or r["family"] == family)
            ]
            eligible = [
                r
                for r in group
                if references[r["session_id"], r["version"]]["task_correct"]
            ]
            times = sorted(r["client_ttft_ms"] for r in group)
            p95 = statistics.quantiles(times, n=100, method="inclusive")[94]
            summary.append(
                {
                    "family": family,
                    "arm": arm,
                    "events": len(group),
                    "task_accuracy": statistics.mean(r["task_correct"] for r in group),
                    "reference_correct_events": len(eligible),
                    "cache_induced_errors": sum(
                        not r["task_correct"] for r in eligible
                    ),
                    "mean_http_ttft_ms": statistics.mean(times),
                    "p95_http_ttft_ms": p95,
                }
            )
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = [
        "# CacheBlend 官方实现：HTTP 小规模管线验证",
        "",
        "公开来源中间编辑衍生负载；8 条题链、各 4 次编辑、单次测量。"
        "加了官方分块分隔符，不能与之前原型的耗时直接比较；不是论文原始负载复现。",
        "各 arm 独立启动服务并播种原始上下文，随后按版本连续编辑；"
        "CacheBlend 使用上游实际缓存管理行为，不另加私有缓存链策略。",
        "CacheBlend 要求融合块已经存在，因此先单独生成新 B′ 块的缓存；"
        "这次额外请求的完整耗时计入客户端 TTFT，并另存 main_request_ttft_ms。"
        "启用上游 save_unfull_chunk 选项，保证末尾 C 的完整块可再次命中。",
        "HTTP token-ID 接口：从客户端发请求至首个生成 token，含 API、通信、"
        "排队、检索、搬运、选择和重算；不含初次播种、分词与后续解码。",
        "",
        "| 类别 | 策略 | 事件 | 正确率 | 参考正确 | 缓存新增错误 "
        "| TTFT均值 ms | P95 ms |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for g in summary:
        lines.append(
            f"| {g['family']} | {g['arm']} | {g['events']} "
            f"| {g['task_accuracy']:.1%} | {g['reference_correct_events']} "
            f"| {g['cache_induced_errors']} | {g['mean_http_ttft_ms']:.2f} "
            f"| {g['p95_http_ttft_ms']:.2f} |"
        )
    lines += [
        "",
        "每类仅两条独立题链，不代表总体精度。完整重算质量低的类别"
        "保留结果，但不能证明精度保持。尚未扫描预算或验证 sparse 后端，"
        "不能据此声称我们的策略优于 CacheBlend。",
        "",
    ]
    (root / "report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18265)
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()
    if Path(sys.prefix).resolve() != PROJECT / ".venv":
        raise RuntimeError("use dedicated venv")
    data_path = PROJECT / "data/pilot.json"
    locked = json.loads((PROJECT / "data.lock.json").read_text())
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if digest != locked["input_sha256"]:
        raise ValueError("pilot input changed")
    data = json.loads(data_path.read_text())
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    for script in ("verify_sources.py", "verify_runtime.py"):
        subprocess.run([sys.executable, str(PROJECT / "scripts" / script)], check=True)
    manifest = {
        "status": "planned",
        "input_sha256": digest,
        "scope": "single-flight token-ID HTTP pilot, adapted public tasks",
        "config": {
            "gpu": args.gpu,
            "port": args.port,
            "ratio": 0.15,
            "check_layers": [1],
            "sparse": False,
        },
    }
    env = os.environ.copy()
    for name in list(env):
        if name.startswith("LMCACHE_") or name in ("PYTHONPATH", "PYTHONHOME"):
            env.pop(name)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONHASHSEED": "0",
            "VLLM_USE_V1": "1",
            "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
            "VLLM_NO_USAGE_STATS": "1",
            "DO_NOT_TRACK": "1",
            "LMCACHE_TRACK_USAGE": "false",
            "LMCACHE_LOG_LEVEL": "INFO",
            "XDG_CACHE_HOME": str(PROJECT / ".cache"),
            "VLLM_CACHE_ROOT": str(PROJECT / ".cache/vllm"),
            "PATH": str(PROJECT / ".venv/bin") + os.pathsep + env.get("PATH", ""),
        }
    )
    rows = []
    try:
        for arm in ARMS:
            before = gpu_state(args.gpu)
            if before["pids"] or before["memory_mib"] > 100:
                raise RuntimeError("GPU is busy")
            mode_env = env.copy()
            alias = f"cacheblend-pilot-{arm}-{uuid.uuid4().hex}"
            command = [
                sys.executable,
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                data["model"],
                "--host",
                "127.0.0.1",
                "--port",
                str(args.port),
                "--max-model-len",
                "4096",
                "--max-num-batched-tokens",
                "4096",
                "--max-num-seqs",
                "1",
                "--gpu-memory-utilization",
                "0.45",
                "--enforce-eager",
                "--no-enable-chunked-prefill",
                "--disable-log-requests",
                "--generation-config",
                "vllm",
                "--served-model-name",
                alias,
            ]
            command += [
                "--enable-prefix-caching"
                if arm == "prefix"
                else "--no-enable-prefix-caching"
            ]
            if arm == "cacheblend15":
                mode_env.update(
                    {
                        "LMCACHE_ENABLE_BLENDING": "True",
                        "LMCACHE_BLEND_SPECIAL_STR": "# #",
                        "LMCACHE_USE_LAYERWISE": "True",
                        "LMCACHE_BLEND_CHECK_LAYERS": "1",
                        "LMCACHE_BLEND_RECOMPUTE_RATIOS": "0.15",
                        "LMCACHE_CHUNK_SIZE": "256",
                        "LMCACHE_LOCAL_CPU": "True",
                        "LMCACHE_MAX_LOCAL_CPU_SIZE": "5",
                        "LMCACHE_SAVE_UNFULL_CHUNK": "True",
                    }
                )
                command += [
                    "--kv-transfer-config",
                    json.dumps(
                        {"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}
                    ),
                ]
            child = None
            try:
                with (root / f"{arm}.log").open("w") as log:
                    child = subprocess.Popen(
                        command,
                        cwd=PROJECT,
                        env=mode_env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    manifest.update(
                        status="running", current_arm=arm, server_pid=child.pid
                    )
                    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
                    rows += asyncio.run(measure(args, data, root, arm, child, alias))
            finally:
                if child is not None:
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    # Wait for our engine subprocess to release CUDA before next arm.
                    for _ in range(20):
                        if not gpu_state(args.gpu)["pids"]:
                            break
                        time.sleep(0.5)
        text = (root / "cacheblend15.log").read_text()
        hits = {
            rid: int(count)
            for rid, count in re.findall(
                r"Reqid: (\S+), Total tokens \d+, LMCache hit tokens: (\d+)", text
            )
        }
        verified = 0
        for row in rows:
            if row["arm"] == "cacheblend15":
                count = hits.get(row["request_id"] + "-0", -1)
                row["lmcache_hit_tokens"] = count
                if count > row["suffix_start"]:
                    verified += 1
                else:
                    raise RuntimeError("CacheBlend did not reach the unchanged C chunk")
        (root / "results.json").write_text(json.dumps(rows, indent=2))
        report(root, rows)
        manifest.update(
            status="completed",
            events=len(rows),
            verified_nonprefix_events=verified,
            new_chunk_cost_included=True,
        )
    except BaseException as exc:
        manifest.update(status="failed", error=repr(exc))
        raise
    finally:
        manifest["gpu_after"] = gpu_state(args.gpu)
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
