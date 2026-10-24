"""Compare native full/prefix and official CacheBlend through one HTTP interface."""

import argparse
import asyncio
import hashlib
import json
import os
import re
import shutil
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
        for session_index, session in enumerate(data["sessions"]):
            if session_index % 10 == 0:
                state = gpu_state(args.gpu)
                with (root / "gpu_samples.jsonl").open("a") as stream:
                    stream.write(json.dumps({"arm": arm, **state}) + "\n")
                for pid in state["pids"]:
                    if os.getpgid(pid) != server.pid:
                        raise RuntimeError("another process started using the test GPU")
            await request(client, alias, session["versions"][0]["token_ids"])
            for version, item in enumerate(session["versions"][1:], 1):
                pipeline_started = time.perf_counter()
                preparation = None
                if arm.startswith("cacheblend"):
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
                    source_id=session.get("source_id", session["session_id"]),
                    placement=session.get("placement", "pilot"),
                    variant=session.get("variant", "original"),
                    prompt_tokens=session["prompt_tokens"],
                    suffix_tokens=session.get("suffix_tokens"),
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


def report(root, rows, arms=ARMS):
    references = {
        (r["session_id"], r["version"]): r for r in rows if r["arm"] == "full"
    }
    summary = []
    for family in ("all", *sorted({r["family"] for r in rows})):
        for arm in arms:
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
        "公开来源中间编辑衍生负载；题链及编辑数量见输入 manifest。"
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
        "独立题链数量有限，不代表总体精度。完整重算质量低的类别"
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
    parser.add_argument("--input", type=Path, default=PROJECT / "data/pilot.json")
    parser.add_argument("--lock", type=Path, default=PROJECT / "data.lock.json")
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--controls-from", type=Path)
    parser.add_argument(
        "--attention-backend", choices=("flashattn", "sparse"), default="flashattn"
    )
    parser.add_argument(
        "--sparse-correctness-fix", action="store_true", dest="sparse_index_fix"
    )
    parser.add_argument("--strict-causal", action="store_true")
    parser.add_argument("--precheck-kv-write", action="store_true")
    parser.add_argument("--query-tail-tokens", type=int, choices=(0, 64))
    parser.add_argument(
        "--query-control",
        choices=("topk", "query", "tail64", "pre_tail64", "random64", "boundary"),
    )
    parser.add_argument("--query-fair", action="store_true")
    parser.add_argument(
        "--query-template", choices=("topk", "none", "question", "template", "both")
    )
    parser.add_argument("--query-boundary", choices=("topk", "query", "fresh"))
    args = parser.parse_args()
    if args.query_boundary is not None and (
        args.query_template is not None
        or args.query_control is not None
        or args.query_tail_tokens is not None
        or not (args.strict_causal and args.precheck_kv_write)
    ):
        parser.error("boundary control requires strict/common corrections only")
    if args.query_template is not None and (
        args.query_control is not None
        or args.query_fair
        or args.query_tail_tokens is not None
        or not (args.strict_causal and args.precheck_kv_write)
    ):
        parser.error(
            "question/template controls require strict/common corrections "
            "and no other selector patch"
        )
    if args.query_fair and args.query_control not in ("query", "tail64", "boundary"):
        parser.error("fair comparison requires a supported selector")
    if args.query_control == "boundary" and not args.query_fair:
        parser.error("boundary requires fair comparison patch")
    if args.query_control is not None and (
        args.query_tail_tokens is not None
        or not (args.strict_causal and args.precheck_kv_write)
    ):
        parser.error(
            "query controls require strict/common correction and no query-tail patch"
        )
    if args.query_tail_tokens is not None and not (
        args.strict_causal and args.precheck_kv_write
    ):
        parser.error("query-tail requires strict mask and common KV correction")
    if args.strict_causal and (
        args.attention_backend != "sparse" or args.sparse_index_fix
    ):
        parser.error("strict mask requires FlashInfer and no block-mask patch")
    if Path(sys.prefix).resolve() != PROJECT / ".venv":
        raise RuntimeError("use dedicated venv")
    data_path = args.input.resolve()
    locked = json.loads(args.lock.read_text())
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if digest != locked["input_sha256"]:
        raise ValueError("pilot input changed")
    data = json.loads(data_path.read_text())
    if "full" not in args.arms or "prefix" not in args.arms:
        raise ValueError("full and prefix controls are required")
    for arm in args.arms:
        if arm not in ("full", "prefix"):
            if not re.fullmatch(r"cacheblend\d+", arm):
                raise ValueError(f"unsupported arm: {arm}")
            if not 0 < int(arm.removeprefix("cacheblend")) <= 100:
                raise ValueError("invalid blend ratio")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(data_path, root / "input.json")
    shutil.copyfile(args.lock, root / "input.lock.json")
    snapshot = root / "source_snapshot"
    snapshot.mkdir()
    for path in (PROJECT / "scripts").glob("*.py"):
        shutil.copyfile(path, snapshot / path.name)
    for name in ("sources.lock.json", "requirements.lock"):
        shutil.copyfile(PROJECT / name, snapshot / name)
    for script in ("verify_sources.py", "verify_runtime.py"):
        verification = [sys.executable, str(PROJECT / "scripts" / script)]
        if script == "verify_runtime.py" and args.sparse_index_fix:
            verification.append("--allow-sparse-correctness-fix")
        if script == "verify_runtime.py" and args.strict_causal:
            verification.append("--allow-strict-causal")
        if script == "verify_runtime.py" and args.precheck_kv_write:
            verification.append("--allow-precheck-kv-write")
        if script == "verify_runtime.py" and args.query_tail_tokens is not None:
            verification.append("--allow-query-tail")
        if script == "verify_runtime.py" and args.query_control is not None:
            verification.append(
                "--allow-query-fair" if args.query_fair else "--allow-query-controls"
            )
        if script == "verify_runtime.py" and args.query_template is not None:
            verification.append("--allow-query-template")
        if script == "verify_runtime.py" and args.query_boundary is not None:
            verification.append("--allow-query-boundary")
        subprocess.run(verification, check=True)
    manifest = {
        "status": "planned",
        "input_sha256": digest,
        "scope": "single-flight token-ID HTTP pilot, adapted public tasks",
        "config": {
            "gpu": args.gpu,
            "port": args.port,
            "ratios": {
                arm: int(arm.removeprefix("cacheblend")) / 100
                for arm in args.arms
                if arm.startswith("cacheblend")
            },
            "check_layers": [1],
            "sparse": args.attention_backend == "sparse",
            "sparse_correctness_fix": args.sparse_index_fix,
            "strict_causal": args.strict_causal,
            "precheck_kv_write": args.precheck_kv_write,
            "query_tail_tokens": args.query_tail_tokens,
            "query_control": args.query_control,
            "query_fair": args.query_fair,
            "query_template": args.query_template,
            "query_boundary": args.query_boundary,
            "arms": args.arms,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
        },
        "source_snapshot_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in snapshot.iterdir()
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
            "VLLM_ATTENTION_BACKEND": (
                "FLASHINFER" if args.attention_backend == "sparse" else "FLASH_ATTN"
            ),
            "VLLM_NO_USAGE_STATS": "1",
            "DO_NOT_TRACK": "1",
            "LMCACHE_TRACK_USAGE": "false",
            "LMCACHE_LOG_LEVEL": "INFO",
            "XDG_CACHE_HOME": str(PROJECT / ".cache"),
            "VLLM_CACHE_ROOT": str(PROJECT / ".cache/vllm"),
            "PATH": str(PROJECT / ".venv/bin") + os.pathsep + env.get("PATH", ""),
        }
    )
    if args.attention_backend == "sparse":
        env.update(
            {
                "CUDA_HOME": "/usr/local/cuda",
                "MAX_JOBS": "4",
                "FLASHINFER_WORKSPACE_DIR": str(PROJECT / ".cache/flashinfer"),
                "TORCH_CUDA_ARCH_LIST": "8.0",
                "PATH": "/usr/local/cuda/bin" + os.pathsep + env["PATH"],
            }
        )
    rows = []
    if args.controls_from:
        control_manifest = json.loads(
            (args.controls_from / "manifest.json").read_text()
        )
        if control_manifest["input_sha256"] != digest:
            raise ValueError("control source input differs")
        config = control_manifest["config"]
        for field in ("max_num_seqs", "max_model_len", "sparse"):
            if config[field] != manifest["config"][field]:
                raise ValueError(f"control source config differs: {field}")
        control_path = args.controls_from / "requests.jsonl"
        rows = [
            json.loads(line)
            for line in control_path.read_text().splitlines()
            if json.loads(line)["arm"] in ("full", "prefix")
        ]
        expected = {
            (s["session_id"], v)
            for s in data["sessions"]
            for v in range(1, len(s["versions"]))
        }
        for arm in ("full", "prefix"):
            control_rows = [r for r in rows if r["arm"] == arm]
            if (
                len(control_rows) != len(expected)
                or {(r["session_id"], r["version"]) for r in control_rows} != expected
            ):
                raise ValueError("control source is incomplete")
        (root / "requests.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
        manifest["controls_from"] = str(args.controls_from.resolve())
        manifest["controls_sha256"] = hashlib.sha256(
            control_path.read_bytes()
        ).hexdigest()
    try:
        for arm in args.arms:
            if args.controls_from and arm in ("full", "prefix"):
                continue
            before = gpu_state(args.gpu)
            if before["pids"] or before["memory_mib"] > 100:
                raise RuntimeError("GPU is busy")
            mode_env = env.copy()
            if args.query_tail_tokens is not None:
                mode_env["LMCACHE_QUERY_TAIL_TOKENS"] = str(args.query_tail_tokens)
            if args.query_control is not None:
                mode_env["LMCACHE_QUERY_CONTROL"] = args.query_control
            if args.query_template is not None:
                mode_env["LMCACHE_QUERY_TEMPLATE"] = args.query_template
            if args.query_boundary is not None:
                mode_env["LMCACHE_QUERY_BOUNDARY"] = args.query_boundary
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
                str(args.max_model_len),
                "--max-num-batched-tokens",
                str(args.max_model_len),
                "--max-num-seqs",
                str(args.max_num_seqs),
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
            if arm.startswith("cacheblend"):
                if args.attention_backend == "sparse":
                    mode_env["LMCACHE_EXTRA_CONFIG"] = '{"enable_sparse": true}'
                mode_env.update(
                    {
                        "LMCACHE_ENABLE_BLENDING": "True",
                        "LMCACHE_BLEND_SPECIAL_STR": data["pilot_provenance"].get(
                            "separator_text", "# #"
                        ),
                        "LMCACHE_USE_LAYERWISE": "True",
                        "LMCACHE_BLEND_CHECK_LAYERS": "1",
                        "LMCACHE_BLEND_RECOMPUTE_RATIOS": str(
                            int(arm.removeprefix("cacheblend")) / 100
                        ),
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
        hits = {}
        for arm in args.arms:
            if arm.startswith("cacheblend"):
                text = (root / f"{arm}.log").read_text()
                hits.update(
                    {
                        rid: int(count)
                        for rid, count in re.findall(
                            r"Reqid: (\S+), Total tokens \d+, "
                            r"LMCache hit tokens: (\d+)",
                            text,
                        )
                    }
                )
        verified = 0
        for row in rows:
            if row["arm"].startswith("cacheblend"):
                count = hits.get(row["request_id"] + "-0", -1)
                row["lmcache_hit_tokens"] = count
                if count > row["suffix_start"]:
                    verified += 1
                else:
                    raise RuntimeError("CacheBlend did not reach the unchanged C chunk")
        (root / "results.json").write_text(json.dumps(rows, indent=2))
        report(root, rows, args.arms)
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
