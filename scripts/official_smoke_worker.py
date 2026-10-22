"""Execute the unmodified upstream example with bounded hardware settings."""

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    os.environ.update(
        {
            "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
            "LMCACHE_LOG_LEVEL": "DEBUG",
            "LMCACHE_TRACK_USAGE": "false",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    import lmcache
    import vllm

    for package in (lmcache, vllm):
        if not Path(package.__file__).resolve().is_relative_to(PROJECT / ".venv"):
            raise RuntimeError(
                f"import escaped isolated environment: {package.__file__}"
            )
    path = PROJECT / "upstream/lmcache/examples/blend_kv_v1/blend.py"
    spec = importlib.util.spec_from_file_location("official_cacheblend_example", path)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    rows = []

    @contextlib.contextmanager
    def bounded_llm(connector, model):
        ktc = official.KVTransferConfig(kv_connector=connector, kv_role="kv_both")
        llm_args = official.EngineArgs(
            model=model,
            kv_transfer_config=ktc,
            max_model_len=config["max_model_len"],
            gpu_memory_utilization=config["gpu_memory_utilization"],
            max_num_seqs=config["max_num_seqs"],
            max_num_batched_tokens=config["max_model_len"],
            enable_chunked_prefill=False,
            enable_prefix_caching=False,
            enforce_eager=True,
        )
        llm = official.LLM(**asdict(llm_args))
        try:
            yield llm
        finally:
            official.LMCacheEngineBuilder.destroy(official.ENGINE_NAME)

    def measured_output(llm, prompt, sampling_params, req_str):
        started = time.perf_counter()
        outputs = llm.generate(
            prompts={"prompt_token_ids": prompt},
            sampling_params=sampling_params,
        )
        elapsed = (time.perf_counter() - started) * 1000
        output = outputs[0]
        metrics = output.metrics
        row = {
            "stage": req_str,
            "prompt_tokens": len(prompt),
            "input_sha256": hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
            "one_token_generate_wall_ms": elapsed,
            "generated_ids": list(output.outputs[0].token_ids),
            "generated_text": output.outputs[0].text,
            "engine_ttft_ms": (metrics.first_token_time - metrics.arrival_time) * 1000
            if metrics is not None and metrics.first_token_time is not None
            else None,
            "note": "Offline one-token smoke timing with debug logging; not HTTP TTFT.",
        }
        rows.append(row)
        args.output.write_text(json.dumps({"rows": rows, "config": config}, indent=2))
        print("CACHEBLEND_SMOKE_ROW " + json.dumps(row), flush=True)

    official.build_llm_with_lmcache = bounded_llm
    official.print_output = measured_output
    sys.argv = [str(path), "--model", config["model"]]
    if config["enable_sparse"]:
        sys.argv += ["--enable-sparse"]
    # The example hardcodes these; the checked-in config must agree.
    if config["blend_recompute_ratio"] != 0.15 or config["blend_check_layers"] != [1]:
        raise ValueError("smoke runner preserves the official example's selector")
    official.main()
    if len(rows) != 4 or any(len(row["generated_ids"]) != 1 for row in rows):
        raise RuntimeError(
            "official example did not complete all four one-token stages"
        )


if __name__ == "__main__":
    main()
