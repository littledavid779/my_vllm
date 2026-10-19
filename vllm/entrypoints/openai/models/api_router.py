# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _with_value_policy_derived_metrics(
    stats: dict[str, Any],
) -> dict[str, Any]:
    enriched = dict(stats)
    query_count = int(stats.get("prefix_cache_query_count", 0) or 0)
    evicted_blocks = int(stats.get("evicted_blocks", 0) or 0)
    prefix_index_count = int(stats.get("evicted_prefix_index_count", 0) or 0)
    last_hit_count = int(
        stats.get("evicted_seconds_since_last_apc_hit_count", 0) or 0
    )
    enriched["avg_effective_prefix_hit_blocks"] = _safe_div(
        stats.get("effective_prefix_hit_blocks_sum", 0) or 0, query_count
    )
    enriched["avg_effective_prefix_hit_tokens"] = _safe_div(
        stats.get("effective_prefix_hit_tokens_sum", 0) or 0, query_count
    )
    enriched["prefix_cache_zero_hit_ratio"] = _safe_div(
        stats.get("prefix_cache_zero_hit_count", 0) or 0, query_count
    )
    enriched["avg_evicted_apc_hit_count"] = _safe_div(
        stats.get("evicted_apc_hit_count_sum", 0) or 0, evicted_blocks
    )
    enriched["avg_evicted_prefix_index"] = _safe_div(
        stats.get("evicted_prefix_index_sum", 0) or 0, prefix_index_count
    )
    enriched["avg_evicted_chain_downstream_blocks"] = _safe_div(
        stats.get("evicted_chain_downstream_blocks_sum", 0) or 0, evicted_blocks
    )
    enriched["avg_seconds_since_last_apc_hit_on_eviction"] = _safe_div(
        stats.get("evicted_seconds_since_last_apc_hit_sum", 0.0) or 0.0,
        last_hit_count,
    )
    return enriched


def models(request: Request) -> OpenAIServingModels:
    return request.app.state.openai_serving_models


@router.get("/v1/models")
async def show_available_models(raw_request: Request):
    handler = models(raw_request)

    models_ = await handler.show_available_models()
    return JSONResponse(content=models_.model_dump())


@router.get("/debug/value_policy_stats")
async def show_value_policy_stats(raw_request: Request, reset: bool = False):
    engine_client = raw_request.app.state.engine_client
    get_stats = getattr(engine_client, "get_value_policy_stats", None)
    if get_stats is None:
        raise HTTPException(
            status_code=404,
            detail="value policy stats are not available for this engine",
        )

    try:
        stats = await get_stats(reset=reset)
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=404,
            detail="value policy stats are not implemented for this engine",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return JSONResponse(
        content={
            "source": "online_engine_core.scheduler.kv_cache_manager",
            "stats": _with_value_policy_derived_metrics(stats),
        }
    )


def attach_router(app: FastAPI):
    app.include_router(router)
