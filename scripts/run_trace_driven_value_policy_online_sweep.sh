#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-${ROOT_DIR}/.venv/bin/python}"
VLLM_BIN="${VLLM_BIN:-${ROOT_DIR}/.venv/bin/vllm}"
MODEL_PATH="${MODEL_PATH:-/public/model_zoo/Qwen3-0.6B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-0.6b}"
GPU_ID="${GPU_ID:-0}"
HOST="${HOST:-127.0.0.1}"
PORT_BASE="${PORT_BASE:-8132}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.05}"
NUM_GPU_BLOCKS_OVERRIDE="${NUM_GPU_BLOCKS_OVERRIDE:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
INPUT_FILE="${INPUT_FILE:-${ROOT_DIR}/results/trace_driven_sharegpt_multiturn_20260623/sharegpt_burstgpt_trace_128.json}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT_DIR}/results/trace_driven_sharegpt_burstgpt_policy_compare_online}"
SEED="${SEED:-0}"
NUM_CLIENTS="${NUM_CLIENTS:-4}"
MAX_ACTIVE_CONVERSATIONS="${MAX_ACTIVE_CONVERSATIONS:-16}"
MAX_NUM_REQUESTS="${MAX_NUM_REQUESTS:-128}"
COMPLETE_WORKLOAD="${COMPLETE_WORKLOAD:-0}"
CONVERSATION_SAMPLING="${CONVERSATION_SAMPLING:-round_robin}"
REQUEST_RATE="${REQUEST_RATE:-0}"
LIMIT_MIN_TOKENS="${LIMIT_MIN_TOKENS:-16}"
LIMIT_MAX_TOKENS="${LIMIT_MAX_TOKENS:-32}"
SERVER_WAIT_SEC="${SERVER_WAIT_SEC:-300}"
CANDIDATE_WINDOW="${CANDIDATE_WINDOW:-64}"
MIN_USAGE="${MIN_USAGE:-0.20}"
HOTNESS_HALF_LIFE_S="${HOTNESS_HALF_LIFE_S:-30.0}"
MIN_REORDER_HIT_BLOCKS="${MIN_REORDER_HIT_BLOCKS:-0}"
MIN_REORDER_SCORE_MARGIN="${MIN_REORDER_SCORE_MARGIN:-0.0}"
KV_TRANSFER_CONFIG_JSON="${KV_TRANSFER_CONFIG_JSON:-}"

POLICIES=(
  "lru:lru:1.0:1.0:1.0:1.0:0:0.0"
  "prefix_only:prefix_only:1.0:1.0:1.0:1.0:0:0.0"
  "hotness_only:hotness_only:1.0:1.0:1.0:1.0:0:0.0"
  "hybrid_hot1:hybrid:1.0:1.0:1.0:1.0:0:0.0"
  "hybrid_hot2:hybrid:1.0:2.0:1.0:1.0:0:0.0"
  "chain_hybrid:chain_hybrid:1.0:1.0:1.0:1.0:0:0.0"
  "chain_share025:chain_hybrid:1.0:0.0:0.0:0.25:0:0.0"
  "chain_share05:chain_hybrid:1.0:0.0:0.0:0.5:0:0.0"
  "chain_share1:chain_hybrid:1.0:0.0:0.0:1.0:0:0.0"
  "chain_share2:chain_hybrid:1.0:0.0:0.0:2.0:0:0.0"
  "shared_chain1:shared_chain_hybrid:1.0:0.0:0.0:1.0:0:0.0"
  "shared_chain1_hit8:shared_chain_hybrid:1.0:0.0:0.0:1.0:8:0.0"
  "shared_chain1_hit16:shared_chain_hybrid:1.0:0.0:0.0:1.0:16:0.0"
  "shared_chain1_hit20:shared_chain_hybrid:1.0:0.0:0.0:1.0:20:0.0"
  "shared_chain1_hit24:shared_chain_hybrid:1.0:0.0:0.0:1.0:24:0.0"
  "shared_chain1_hit28:shared_chain_hybrid:1.0:0.0:0.0:1.0:28:0.0"
  "shared_chain1_hit32:shared_chain_hybrid:1.0:0.0:0.0:1.0:32:0.0"
  "shared_chain1_hit32_margin2:shared_chain_hybrid:1.0:0.0:0.0:1.0:32:2.0"
  "shared_chain2:shared_chain_hybrid:1.0:0.0:0.0:2.0:0:0.0"
)

usage() {
  cat <<EOF
Usage: $0 [policy_label ...]

Known labels: lru prefix_only hotness_only hybrid_hot1 hybrid_hot2 chain_hybrid
              chain_share025 chain_share05 chain_share1 chain_share2
              shared_chain1 shared_chain1_hit8 shared_chain1_hit16
              shared_chain1_hit20 shared_chain1_hit24 shared_chain1_hit28
              shared_chain1_hit32 shared_chain1_hit32_margin2 shared_chain2

Environment overrides:
  PYTHON VLLM_BIN MODEL_PATH SERVED_MODEL_NAME GPU_ID HOST PORT_BASE
  INPUT_FILE RESULT_ROOT SEED NUM_CLIENTS MAX_ACTIVE_CONVERSATIONS MAX_NUM_REQUESTS
  GPU_MEMORY_UTILIZATION NUM_GPU_BLOCKS_OVERRIDE MAX_MODEL_LEN REQUEST_RATE
  CONVERSATION_SAMPLING COMPLETE_WORKLOAD (1: no request cap, fixed client assignment)
  LIMIT_MIN_TOKENS LIMIT_MAX_TOKENS SERVER_WAIT_SEC CANDIDATE_WINDOW MIN_USAGE
  MIN_REORDER_HIT_BLOCKS MIN_REORDER_SCORE_MARGIN KV_TRANSFER_CONFIG_JSON
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -f "${INPUT_FILE}" ]]; then
  echo "input file not found: ${INPUT_FILE}" >&2
  exit 1
fi

mkdir -p "${RESULT_ROOT}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RESULT_ROOT}/${TIMESTAMP}"
mkdir -p "${RUN_ROOT}/seed_${SEED}"

cleanup_pid() {
  local pid="${1:-}"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
  fi
}

probe_server() {
  local url="$1"
  curl --noproxy '*' -sSf --connect-timeout 1 --max-time 2 \
    "${url}/v1/models" >/dev/null 2>&1
}

write_json_config() {
  local path="$1"
  local label="$2"
  local policy="$3"
  local share_weight="$4"
  local hotness_weight="$5"
  local restore_weight="$6"
  local chain_weight="$7"
  local min_reorder_hit_blocks="$8"
  local min_reorder_score_margin="$9"
  "${PYTHON}" - "$path" "$label" "$policy" "$share_weight" \
    "$hotness_weight" "$restore_weight" "$chain_weight" \
    "$min_reorder_hit_blocks" "$min_reorder_score_margin" <<'PY'
import json
import sys

path, label, policy, share, hotness, restore, chain, min_hit, margin = sys.argv[1:]
payload = {
    "policy": label,
    "gpu_eviction_policy": policy,
    "share_weight": float(share),
    "hotness_weight": float(hotness),
    "restore_weight": float(restore),
    "chain_weight": float(chain),
    "min_reorder_hit_blocks": int(min_hit),
    "min_reorder_score_margin": float(margin),
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=True, indent=2)
PY
}

run_one() {
  local label="$1"
  local port_offset="$2"
  local policy="$3"
  local share_weight="$4"
  local hotness_weight="$5"
  local restore_weight="$6"
  local chain_weight="$7"
  local min_reorder_hit_blocks="$8"
  local min_reorder_score_margin="$9"

  local port=$((PORT_BASE + port_offset))
  local url="http://${HOST}:${port}"
  local run_dir="${RUN_ROOT}/seed_${SEED}/${label}"
  local server_log="${run_dir}/server.log"
  local bench_log="${run_dir}/bench.log"
  mkdir -p "${run_dir}"

  write_json_config "${run_dir}/gpu_eviction_policy.json" "${label}" \
    "${policy}" "${share_weight}" "${hotness_weight}" \
    "${restore_weight}" "${chain_weight}" "${min_reorder_hit_blocks}" \
    "${min_reorder_score_margin}"

  "${PYTHON}" - "${run_dir}/run_config.json" <<PY
import json
import sys

config = {
    "model_path": "${MODEL_PATH}",
    "served_model_name": "${SERVED_MODEL_NAME}",
    "input_file": "${INPUT_FILE}",
    "seed": int("${SEED}"),
    "num_clients": int("${NUM_CLIENTS}"),
    "max_active_conversations": int("${MAX_ACTIVE_CONVERSATIONS}"),
    "max_num_requests": None if "${COMPLETE_WORKLOAD}" == "1" else int("${MAX_NUM_REQUESTS}"),
    "complete_workload": "${COMPLETE_WORKLOAD}" == "1",
    "gpu_memory_utilization": float("${GPU_MEMORY_UTILIZATION}"),
    "num_gpu_blocks_override": (
        int("${NUM_GPU_BLOCKS_OVERRIDE}")
        if "${NUM_GPU_BLOCKS_OVERRIDE}"
        else None
    ),
    "max_model_len": int("${MAX_MODEL_LEN}"),
    "request_rate": float("${REQUEST_RATE}"),
    "conversation_sampling": "${CONVERSATION_SAMPLING}",
    "limit_min_tokens": int("${LIMIT_MIN_TOKENS}"),
    "limit_max_tokens": int("${LIMIT_MAX_TOKENS}"),
    "min_reorder_hit_blocks": int("${min_reorder_hit_blocks}"),
    "min_reorder_score_margin": float("${min_reorder_score_margin}"),
    "host": "${HOST}",
    "port": int("${port}"),
    "kv_transfer_config_json": ${KV_TRANSFER_CONFIG_JSON@Q},
}
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=True, indent=2)
PY

  local kv_transfer_args=()
  if [[ -n "${KV_TRANSFER_CONFIG_JSON}" ]]; then
    kv_transfer_args=(--kv-transfer-config "${KV_TRANSFER_CONFIG_JSON}")
  fi

  local gpu_block_args=()
  if [[ -n "${NUM_GPU_BLOCKS_OVERRIDE}" ]]; then
    gpu_block_args=(--num-gpu-blocks-override "${NUM_GPU_BLOCKS_OVERRIDE}")
  fi

  echo "==> ${label}: starting server on ${url}"
  env \
    VLLM_SERVER_DEV_MODE=1 \
    VLLM_EXPERIMENTAL_GPU_EVICTION_POLICY="${policy}" \
    VLLM_EXPERIMENTAL_GPU_EVICTION_SHARE_WEIGHT="${share_weight}" \
    VLLM_EXPERIMENTAL_GPU_EVICTION_HOTNESS_WEIGHT="${hotness_weight}" \
    VLLM_EXPERIMENTAL_GPU_EVICTION_RESTORE_WEIGHT="${restore_weight}" \
    VLLM_EXPERIMENTAL_GPU_EVICTION_CHAIN_WEIGHT="${chain_weight}" \
    VLLM_EXPERIMENTAL_GPU_EVICTION_HOTNESS_HALF_LIFE_S="${HOTNESS_HALF_LIFE_S}" \
    VLLM_EXPERIMENTAL_GPU_EVICTION_CANDIDATE_WINDOW="${CANDIDATE_WINDOW}" \
    VLLM_EXPERIMENTAL_GPU_EVICTION_MIN_USAGE="${MIN_USAGE}" \
    VLLM_EXPERIMENTAL_GPU_EVICTION_MIN_REORDER_HIT_BLOCKS="${min_reorder_hit_blocks}" \
    VLLM_EXPERIMENTAL_GPU_EVICTION_MIN_REORDER_SCORE_MARGIN="${min_reorder_score_margin}" \
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    "${VLLM_BIN}" serve "${MODEL_PATH}" \
      --host "${HOST}" \
      --port "${port}" \
      --served-model-name "${SERVED_MODEL_NAME}" \
      --trust-remote-code \
      --max-model-len "${MAX_MODEL_LEN}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      "${gpu_block_args[@]}" \
      --enable-prefix-caching \
      "${kv_transfer_args[@]}" \
      >"${server_log}" 2>&1 &
  local server_pid=$!

  for ((i = 0; i < SERVER_WAIT_SEC; ++i)); do
    if probe_server "${url}"; then
      break
    fi
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      echo "server exited before readiness; see ${server_log}" >&2
      return 1
    fi
    sleep 1
  done

  if ! probe_server "${url}"; then
    cleanup_pid "${server_pid}"
    echo "server did not become ready; see ${server_log}" >&2
    return 1
  fi

  echo "==> ${label}: running multi-turn benchmark"
  local workload_args=(--max-num-requests "${MAX_NUM_REQUESTS}")
  if [[ "${COMPLETE_WORKLOAD}" == "1" ]]; then
    workload_args=(--fixed-client-assignment)
  fi
  set +e
  "${PYTHON}" benchmarks/multi_turn/benchmark_serving_multi_turn.py \
    --model "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --url "${url}" \
    --input-file "${INPUT_FILE}" \
    --output-file "${run_dir}/output.json" \
    --request-summary-output "${run_dir}/request_summary.csv" \
    --value-policy-stats-output "${run_dir}/value_policy_stats.json" \
    --num-clients "${NUM_CLIENTS}" \
    --max-active-conversations "${MAX_ACTIVE_CONVERSATIONS}" \
    "${workload_args[@]}" \
    --no-early-stop \
    --conversation-sampling "${CONVERSATION_SAMPLING}" \
    --limit-min-tokens "${LIMIT_MIN_TOKENS}" \
    --limit-max-tokens "${LIMIT_MAX_TOKENS}" \
    --preserve-dataset-assistant-history \
    --request-rate "${REQUEST_RATE}" \
    --seed "${SEED}" \
    >"${bench_log}" 2>&1
  local bench_status=$?
  set -e

  cleanup_pid "${server_pid}"
  if [[ "${bench_status}" -ne 0 ]]; then
    echo "benchmark failed for ${label}; see ${bench_log}" >&2
    return "${bench_status}"
  fi
  echo "==> ${label}: done; result ${run_dir}"
}

cd "${ROOT_DIR}"

requested_labels=" $* "
idx=0
for spec in "${POLICIES[@]}"; do
  IFS=":" read -r label policy share_weight hotness_weight restore_weight chain_weight min_reorder_hit_blocks min_reorder_score_margin <<<"${spec}"
  min_reorder_hit_blocks="${min_reorder_hit_blocks:-${MIN_REORDER_HIT_BLOCKS}}"
  min_reorder_score_margin="${min_reorder_score_margin:-${MIN_REORDER_SCORE_MARGIN}}"
  if [[ $# -gt 0 && "${requested_labels}" != *" ${label} "* ]]; then
    continue
  fi
  run_one "${label}" "${idx}" "${policy}" "${share_weight}" \
    "${hotness_weight}" "${restore_weight}" "${chain_weight}" \
    "${min_reorder_hit_blocks}" "${min_reorder_score_margin}"
  idx=$((idx + 1))
done

"${PYTHON}" scripts/summarize_value_policy_runs.py \
  --root "${RUN_ROOT}" \
  --output-csv "${RUN_ROOT}/policy_compare_summary.csv" \
  --output-md "${RUN_ROOT}/policy_compare_summary.md"

echo "all requested runs finished under ${RUN_ROOT}"
