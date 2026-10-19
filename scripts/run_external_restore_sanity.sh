#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-${ROOT_DIR}/.venv/bin/python}"
VLLM_BIN="${VLLM_BIN:-${ROOT_DIR}/.venv/bin/vllm}"
MODEL_PATH="${MODEL_PATH:-/public/model_zoo/Qwen3-0.6B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-0.6b}"
GPU_ID="${GPU_ID:-0}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8140}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.05}"
NUM_GPU_BLOCKS_OVERRIDE="${NUM_GPU_BLOCKS_OVERRIDE:-128}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
CPU_BYTES_TO_USE="${CPU_BYTES_TO_USE:-1073741824}"
STORE_THRESHOLD="${STORE_THRESHOLD:-0}"
MAX_TRACKER_SIZE="${MAX_TRACKER_SIZE:-64000}"
TARGET_PROMPT_TOKENS="${TARGET_PROMPT_TOKENS:-512}"
POLLUTER_TARGET_PROMPT_TOKENS="${POLLUTER_TARGET_PROMPT_TOKENS:-2000}"
SCENARIO="${SCENARIO:-forced-reset}"
SERVER_WAIT_SEC="${SERVER_WAIT_SEC:-300}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT_DIR}/results/external_restore_sanity}"

usage() {
  cat <<EOF
Usage: $0

Runs a deterministic t0 external-restore check.

SCENARIO=forced-reset:
  cold -> GPU-hit store flush -> local GPU cache reset -> restore

SCENARIO=natural-eviction:
  target A -> distinct polluter P -> target A

Environment overrides:
  PYTHON VLLM_BIN MODEL_PATH SERVED_MODEL_NAME GPU_ID HOST PORT
  GPU_MEMORY_UTILIZATION NUM_GPU_BLOCKS_OVERRIDE MAX_MODEL_LEN
  CPU_BYTES_TO_USE STORE_THRESHOLD MAX_TRACKER_SIZE TARGET_PROMPT_TOKENS
  POLLUTER_TARGET_PROMPT_TOKENS SCENARIO SERVER_WAIT_SEC RESULT_ROOT
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -gt 0 ]]; then
  usage >&2
  exit 2
fi

case "${SCENARIO}" in
  forced-reset)
    CLIENT_SCRIPT="benchmarks/external_restore_sanity.py"
    RUN_LABEL="t0_forced_reset"
    ;;
  natural-eviction)
    CLIENT_SCRIPT="benchmarks/external_restore_natural.py"
    RUN_LABEL="t0_natural_eviction"
    ;;
  *)
    echo "unknown SCENARIO: ${SCENARIO}" >&2
    exit 2
    ;;
esac

if [[ ! -x "${PYTHON}" ]]; then
  echo "python executable not found: ${PYTHON}" >&2
  exit 1
fi

if [[ ! -x "${VLLM_BIN}" ]]; then
  echo "vllm executable not found: ${VLLM_BIN}" >&2
  exit 1
fi

URL="http://${HOST}:${PORT}"
if curl --noproxy '*' -sSf --connect-timeout 1 --max-time 2 \
  "${URL}/v1/models" >/dev/null 2>&1; then
  echo "port ${PORT} already serves a vLLM endpoint; choose another PORT" >&2
  exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RESULT_ROOT}/${TIMESTAMP}/${RUN_LABEL}"
SERVER_LOG="${RUN_DIR}/server.log"
CLIENT_LOG="${RUN_DIR}/client.log"
REPORT_JSON="${RUN_DIR}/report.json"
mkdir -p "${RUN_DIR}"

KV_TRANSFER_CONFIG_JSON="$(
  "${PYTHON}" - \
    "${CPU_BYTES_TO_USE}" \
    "${STORE_THRESHOLD}" \
    "${MAX_TRACKER_SIZE}" <<'PY'
import json
import sys

cpu_bytes, store_threshold, max_tracker_size = sys.argv[1:]
print(
    json.dumps(
        {
            "kv_connector": "OffloadingConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "cpu_bytes_to_use": int(cpu_bytes),
                "eviction_policy": "lru",
                "store_threshold": int(store_threshold),
                "max_tracker_size": int(max_tracker_size),
            },
        }
    )
)
PY
)"

"${PYTHON}" - \
  "${RUN_DIR}/run_config.json" \
  "${MODEL_PATH}" \
  "${SERVED_MODEL_NAME}" \
  "${GPU_ID}" \
  "${URL}" \
  "${GPU_MEMORY_UTILIZATION}" \
  "${NUM_GPU_BLOCKS_OVERRIDE}" \
  "${MAX_MODEL_LEN}" \
  "${CPU_BYTES_TO_USE}" \
  "${STORE_THRESHOLD}" \
  "${MAX_TRACKER_SIZE}" \
  "${TARGET_PROMPT_TOKENS}" \
  "${POLLUTER_TARGET_PROMPT_TOKENS}" \
  "${SCENARIO}" <<'PY'
import json
import sys

(
    output,
    model_path,
    served_model_name,
    gpu_id,
    url,
    gpu_memory_utilization,
    num_gpu_blocks_override,
    max_model_len,
    cpu_bytes_to_use,
    store_threshold,
    max_tracker_size,
    target_prompt_tokens,
    polluter_target_prompt_tokens,
    scenario,
) = sys.argv[1:]
config = {
    "model_path": model_path,
    "served_model_name": served_model_name,
    "gpu_id": int(gpu_id),
    "url": url,
    "gpu_memory_utilization": float(gpu_memory_utilization),
    "num_gpu_blocks_override": int(num_gpu_blocks_override),
    "max_model_len": int(max_model_len),
    "cpu_bytes_to_use": int(cpu_bytes_to_use),
    "store_threshold": int(store_threshold),
    "max_tracker_size": int(max_tracker_size),
    "target_prompt_tokens": int(target_prompt_tokens),
    "polluter_target_prompt_tokens": int(polluter_target_prompt_tokens),
    "scenario": scenario,
}
with open(output, "w", encoding="utf-8") as file:
    json.dump(config, file, indent=2, sort_keys=True)
PY

server_pid=""
cleanup() {
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "starting t0 ${SCENARIO} server on ${URL}"
env \
  VLLM_SERVER_DEV_MODE=1 \
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  "${VLLM_BIN}" serve "${MODEL_PATH}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --trust-remote-code \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --num-gpu-blocks-override "${NUM_GPU_BLOCKS_OVERRIDE}" \
    --enable-prefix-caching \
    --kv-transfer-config "${KV_TRANSFER_CONFIG_JSON}" \
    >"${SERVER_LOG}" 2>&1 &
server_pid=$!

ready=0
for ((i = 0; i < SERVER_WAIT_SEC; ++i)); do
  if curl --noproxy '*' -sSf --connect-timeout 1 --max-time 2 \
    "${URL}/v1/models" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "server exited before readiness; see ${SERVER_LOG}" >&2
    exit 1
  fi
  sleep 1
done

if [[ "${ready}" -ne 1 ]]; then
  echo "server did not become ready; see ${SERVER_LOG}" >&2
  exit 1
fi

client_args=(
  --url "${URL}"
  --model "${SERVED_MODEL_NAME}"
  --output "${REPORT_JSON}"
  --target-prompt-tokens "${TARGET_PROMPT_TOKENS}"
)
if [[ "${SCENARIO}" == "natural-eviction" ]]; then
  client_args+=(
    --polluter-target-prompt-tokens "${POLLUTER_TARGET_PROMPT_TOKENS}"
  )
fi

set +e
"${PYTHON}" "${CLIENT_SCRIPT}" \
  "${client_args[@]}" \
  >"${CLIENT_LOG}" 2>&1
client_status=$?
set -e

cleanup
server_pid=""

sed -n '1,200p' "${CLIENT_LOG}"
echo "result directory: ${RUN_DIR}"
exit "${client_status}"
