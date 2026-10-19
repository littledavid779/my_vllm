#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-${ROOT_DIR}/.venv/bin/python}"
VLLM_BIN="${VLLM_BIN:-${ROOT_DIR}/.venv/bin/vllm}"
MODEL_PATH="${MODEL_PATH:-/public/model_zoo/Qwen3-0.6B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-0.6b}"
GPU_ID="${GPU_ID:-0}"
HOST="${HOST:-127.0.0.1}"
PORT_BASE="${PORT_BASE:-8140}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.05}"
NUM_GPU_BLOCKS_OVERRIDE="${NUM_GPU_BLOCKS_OVERRIDE:-116}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MINIMAL_CPU_BYTES="${MINIMAL_CPU_BYTES:-67108864}"
CONSTRAINED_CPU_BYTES="${CONSTRAINED_CPU_BYTES:-207355904}"
LARGE_CPU_BYTES="${LARGE_CPU_BYTES:-291504128}"
MAX_TRACKER_SIZE="${MAX_TRACKER_SIZE:-64000}"
TARGET_PROMPT_TOKENS="${TARGET_PROMPT_TOKENS:-512}"
POLLUTER_TARGET_PROMPT_TOKENS="${POLLUTER_TARGET_PROMPT_TOKENS:-1800}"
SERVER_WAIT_SEC="${SERVER_WAIT_SEC:-300}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT_DIR}/results/external_restore_admission}"
ARM="${ARM:-suite}"
REPETITIONS="${REPETITIONS:-1}"
RUN_ID="${RUN_ID:-}"

usage() {
  cat <<EOF
Usage: $0

Runs a controlled A -> A -> A -> P -> Q -> A admission comparison. Every arm
starts a fresh vLLM server.

ARM choices:
  suite                    Run all five arms (default)
  no-offload               No CPU KV connector
  all-block-constrained    All-block admission with a 198 MiB CPU tier
  all-block-large          All-block admission with a 278 MiB CPU tier
  selective-constrained   Threshold-2 admission with a 198 MiB CPU tier
  selective-minimal       Threshold-2 admission with a 64 MiB CPU tier
  promotion-pair          Run selective-minimal with forced restore/recompute
  selective-minimal-restore
  selective-minimal-recompute

Environment overrides:
  PYTHON VLLM_BIN MODEL_PATH SERVED_MODEL_NAME GPU_ID HOST PORT_BASE
  GPU_MEMORY_UTILIZATION NUM_GPU_BLOCKS_OVERRIDE MAX_MODEL_LEN
  MINIMAL_CPU_BYTES CONSTRAINED_CPU_BYTES LARGE_CPU_BYTES MAX_TRACKER_SIZE
  TARGET_PROMPT_TOKENS POLLUTER_TARGET_PROMPT_TOKENS SERVER_WAIT_SEC
  RESULT_ROOT ARM REPETITIONS RUN_ID
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

case "${ARM}" in
  suite)
    ARMS=(
      no-offload
      all-block-constrained
      all-block-large
      selective-constrained
      selective-minimal
    )
    ;;
  promotion-pair)
    ARMS=(
      selective-minimal-restore
      selective-minimal-recompute
    )
    ;;
  no-offload | all-block-constrained | all-block-large \
    | selective-constrained | selective-minimal \
    | selective-minimal-restore | selective-minimal-recompute)
    ARMS=("${ARM}")
    ;;
  *)
    echo "unknown ARM: ${ARM}" >&2
    usage >&2
    exit 2
    ;;
esac

if ! [[ "${REPETITIONS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "REPETITIONS must be a positive integer, got: ${REPETITIONS}" >&2
  exit 2
fi

if [[ ! -x "${PYTHON}" ]]; then
  echo "python executable not found: ${PYTHON}" >&2
  exit 1
fi

if [[ ! -x "${VLLM_BIN}" ]]; then
  echo "vllm executable not found: ${VLLM_BIN}" >&2
  exit 1
fi

if [[ -z "${RUN_ID}" ]]; then
  RUN_ID="$(date +%Y%m%d_%H%M%S)"
fi
RUN_ROOT="${RESULT_ROOT}/${RUN_ID}"
mkdir -p "${RUN_ROOT}"

current_server_pid=""
cleanup() {
  if [[ -n "${current_server_pid}" ]] \
    && kill -0 "${current_server_pid}" 2>/dev/null; then
    kill "${current_server_pid}" 2>/dev/null || true
    wait "${current_server_pid}" 2>/dev/null || true
  fi
  current_server_pid=""
}
trap cleanup EXIT

run_one() {
  local arm="$1"
  local port_offset="$2"
  local repetition="$3"
  local mode
  local expected_final_path
  local enable_offload
  local cpu_bytes
  local store_threshold
  local count_store_attempts
  local promotion_policy

  case "${arm}" in
    no-offload)
      mode="no-offload"
      expected_final_path="recompute"
      enable_offload="false"
      cpu_bytes=0
      store_threshold=0
      count_store_attempts="false"
      promotion_policy="none"
      ;;
    all-block-constrained)
      mode="all-block"
      expected_final_path="recompute"
      enable_offload="true"
      cpu_bytes="${CONSTRAINED_CPU_BYTES}"
      store_threshold=0
      count_store_attempts="false"
      promotion_policy="always_restore"
      ;;
    all-block-large)
      mode="all-block"
      expected_final_path="restore"
      enable_offload="true"
      cpu_bytes="${LARGE_CPU_BYTES}"
      store_threshold=0
      count_store_attempts="false"
      promotion_policy="always_restore"
      ;;
    selective-constrained)
      mode="selective"
      expected_final_path="restore"
      enable_offload="true"
      cpu_bytes="${CONSTRAINED_CPU_BYTES}"
      store_threshold=2
      count_store_attempts="true"
      promotion_policy="always_restore"
      ;;
    selective-minimal)
      mode="selective"
      expected_final_path="restore"
      enable_offload="true"
      cpu_bytes="${MINIMAL_CPU_BYTES}"
      store_threshold=2
      count_store_attempts="true"
      promotion_policy="always_restore"
      ;;
    selective-minimal-restore)
      mode="selective"
      expected_final_path="restore"
      enable_offload="true"
      cpu_bytes="${MINIMAL_CPU_BYTES}"
      store_threshold=2
      count_store_attempts="true"
      promotion_policy="always_restore"
      ;;
    selective-minimal-recompute)
      mode="selective"
      expected_final_path="recompute"
      enable_offload="true"
      cpu_bytes="${MINIMAL_CPU_BYTES}"
      store_threshold=2
      count_store_attempts="true"
      promotion_policy="always_recompute"
      ;;
    *)
      echo "unknown arm: ${arm}" >&2
      return 2
      ;;
  esac

  local port=$((PORT_BASE + port_offset))
  local url="http://${HOST}:${port}"
  local run_dir="${RUN_ROOT}/rep_${repetition}/${arm}"
  local server_log="${run_dir}/server.log"
  local client_log="${run_dir}/client.log"
  local report_json="${run_dir}/report.json"
  mkdir -p "${run_dir}"

  if curl --noproxy '*' -sSf --connect-timeout 1 --max-time 2 \
    "${url}/v1/models" >/dev/null 2>&1; then
    echo "port ${port} already serves a vLLM endpoint" >&2
    return 1
  fi

  local kv_transfer_config_json=""
  if [[ "${enable_offload}" == "true" ]]; then
    kv_transfer_config_json="$(
      "${PYTHON}" - \
        "${cpu_bytes}" \
        "${store_threshold}" \
        "${count_store_attempts}" \
        "${MAX_TRACKER_SIZE}" \
        "${promotion_policy}" <<'PY'
import json
import sys

cpu_bytes, threshold, count_attempts, max_tracker_size, promotion_policy = (
    sys.argv[1:]
)
print(
    json.dumps(
        {
            "kv_connector": "OffloadingConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "cpu_bytes_to_use": int(cpu_bytes),
                "eviction_policy": "lru",
                "store_threshold": int(threshold),
                "count_store_attempts": count_attempts.lower() == "true",
                "max_tracker_size": int(max_tracker_size),
                "promotion_policy": promotion_policy,
            },
        }
    )
)
PY
    )"
  fi

  "${PYTHON}" - \
    "${run_dir}/run_config.json" \
    "${arm}" \
    "${mode}" \
    "${expected_final_path}" \
    "${MODEL_PATH}" \
    "${SERVED_MODEL_NAME}" \
    "${GPU_ID}" \
    "${url}" \
    "${GPU_MEMORY_UTILIZATION}" \
    "${NUM_GPU_BLOCKS_OVERRIDE}" \
    "${MAX_MODEL_LEN}" \
    "${enable_offload}" \
    "${cpu_bytes}" \
    "${store_threshold}" \
    "${count_store_attempts}" \
    "${MAX_TRACKER_SIZE}" \
    "${promotion_policy}" \
    "${TARGET_PROMPT_TOKENS}" \
    "${POLLUTER_TARGET_PROMPT_TOKENS}" \
    "${kv_transfer_config_json}" <<'PY'
import json
import sys

(
    output,
    arm,
    mode,
    expected_final_path,
    model_path,
    served_model_name,
    gpu_id,
    url,
    gpu_memory_utilization,
    num_gpu_blocks_override,
    max_model_len,
    enable_offload,
    cpu_bytes,
    store_threshold,
    count_store_attempts,
    max_tracker_size,
    promotion_policy,
    target_prompt_tokens,
    polluter_target_prompt_tokens,
    kv_transfer_config_json,
) = sys.argv[1:]
config = {
    "arm": arm,
    "mode": mode,
    "expected_final_path": expected_final_path,
    "model_path": model_path,
    "served_model_name": served_model_name,
    "gpu_id": int(gpu_id),
    "url": url,
    "gpu_memory_utilization": float(gpu_memory_utilization),
    "num_gpu_blocks_override": int(num_gpu_blocks_override),
    "max_model_len": int(max_model_len),
    "enable_offload": enable_offload == "true",
    "cpu_bytes_to_use": int(cpu_bytes),
    "store_threshold": int(store_threshold),
    "count_store_attempts": count_store_attempts == "true",
    "max_tracker_size": int(max_tracker_size),
    "promotion_policy": promotion_policy,
    "target_prompt_tokens": int(target_prompt_tokens),
    "polluter_target_prompt_tokens": int(polluter_target_prompt_tokens),
    "kv_transfer_config": (
        json.loads(kv_transfer_config_json) if kv_transfer_config_json else None
    ),
}
with open(output, "w", encoding="utf-8") as file:
    json.dump(config, file, indent=2, sort_keys=True)
PY

  local kv_transfer_args=()
  if [[ -n "${kv_transfer_config_json}" ]]; then
    kv_transfer_args=(--kv-transfer-config "${kv_transfer_config_json}")
  fi

  echo "==> ${arm}: starting a fresh server on ${url}"
  env \
    VLLM_SERVER_DEV_MODE=1 \
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    "${VLLM_BIN}" serve "${MODEL_PATH}" \
      --host "${HOST}" \
      --port "${port}" \
      --served-model-name "${SERVED_MODEL_NAME}" \
      --trust-remote-code \
      --max-model-len "${MAX_MODEL_LEN}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      --num-gpu-blocks-override "${NUM_GPU_BLOCKS_OVERRIDE}" \
      --enable-prefix-caching \
      "${kv_transfer_args[@]}" \
      >"${server_log}" 2>&1 &
  current_server_pid=$!

  local ready=0
  for ((i = 0; i < SERVER_WAIT_SEC; ++i)); do
    if curl --noproxy '*' -sSf --connect-timeout 1 --max-time 2 \
      "${url}/v1/models" >/dev/null 2>&1; then
      ready=1
      break
    fi
    if ! kill -0 "${current_server_pid}" 2>/dev/null; then
      echo "${arm}: server exited before readiness; see ${server_log}" >&2
      cleanup
      return 1
    fi
    sleep 1
  done

  if [[ "${ready}" -ne 1 ]]; then
    echo "${arm}: server did not become ready; see ${server_log}" >&2
    cleanup
    return 1
  fi

  set +e
  "${PYTHON}" "${ROOT_DIR}/benchmarks/external_restore_admission.py" \
    --url "${url}" \
    --model "${SERVED_MODEL_NAME}" \
    --output "${report_json}" \
    --mode "${mode}" \
    --expected-final-path "${expected_final_path}" \
    --target-prompt-tokens "${TARGET_PROMPT_TOKENS}" \
    --polluter-target-prompt-tokens "${POLLUTER_TARGET_PROMPT_TOKENS}" \
    >"${client_log}" 2>&1
  local client_status=$?
  set -e

  cleanup
  sed -n '1,200p' "${client_log}"
  echo "${arm} result: ${run_dir}"
  return "${client_status}"
}

overall_status=0
offset=0
for ((repetition = 1; repetition <= REPETITIONS; ++repetition)); do
  repetition_arms=("${ARMS[@]}")
  if [[ "${ARM}" == "promotion-pair" ]] && ((repetition % 2 == 0)); then
    repetition_arms=(
      selective-minimal-recompute
      selective-minimal-restore
    )
  fi
  for arm in "${repetition_arms[@]}"; do
    if ! run_one "${arm}" "${offset}" "${repetition}"; then
      overall_status=1
    fi
    offset=$((offset + 1))
  done
done

"${PYTHON}" \
  "${ROOT_DIR}/scripts/summarize_external_restore_admission.py" \
  "${RUN_ROOT}"

echo "suite result: ${RUN_ROOT}"
exit "${overall_status}"
