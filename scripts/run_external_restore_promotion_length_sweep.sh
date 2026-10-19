#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="${RUNNER:-${ROOT_DIR}/scripts/run_external_restore_admission.sh}"
PYTHON="${PYTHON:-${ROOT_DIR}/.venv/bin/python}"
GPU_ID="${GPU_ID:-1}"
PORT_BASE="${PORT_BASE:-8200}"
REPETITIONS="${REPETITIONS:-5}"
MINIMAL_CPU_BYTES="${MINIMAL_CPU_BYTES:-167772160}"
PROMOTION_TARGETS="${PROMOTION_TARGETS:-64 128 256 512 1024}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT_DIR}/results/external_restore_promotion_length}"
SWEEP_ID="${SWEEP_ID:-$(date +%Y%m%d_%H%M%S)}"

usage() {
  cat <<EOF
Usage: $0

Runs a counterbalanced forced RESTORE/RECOMPUTE sweep over target prefix
lengths. Each arm starts a fresh vLLM server.

Environment overrides:
  RUNNER PYTHON GPU_ID PORT_BASE REPETITIONS MINIMAL_CPU_BYTES
  PROMOTION_TARGETS RESULT_ROOT SWEEP_ID

Example:
  PROMOTION_TARGETS="64 128" REPETITIONS=1 $0
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

if ! [[ "${REPETITIONS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "REPETITIONS must be a positive integer, got: ${REPETITIONS}" >&2
  exit 2
fi

if ! [[ "${MINIMAL_CPU_BYTES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MINIMAL_CPU_BYTES must be positive, got: ${MINIMAL_CPU_BYTES}" >&2
  exit 2
fi

read -r -a targets <<<"${PROMOTION_TARGETS}"
if [[ ${#targets[@]} -eq 0 ]]; then
  echo "PROMOTION_TARGETS must contain at least one token length" >&2
  exit 2
fi
for target in "${targets[@]}"; do
  if ! [[ "${target}" =~ ^[1-9][0-9]*$ ]]; then
    echo "invalid target token length: ${target}" >&2
    exit 2
  fi
done

SWEEP_ROOT="${RESULT_ROOT}/${SWEEP_ID}"
mkdir -p "${SWEEP_ROOT}"

overall_status=0
port_base="${PORT_BASE}"
for target in "${targets[@]}"; do
  target_root="${SWEEP_ROOT}/tokens_${target}"
  echo "==> target ${target} tokens; result root: ${target_root}"
  if ! env \
    GPU_ID="${GPU_ID}" \
    PORT_BASE="${port_base}" \
    REPETITIONS="${REPETITIONS}" \
    MINIMAL_CPU_BYTES="${MINIMAL_CPU_BYTES}" \
    TARGET_PROMPT_TOKENS="${target}" \
    RESULT_ROOT="${target_root}" \
    RUN_ID="run" \
    ARM="promotion-pair" \
    "${RUNNER}"; then
    overall_status=1
  fi
  port_base=$((port_base + 2 * REPETITIONS))
done

"${PYTHON}" \
  "${ROOT_DIR}/scripts/summarize_external_restore_promotion_length.py" \
  "${SWEEP_ROOT}"

echo "promotion length sweep result: ${SWEEP_ROOT}"
exit "${overall_status}"
