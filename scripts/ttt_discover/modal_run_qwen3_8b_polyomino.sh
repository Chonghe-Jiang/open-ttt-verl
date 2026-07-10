#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ACTION="${1:-run}"
shift || true

if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi

if [[ -z "${MODAL_TOKEN_SECRET:-}" && -n "${MODAL_SECRET_KEY:-}" ]]; then
  export MODAL_TOKEN_SECRET="${MODAL_SECRET_KEY}"
fi

export MODAL_GPU="${MODAL_GPU:-H200:2}"
export MODAL_GPUS="${MODAL_GPUS:-0,1}"
export MODAL_REPO_ROOT="${MODAL_REPO_ROOT:-${ROOT}}"

exec modal run --detach --timestamps --name polynomino-qwen3-8b-h200-g4n16-polyomino-train \
  "${ROOT}/scripts/ttt_discover/modal_qwen3_8b_polyomino.py" \
  --action "${ACTION}" \
  "$@"
