#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"

export WORKSPACE=${WORKSPACE:-${HOME}/scratch/open-ttt-workspace}
export ERDOS_DIR=${ERDOS_DIR:-${REPO_DIR}}
export MODEL_ROOT=${MODEL_ROOT:-${WORKSPACE}/models}
export CKPT_ROOT=${CKPT_ROOT:-${WORKSPACE}/ckpt}
export MODEL_DIR=${MODEL_DIR:-${MODEL_ROOT}}
export LOG_DIR=${LOG_DIR:-${ERDOS_DIR}/logs}

RAW_MODEL=${MODEL_RAW:-${MODEL_ROOT}/gpt-oss-20b}
BF16_MODEL=${MODEL_BF16:-${MODEL_ROOT}/gpt-oss-20b-bf16}
TORCH_DIST_CKPT=${CKPT_OUT:-${CKPT_ROOT}/gpt-oss-20b_torch_dist}

echo "Repo: ${ERDOS_DIR}"
echo "Workspace: ${WORKSPACE}"
echo "Raw model: ${RAW_MODEL}"
echo "BF16 model: ${BF16_MODEL}"
echo "Megatron checkpoint: ${TORCH_DIST_CKPT}"

if [ "${SKIP_DOWNLOAD:-0}" != "1" ]; then
  if [ -f "${RAW_MODEL}/config.json" ] && find "${RAW_MODEL}" -maxdepth 1 -type f \( -name '*.safetensors' -o -name 'pytorch_model*.bin' \) -print -quit 2>/dev/null | grep -q .; then
    echo "Skipping download; raw model already exists."
  else
    MODEL_ROOT="${MODEL_ROOT}" MODEL_DIR="${RAW_MODEL}" "${SCRIPT_DIR}/download_gpt_oss_20b.sh"
  fi
fi

if [ "${SKIP_PREPARE:-0}" != "1" ]; then
  if [ -f "${BF16_MODEL}/config.json" ] && [ -f "${TORCH_DIST_CKPT}/latest_checkpointed_iteration.txt" ]; then
    echo "Skipping prepare; BF16 model and torch_dist checkpoint already exist."
  else
    MODEL_ROOT="${MODEL_ROOT}" MODEL_RAW="${RAW_MODEL}" MODEL_BF16="${BF16_MODEL}" CKPT_ROOT="${CKPT_ROOT}" CKPT_OUT="${TORCH_DIST_CKPT}" \
      "${SCRIPT_DIR}/prepare_gpt_oss_20b_slime.sh"
  fi
fi

if [ "${SKIP_TRAIN:-0}" = "1" ]; then
  echo "SKIP_TRAIN=1 set; stopping after preparation."
  exit 0
fi

MODEL_DIR="${MODEL_ROOT}" MODEL_BF16="${BF16_MODEL}" CKPT_ROOT="${CKPT_ROOT}" REF_CKPT="${TORCH_DIST_CKPT}" \
  "${REPO_DIR}/run_gpt_oss_20b_erdos.sh"
