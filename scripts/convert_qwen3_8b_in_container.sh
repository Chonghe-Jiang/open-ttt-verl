#!/bin/bash
set -Eeuo pipefail

REPO_DIR=${REPO_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)}
WORKSPACE=${WORKSPACE:-/root/workspace}
LOG_ROOT=${LOG_ROOT:-${WORKSPACE}/logs}
MODEL_RAW=${MODEL_RAW:-${WORKSPACE}/models/Qwen3-8B}
CKPT_OUT=${CKPT_OUT:-${WORKSPACE}/ckpt/Qwen3-8B_torch_dist}

NPROC=${NPROC:-2}
TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-1}
PIPELINE_MODEL_PARALLEL_SIZE=${PIPELINE_MODEL_PARALLEL_SIZE:-2}
EXPERT_MODEL_PARALLEL_SIZE=${EXPERT_MODEL_PARALLEL_SIZE:-1}
EXPERT_TENSOR_PARALLEL_SIZE=${EXPERT_TENSOR_PARALLEL_SIZE:-1}
FORCE_CONVERT=${FORCE_CONVERT:-0}

mkdir -p "${LOG_ROOT}" "$(dirname "${CKPT_OUT}")"

LOG_FILE=${LOG_FILE:-${LOG_ROOT}/convert_qwen3_8b_in_container_$(date +%Y%m%d_%H%M%S).log}
touch "${LOG_FILE}"
rm -f "${LOG_ROOT}/convert_qwen3_8b_in_container.latest.log"
ln -s "${LOG_FILE}" "${LOG_ROOT}/convert_qwen3_8b_in_container.latest.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
trap 'status=$?; echo "Exit status: ${status}"; echo "Ended at: $(date -Is)"' EXIT

WORLD_SIZE=$((TENSOR_MODEL_PARALLEL_SIZE * PIPELINE_MODEL_PARALLEL_SIZE * EXPERT_MODEL_PARALLEL_SIZE * EXPERT_TENSOR_PARALLEL_SIZE))
if [ "${WORLD_SIZE}" -ne "${NPROC}" ]; then
  echo "Invalid parallel config: NPROC=${NPROC}, but TP*PP*EP*ETP=${WORLD_SIZE}" >&2
  exit 1
fi

echo "Logging to ${LOG_FILE}"
echo "Repo: ${REPO_DIR}"
echo "Workspace: ${WORKSPACE}"
echo "Raw model: ${MODEL_RAW}"
echo "Megatron checkpoint: ${CKPT_OUT}"
echo "NPROC: ${NPROC}"
echo "TP/PP/EP/ETP: ${TENSOR_MODEL_PARALLEL_SIZE}/${PIPELINE_MODEL_PARALLEL_SIZE}/${EXPERT_MODEL_PARALLEL_SIZE}/${EXPERT_TENSOR_PARALLEL_SIZE}"
echo "Host: $(hostname)"

test -d "${REPO_DIR}/slime"
test -f "${MODEL_RAW}/config.json"
if ! find "${MODEL_RAW}" -maxdepth 1 -type f \( -name '*.safetensors' -o -name 'pytorch_model*.bin' \) -print -quit 2>/dev/null | grep -q .; then
  echo "Raw model weights are missing under ${MODEL_RAW}" >&2
  exit 1
fi

if [ -f "${CKPT_OUT}/latest_checkpointed_iteration.txt" ] && [ "${FORCE_CONVERT}" != "1" ]; then
  echo "Megatron torch_dist checkpoint already exists at ${CKPT_OUT}; set FORCE_CONVERT=1 to rebuild."
  exit 0
fi

cd "${REPO_DIR}/slime"
export PYTHONPATH="${REPO_DIR}/Megatron-LM:${REPO_DIR}:${REPO_DIR}/slime:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NPROC_REQUIRED="${NPROC}"

python3 - <<'PY'
import os
import torch

required = int(os.environ.get("NPROC_REQUIRED", os.environ.get("NPROC", "2")))
count = torch.cuda.device_count()
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", count)
if not torch.cuda.is_available() or count < required:
    raise SystemExit(f"Need at least {required} visible CUDA devices, got {count}.")
PY

source scripts/models/qwen3-8B.sh
torchrun --nproc_per_node "${NPROC}" tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${MODEL_RAW}" \
  --save "${CKPT_OUT}" \
  --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}" \
  --pipeline-model-parallel-size "${PIPELINE_MODEL_PARALLEL_SIZE}" \
  --expert-model-parallel-size "${EXPERT_MODEL_PARALLEL_SIZE}" \
  --expert-tensor-parallel-size "${EXPERT_TENSOR_PARALLEL_SIZE}"

test -f "${CKPT_OUT}/latest_checkpointed_iteration.txt"
echo "Prepared Qwen3-8B torch_dist checkpoint at ${CKPT_OUT}"
