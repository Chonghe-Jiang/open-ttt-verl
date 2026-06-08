#!/bin/bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/home/qua/code/open-ttt}
WORKSPACE=${WORKSPACE:-${HOME}/scratch/open-ttt-workspace}
LOG_ROOT=${LOG_ROOT:-${WORKSPACE}/logs}
IMAGE_ROOT=${IMAGE_ROOT:-${WORKSPACE}/images}
SANDBOX_PATH=${SANDBOX_PATH:-${IMAGE_ROOT}/slime_latest.sandbox}
SIF_PATH=${SIF_PATH:-${IMAGE_ROOT}/slime_latest.sif}
CONTAINER_PATH=${CONTAINER_PATH:-}
MODEL_RAW=${MODEL_RAW:-${WORKSPACE}/models/Qwen3-8B}
CKPT_OUT=${CKPT_OUT:-${WORKSPACE}/ckpt/Qwen3-8B_torch_dist}
NPROC=${NPROC:-2}
TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-1}
PIPELINE_MODEL_PARALLEL_SIZE=${PIPELINE_MODEL_PARALLEL_SIZE:-2}
EXPERT_MODEL_PARALLEL_SIZE=${EXPERT_MODEL_PARALLEL_SIZE:-1}
EXPERT_TENSOR_PARALLEL_SIZE=${EXPERT_TENSOR_PARALLEL_SIZE:-1}
FORCE_CONVERT=${FORCE_CONVERT:-0}

mkdir -p "${LOG_ROOT}" "$(dirname "${CKPT_OUT}")"

LOG_FILE=${LOG_FILE:-${LOG_ROOT}/convert_qwen3_8b_with_apptainer_$(date +%Y%m%d_%H%M%S).log}
touch "${LOG_FILE}"
rm -f "${LOG_ROOT}/convert_qwen3_8b_with_apptainer.latest.log"
ln -s "${LOG_FILE}" "${LOG_ROOT}/convert_qwen3_8b_with_apptainer.latest.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
trap 'status=$?; echo "Exit status: ${status}"; echo "Ended at: $(date -Is)"' EXIT

if [ -z "${CONTAINER_PATH}" ]; then
  if [ -d "${SANDBOX_PATH}" ]; then
    CONTAINER_PATH=${SANDBOX_PATH}
  else
    CONTAINER_PATH=${SIF_PATH}
  fi
fi

WORLD_SIZE=$((TENSOR_MODEL_PARALLEL_SIZE * PIPELINE_MODEL_PARALLEL_SIZE * EXPERT_MODEL_PARALLEL_SIZE * EXPERT_TENSOR_PARALLEL_SIZE))
if [ "${WORLD_SIZE}" -ne "${NPROC}" ]; then
  echo "Invalid parallel config: NPROC=${NPROC}, but TP*PP*EP*ETP=${WORLD_SIZE}" >&2
  exit 1
fi

echo "Logging to ${LOG_FILE}"
echo "Repo: ${REPO_DIR}"
echo "Workspace: ${WORKSPACE}"
echo "Container path: ${CONTAINER_PATH}"
echo "Raw model: ${MODEL_RAW}"
echo "Megatron checkpoint: ${CKPT_OUT}"
echo "NPROC: ${NPROC}"
echo "Tensor parallel: ${TENSOR_MODEL_PARALLEL_SIZE}"
echo "Pipeline parallel: ${PIPELINE_MODEL_PARALLEL_SIZE}"
echo "Expert parallel: ${EXPERT_MODEL_PARALLEL_SIZE}"
echo "Expert tensor parallel: ${EXPERT_TENSOR_PARALLEL_SIZE}"
echo "Force convert: ${FORCE_CONVERT}"
echo "Host: $(hostname)"

test -d "${REPO_DIR}"
test -e "${CONTAINER_PATH}" || {
  echo "Missing container: ${CONTAINER_PATH}" >&2
  echo "Run scripts/build_slime_sandbox.sh first, or provide CONTAINER_PATH." >&2
  exit 1
}
test -f "${MODEL_RAW}/config.json"
if ! find "${MODEL_RAW}" -maxdepth 1 -type f \( -name '*.safetensors' -o -name 'pytorch_model*.bin' \) -print -quit 2>/dev/null | grep -q .; then
  echo "Raw model weights are missing under ${MODEL_RAW}" >&2
  exit 1
fi

if [ -f "${CKPT_OUT}/latest_checkpointed_iteration.txt" ] && [ "${FORCE_CONVERT}" != "1" ]; then
  echo "Megatron torch_dist checkpoint already exists at ${CKPT_OUT}; set FORCE_CONVERT=1 to rebuild."
  exit 0
fi

if [ -f /etc/profile.d/modules.sh ]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh || true
fi
if command -v module >/dev/null 2>&1; then
  module load apptainer >/dev/null 2>&1 || true
  module load singularity >/dev/null 2>&1 || true
fi

if command -v apptainer >/dev/null 2>&1; then
  RUNTIME=apptainer
elif command -v singularity >/dev/null 2>&1; then
  RUNTIME=singularity
elif [ -x /usr/bin/apptainer ]; then
  RUNTIME=/usr/bin/apptainer
elif [ -x /usr/bin/singularity ]; then
  RUNTIME=/usr/bin/singularity
else
  echo "Neither apptainer nor singularity was found." >&2
  exit 1
fi

echo "Started at: $(date -Is)"
echo "Runtime: ${RUNTIME}"
"${RUNTIME}" exec --nv --cleanenv \
  --bind /home/qua:/home/qua \
  "${CONTAINER_PATH}" \
  bash -c "
    set -euo pipefail
    cd '${REPO_DIR}/slime'
    export PYTHONPATH='${REPO_DIR}/Megatron-LM:${REPO_DIR}/slime:${REPO_DIR}:'\"\${PYTHONPATH:-}\"
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    echo \"CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-unset}\"
    ls -l /dev/nvidia* 2>/dev/null || true
    python3 - <<'PY'
import os
import torch
required = int(os.environ.get('NPROC_REQUIRED', '${NPROC}'))
count = torch.cuda.device_count()
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'gpus', count)
if not torch.cuda.is_available() or count < required:
    raise SystemExit(f'Need at least {required} visible CUDA devices, got {count}. Re-submit with {required} GPUs.')
PY
    source scripts/models/qwen3-8B.sh
    torchrun --nproc_per_node '${NPROC}' tools/convert_hf_to_torch_dist.py \
      \"\${MODEL_ARGS[@]}\" \
      --hf-checkpoint '${MODEL_RAW}' \
      --save '${CKPT_OUT}' \
      --tensor-model-parallel-size '${TENSOR_MODEL_PARALLEL_SIZE}' \
      --pipeline-model-parallel-size '${PIPELINE_MODEL_PARALLEL_SIZE}' \
      --expert-model-parallel-size '${EXPERT_MODEL_PARALLEL_SIZE}' \
      --expert-tensor-parallel-size '${EXPERT_TENSOR_PARALLEL_SIZE}'
  "

test -f "${CKPT_OUT}/latest_checkpointed_iteration.txt"
echo "Prepared Qwen3-8B torch_dist checkpoint at ${CKPT_OUT}"
