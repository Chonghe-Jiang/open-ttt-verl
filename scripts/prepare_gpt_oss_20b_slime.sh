#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
WORKSPACE=${WORKSPACE:-${HOME}/scratch/open-ttt-workspace}
ERDOS_DIR=${ERDOS_DIR:-${REPO_DIR}}
MODEL_ROOT=${MODEL_ROOT:-${WORKSPACE}/models}
CKPT_ROOT=${CKPT_ROOT:-${WORKSPACE}/ckpt}
LOG_ROOT=${LOG_ROOT:-${WORKSPACE}/logs}
MODEL_RAW=${MODEL_RAW:-${MODEL_ROOT}/gpt-oss-20b}
MODEL_BF16=${MODEL_BF16:-${MODEL_ROOT}/gpt-oss-20b-bf16}
CKPT_OUT=${CKPT_OUT:-${CKPT_ROOT}/gpt-oss-20b_torch_dist}
NPROC=${NPROC:-2}
TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-1}
PIPELINE_MODEL_PARALLEL_SIZE=${PIPELINE_MODEL_PARALLEL_SIZE:-2}
EXPERT_MODEL_PARALLEL_SIZE=${EXPERT_MODEL_PARALLEL_SIZE:-1}
EXPERT_TENSOR_PARALLEL_SIZE=${EXPERT_TENSOR_PARALLEL_SIZE:-1}
FORCE_PREPROCESS=${FORCE_PREPROCESS:-0}
FORCE_CONVERT=${FORCE_CONVERT:-0}

mkdir -p "${MODEL_ROOT}" "${CKPT_ROOT}" "${LOG_ROOT}"
LOG_FILE=${LOG_FILE:-${LOG_ROOT}/prepare_gpt_oss_20b_slime_$(date +%Y%m%d_%H%M%S).log}
touch "${LOG_FILE}"
rm -f "${LOG_ROOT}/prepare_gpt_oss_20b_slime.latest.log"
ln -s "${LOG_FILE}" "${LOG_ROOT}/prepare_gpt_oss_20b_slime.latest.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
trap 'status=$?; echo "Exit status: ${status}"; echo "Ended at: $(date -Is)"' EXIT

echo "Logging to ${LOG_FILE}"
echo "Repo dir: ${ERDOS_DIR}"
echo "Raw model: ${MODEL_RAW}"
echo "BF16 model: ${MODEL_BF16}"
echo "Megatron checkpoint: ${CKPT_OUT}"
echo "NPROC: ${NPROC}"
echo "Tensor parallel: ${TENSOR_MODEL_PARALLEL_SIZE}"
echo "Pipeline parallel: ${PIPELINE_MODEL_PARALLEL_SIZE}"
echo "Expert parallel: ${EXPERT_MODEL_PARALLEL_SIZE}"
echo "Expert tensor parallel: ${EXPERT_TENSOR_PARALLEL_SIZE}"

WORLD_SIZE=$((TENSOR_MODEL_PARALLEL_SIZE * PIPELINE_MODEL_PARALLEL_SIZE * EXPERT_MODEL_PARALLEL_SIZE * EXPERT_TENSOR_PARALLEL_SIZE))
if [ "${WORLD_SIZE}" -ne "${NPROC}" ]; then
  echo "Invalid parallel config: NPROC=${NPROC}, but TP*PP*EP*ETP=${WORLD_SIZE}" >&2
  exit 1
fi

test -f "${MODEL_RAW}/config.json"
if ! find "${MODEL_RAW}" -maxdepth 1 -type f \( -name '*.safetensors' -o -name 'pytorch_model*.bin' \) -print -quit 2>/dev/null | grep -q .; then
  echo "Raw model weights are missing under ${MODEL_RAW}; run scripts/job_download_gpt_oss_20b.sh first." >&2
  exit 1
fi
if ! python3 -c "import torch" >/dev/null 2>&1; then
  echo "python3 cannot import torch. Run this conversion inside the Slime Docker/training environment, or load a Python environment with torch installed." >&2
  exit 1
fi
command -v torchrun >/dev/null 2>&1 || {
  echo "torchrun not found. Run this conversion inside the Slime Docker/training environment." >&2
  exit 1
}

cd "${ERDOS_DIR}/slime"
source scripts/models/gpt-oss-20B.sh

if [ "${FORCE_PREPROCESS}" = "1" ] || [ ! -f "${MODEL_BF16}/config.json" ]; then
  python3 tools/preprocess_gpt_oss.py \
    --input "${MODEL_RAW}" \
    --output "${MODEL_BF16}"
else
  echo "BF16 model already exists at ${MODEL_BF16}; set FORCE_PREPROCESS=1 to rebuild."
fi

test -f "${MODEL_BF16}/config.json"
test -f "${MODEL_BF16}/tokenizer_config.json" -o -f "${MODEL_BF16}/tokenizer.json"

export PYTHONPATH="${ERDOS_DIR}/Megatron-LM:${ERDOS_DIR}/slime:${ERDOS_DIR}:${PYTHONPATH:-}"

if [ "${FORCE_CONVERT}" = "1" ] || [ ! -f "${CKPT_OUT}/latest_checkpointed_iteration.txt" ]; then
  torchrun --nproc_per_node "${NPROC}" tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${MODEL_BF16}" \
    --save "${CKPT_OUT}" \
    --megatron-to-hf-mode bridge \
    --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}" \
    --pipeline-model-parallel-size "${PIPELINE_MODEL_PARALLEL_SIZE}" \
    --expert-model-parallel-size "${EXPERT_MODEL_PARALLEL_SIZE}" \
    --expert-tensor-parallel-size "${EXPERT_TENSOR_PARALLEL_SIZE}"
else
  echo "Megatron torch_dist checkpoint already exists at ${CKPT_OUT}; set FORCE_CONVERT=1 to rebuild."
fi

test -f "${CKPT_OUT}/latest_checkpointed_iteration.txt"
echo "Prepared Slime checkpoint at ${CKPT_OUT}"
