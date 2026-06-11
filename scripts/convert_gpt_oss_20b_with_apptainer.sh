#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR=${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}
WORKSPACE=${WORKSPACE:-${HOME}/scratch/open-ttt-workspace}
LOG_ROOT=${LOG_ROOT:-${WORKSPACE}/logs}
IMAGE_ROOT=${IMAGE_ROOT:-${WORKSPACE}/images}
SANDBOX_PATH=${SANDBOX_PATH:-${IMAGE_ROOT}/slime_latest.sandbox}
SIF_PATH=${SIF_PATH:-${IMAGE_ROOT}/slime_latest.sif}
CONTAINER_PATH=${CONTAINER_PATH:-}
NPROC=${NPROC:-2}
TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-1}
PIPELINE_MODEL_PARALLEL_SIZE=${PIPELINE_MODEL_PARALLEL_SIZE:-2}
EXPERT_MODEL_PARALLEL_SIZE=${EXPERT_MODEL_PARALLEL_SIZE:-1}
EXPERT_TENSOR_PARALLEL_SIZE=${EXPERT_TENSOR_PARALLEL_SIZE:-1}
FORCE_PREPROCESS=${FORCE_PREPROCESS:-0}
FORCE_CONVERT=${FORCE_CONVERT:-0}

mkdir -p "${LOG_ROOT}"

LOG_FILE=${LOG_FILE:-${LOG_ROOT}/convert_gpt_oss_20b_with_apptainer_$(date +%Y%m%d_%H%M%S).log}
touch "${LOG_FILE}"
rm -f "${LOG_ROOT}/convert_gpt_oss_20b_with_apptainer.latest.log"
ln -s "${LOG_FILE}" "${LOG_ROOT}/convert_gpt_oss_20b_with_apptainer.latest.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
trap 'status=$?; echo "Exit status: ${status}"; echo "Ended at: $(date -Is)"' EXIT

if [ -z "${CONTAINER_PATH}" ]; then
  if [ -d "${SANDBOX_PATH}" ]; then
    CONTAINER_PATH=${SANDBOX_PATH}
  else
    CONTAINER_PATH=${SIF_PATH}
  fi
fi

echo "Logging to ${LOG_FILE}"
echo "Repo: ${REPO_DIR}"
echo "Workspace: ${WORKSPACE}"
echo "Container path: ${CONTAINER_PATH}"
echo "NPROC: ${NPROC}"
echo "Tensor parallel: ${TENSOR_MODEL_PARALLEL_SIZE}"
echo "Pipeline parallel: ${PIPELINE_MODEL_PARALLEL_SIZE}"
echo "Expert parallel: ${EXPERT_MODEL_PARALLEL_SIZE}"
echo "Expert tensor parallel: ${EXPERT_TENSOR_PARALLEL_SIZE}"
echo "Force preprocess: ${FORCE_PREPROCESS}"
echo "Force convert: ${FORCE_CONVERT}"
echo "Host: $(hostname)"
echo "User: $(id -un)"
echo "PATH: ${PATH}"

test -d "${REPO_DIR}"
test -e "${CONTAINER_PATH}" || {
  echo "Missing container: ${CONTAINER_PATH}" >&2
  echo "Run scripts/build_slime_sandbox.sh first, or provide CONTAINER_PATH." >&2
  exit 1
}
test -f "${WORKSPACE}/models/gpt-oss-20b/config.json"
if ! find "${WORKSPACE}/models/gpt-oss-20b" -maxdepth 1 -type f \( -name '*.safetensors' -o -name 'pytorch_model*.bin' \) -print -quit 2>/dev/null | grep -q .; then
  echo "Raw model weights are missing under ${WORKSPACE}/models/gpt-oss-20b" >&2
  exit 1
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

BIND_ARGS=(--bind "${REPO_DIR}:${REPO_DIR}" --bind "${WORKSPACE}:${WORKSPACE}")
if [ -n "${APPTAINER_EXTRA_BINDS:-}" ]; then
  IFS=',' read -r -a EXTRA_BINDS <<< "${APPTAINER_EXTRA_BINDS}"
  for bind_path in "${EXTRA_BINDS[@]}"; do
    [ -n "${bind_path}" ] && BIND_ARGS+=(--bind "${bind_path}")
  done
fi

echo "Started at: $(date -Is)"
echo "Runtime: ${RUNTIME}"
"${RUNTIME}" exec --nv --cleanenv \
  "${BIND_ARGS[@]}" \
  "${CONTAINER_PATH}" \
  bash -c "
    set -euo pipefail
    cd '${REPO_DIR}'
    export WORKSPACE='${WORKSPACE}'
    export NPROC='${NPROC}'
    export TENSOR_MODEL_PARALLEL_SIZE='${TENSOR_MODEL_PARALLEL_SIZE}'
    export PIPELINE_MODEL_PARALLEL_SIZE='${PIPELINE_MODEL_PARALLEL_SIZE}'
    export EXPERT_MODEL_PARALLEL_SIZE='${EXPERT_MODEL_PARALLEL_SIZE}'
    export EXPERT_TENSOR_PARALLEL_SIZE='${EXPERT_TENSOR_PARALLEL_SIZE}'
    export FORCE_PREPROCESS='${FORCE_PREPROCESS}'
    export FORCE_CONVERT='${FORCE_CONVERT}'
    echo \"CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-unset}\"
    ls -l /dev/nvidia* 2>/dev/null || true
    python3 - <<'PY'
import os
import torch
required = int(os.environ.get('NPROC', '1'))
count = torch.cuda.device_count()
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'gpus', count)
if not torch.cuda.is_available() or count < required:
    raise SystemExit(f'Need at least {required} visible CUDA devices, got {count}. Re-submit with {required} GPUs.')
PY
    bash scripts/job_prepare_gpt_oss_20b_slime.sh
  "

echo "Finished conversion at: $(date -Is)"
