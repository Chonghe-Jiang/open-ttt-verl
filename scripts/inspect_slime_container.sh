#!/bin/bash
set -euo pipefail

WORKSPACE=${WORKSPACE:-${HOME}/scratch/open-ttt-workspace}
LOG_ROOT=${LOG_ROOT:-${WORKSPACE}/logs}
IMAGE_ROOT=${IMAGE_ROOT:-${WORKSPACE}/images}
SANDBOX_PATH=${SANDBOX_PATH:-${IMAGE_ROOT}/slime_latest.sandbox}
SIF_PATH=${SIF_PATH:-${IMAGE_ROOT}/slime_latest.sif}
CONTAINER_PATH=${CONTAINER_PATH:-}
REQUIRE_CUDA=${REQUIRE_CUDA:-1}

mkdir -p "${LOG_ROOT}"

LOG_FILE=${LOG_FILE:-${LOG_ROOT}/inspect_slime_container_$(date +%Y%m%d_%H%M%S).log}
touch "${LOG_FILE}"
rm -f "${LOG_ROOT}/inspect_slime_container.latest.log"
ln -s "${LOG_FILE}" "${LOG_ROOT}/inspect_slime_container.latest.log"
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
echo "Workspace: ${WORKSPACE}"
echo "Container path: ${CONTAINER_PATH}"
echo "Require CUDA: ${REQUIRE_CUDA}"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unset}"
echo "SLURM_GPUS_ON_NODE: ${SLURM_GPUS_ON_NODE:-unset}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

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

test -e "${CONTAINER_PATH}"

APPTAINERENV_REQUIRE_CUDA="${REQUIRE_CUDA}" \
"${RUNTIME}" exec --nv --cleanenv \
  --bind /home/qua:/home/qua \
  "${CONTAINER_PATH}" \
  bash -c '
    set -euo pipefail
    echo "container-host: $(hostname)"
    echo "container CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
    echo "container NVIDIA_VISIBLE_DEVICES: ${NVIDIA_VISIBLE_DEVICES:-unset}"
    echo "container nvidia-smi: $(command -v nvidia-smi || true)"
    ls -l /dev/nvidia* 2>/dev/null || true
    echo "python: $(command -v python3)"
    python3 -V
    echo "torchrun: $(command -v torchrun || true)"
    python3 - <<PY
import importlib.util
import os
mods = ["torch", "transformers", "safetensors", "megatron", "ray", "sglang"]
for name in mods:
    spec = importlib.util.find_spec(name)
    status = "yes" if spec else "no"
    print("{}: {}".format(name, status))
import torch
print("torch_version:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())
for idx in range(torch.cuda.device_count()):
    print(f"cuda_device_{idx}:", torch.cuda.get_device_name(idx))
require_cuda = os.environ.get("REQUIRE_CUDA", "1").lower() not in {"0", "false", "no"}
if require_cuda and not torch.cuda.is_available():
    raise SystemExit("CUDA is not visible. Re-submit this job with a GPU allocation.")
PY
    python3 -m pip list | grep -Ei "torch|transformers|safetensors|megatron|ray|sglang|flash|deepspeed|triton" || true
    nvidia-smi || true
  '
