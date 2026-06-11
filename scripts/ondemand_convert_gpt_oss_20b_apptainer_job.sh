#!/bin/bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

export WORKSPACE=${WORKSPACE:-/work/mit/ppliang_mit/lsy/open-ttt-workspace}
export LOG_ROOT=${LOG_ROOT:-${WORKSPACE}/logs}
export IMAGE_ROOT=${IMAGE_ROOT:-${WORKSPACE}/images}
export SANDBOX_PATH=${SANDBOX_PATH:-${IMAGE_ROOT}/slime_latest.sandbox}
export SIF_PATH=${SIF_PATH:-${IMAGE_ROOT}/slime_latest.sif}
export NPROC=${NPROC:-2}
export TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-1}
export PIPELINE_MODEL_PARALLEL_SIZE=${PIPELINE_MODEL_PARALLEL_SIZE:-2}
export EXPERT_MODEL_PARALLEL_SIZE=${EXPERT_MODEL_PARALLEL_SIZE:-1}
export EXPERT_TENSOR_PARALLEL_SIZE=${EXPERT_TENSOR_PARALLEL_SIZE:-1}
export FORCE_PREPROCESS=${FORCE_PREPROCESS:-0}
export FORCE_CONVERT=${FORCE_CONVERT:-0}

mkdir -p "${LOG_ROOT}"

echo "=== convert gpt-oss-20b with apptainer job ==="
echo "time: $(date -Is)"
echo "host: $(hostname)"
echo "pwd: $(pwd)"
echo "workspace: ${WORKSPACE}"
echo "sandbox: ${SANDBOX_PATH}"
echo "sif: ${SIF_PATH}"
echo "nproc: ${NPROC}"
echo "tp: ${TENSOR_MODEL_PARALLEL_SIZE}"
echo "pp: ${PIPELINE_MODEL_PARALLEL_SIZE}"
echo "ep: ${EXPERT_MODEL_PARALLEL_SIZE}"
echo "etp: ${EXPERT_TENSOR_PARALLEL_SIZE}"
echo "slurm job: ${SLURM_JOB_ID:-none}"
echo "slurm gpus: ${SLURM_GPUS_ON_NODE:-unset}"
echo "slurm cpus: ${SLURM_CPUS_ON_NODE:-unset}"
echo "slurm mem: ${SLURM_MEM_PER_NODE:-unset}"
echo "path: ${PATH}"

bash scripts/convert_gpt_oss_20b_with_apptainer.sh
