#!/bin/bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

export WORKSPACE=${WORKSPACE:-/work/mit/ppliang_mit/lsy/open-ttt-workspace}
export LOG_ROOT=${LOG_ROOT:-${WORKSPACE}/logs}
export IMAGE_ROOT=${IMAGE_ROOT:-${WORKSPACE}/images}
export SIF_PATH=${SIF_PATH:-${IMAGE_ROOT}/slime_latest.sif}
export NPROC=${NPROC:-8}
export FORCE_PREPROCESS=${FORCE_PREPROCESS:-0}
export FORCE_CONVERT=${FORCE_CONVERT:-0}

mkdir -p "${LOG_ROOT}"

echo "=== convert gpt-oss-20b job ==="
echo "time: $(date -Is)"
echo "host: $(hostname)"
echo "pwd: $(pwd)"
echo "workspace: ${WORKSPACE}"
echo "sif: ${SIF_PATH}"
echo "nproc: ${NPROC}"
echo "slurm job: ${SLURM_JOB_ID:-none}"
echo "slurm gpus: ${SLURM_GPUS_ON_NODE:-unset}"
echo "slurm cpus: ${SLURM_CPUS_ON_NODE:-unset}"
echo "slurm mem: ${SLURM_MEM_PER_NODE:-unset}"
echo "path: ${PATH}"

test -s "${SIF_PATH}"
test -f "${WORKSPACE}/models/gpt-oss-20b/config.json"

bash scripts/convert_gpt_oss_20b_with_sif.sh
