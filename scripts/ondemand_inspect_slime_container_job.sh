#!/bin/bash
set -Eeuo pipefail

cd /home/qua/code/open-ttt

export WORKSPACE=${WORKSPACE:-/home/qua/scratch/open-ttt-workspace}
export LOG_ROOT=${LOG_ROOT:-${WORKSPACE}/logs}
export IMAGE_ROOT=${IMAGE_ROOT:-${WORKSPACE}/images}
export SANDBOX_PATH=${SANDBOX_PATH:-${IMAGE_ROOT}/slime_latest.sandbox}
export SIF_PATH=${SIF_PATH:-${IMAGE_ROOT}/slime_latest.sif}

echo "=== inspect slime container job ==="
echo "time: $(date -Is)"
echo "host: $(hostname)"
echo "pwd: $(pwd)"
echo "workspace: ${WORKSPACE}"
echo "sandbox: ${SANDBOX_PATH}"
echo "sif: ${SIF_PATH}"
echo "slurm job: ${SLURM_JOB_ID:-none}"
echo "slurm gpus: ${SLURM_GPUS_ON_NODE:-unset}"

bash scripts/inspect_slime_container.sh
