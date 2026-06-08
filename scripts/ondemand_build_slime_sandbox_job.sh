#!/bin/bash
set -Eeuo pipefail

cd /home/qua/code/open-ttt

export WORKSPACE=${WORKSPACE:-/home/qua/scratch/open-ttt-workspace}
export LOG_ROOT=${LOG_ROOT:-${WORKSPACE}/logs}
export IMAGE_ROOT=${IMAGE_ROOT:-${WORKSPACE}/images}
export SANDBOX_PATH=${SANDBOX_PATH:-${IMAGE_ROOT}/slime_latest.sandbox}
export SLIME_IMAGE=${SLIME_IMAGE:-docker://slimerl/slime:latest}
export APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-${WORKSPACE}/apptainer-cache}
export SINGULARITY_CACHEDIR=${SINGULARITY_CACHEDIR:-${WORKSPACE}/apptainer-cache}
export WORKSPACE_TMP=${WORKSPACE_TMP:-${WORKSPACE}/tmp}
export TMPDIR=${WORKSPACE_TMP}
export APPTAINER_TMPDIR=${APPTAINER_TMPDIR:-${WORKSPACE_TMP}}
export SINGULARITY_TMPDIR=${SINGULARITY_TMPDIR:-${WORKSPACE_TMP}}

mkdir -p "${LOG_ROOT}" "${IMAGE_ROOT}" "${APPTAINER_CACHEDIR}" "${TMPDIR}"

echo "=== build slime sandbox job ==="
echo "time: $(date -Is)"
echo "host: $(hostname)"
echo "pwd: $(pwd)"
echo "workspace: ${WORKSPACE}"
echo "sandbox: ${SANDBOX_PATH}"
echo "slurm job: ${SLURM_JOB_ID:-none}"
echo "slurm gpus: ${SLURM_GPUS_ON_NODE:-unset}"
echo "slurm cpus: ${SLURM_CPUS_ON_NODE:-unset}"
echo "slurm mem: ${SLURM_MEM_PER_NODE:-unset}"
echo "path: ${PATH}"

bash scripts/build_slime_sandbox.sh
