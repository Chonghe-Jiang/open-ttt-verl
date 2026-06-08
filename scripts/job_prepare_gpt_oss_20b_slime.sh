#!/bin/bash
set -euo pipefail

cd /home/qua/code/open-ttt

export WORKSPACE=${WORKSPACE:-${HOME}/scratch/open-ttt-workspace}
export LOG_ROOT=${LOG_ROOT:-${WORKSPACE}/logs}
export NPROC=${NPROC:-8}

mkdir -p "${LOG_ROOT}"
JOB_LOG="${LOG_ROOT}/job_prepare_gpt_oss_20b_slime_${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S).log"
touch "${JOB_LOG}"
rm -f "${LOG_ROOT}/job_prepare_gpt_oss_20b_slime.latest.log"
ln -s "${JOB_LOG}" "${LOG_ROOT}/job_prepare_gpt_oss_20b_slime.latest.log"
exec >> "${JOB_LOG}" 2>&1

echo "Outer job log: ${JOB_LOG}"
echo "Workspace: ${WORKSPACE}"
echo "NPROC: ${NPROC}"

bash scripts/prepare_gpt_oss_20b_slime.sh
