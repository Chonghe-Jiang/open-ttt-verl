#!/bin/bash
set -euo pipefail

cd /home/qua/code/open-ttt

export WORKSPACE=${WORKSPACE:-${HOME}/scratch/open-ttt-workspace}
export LOG_ROOT=${LOG_ROOT:-${WORKSPACE}/logs}
export HF_MAX_WORKERS=${HF_MAX_WORKERS:-1}
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}

mkdir -p "${LOG_ROOT}"
JOB_LOG="${LOG_ROOT}/job_download_gpt_oss_20b_${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S).log"
touch "${JOB_LOG}"
rm -f "${LOG_ROOT}/job_download_gpt_oss_20b.latest.log"
ln -s "${JOB_LOG}" "${LOG_ROOT}/job_download_gpt_oss_20b.latest.log"
exec >> "${JOB_LOG}" 2>&1

echo "Outer job log: ${JOB_LOG}"
echo "Workspace: ${WORKSPACE}"
echo "HF_MAX_WORKERS: ${HF_MAX_WORKERS}"
echo "HF_HUB_DISABLE_XET: ${HF_HUB_DISABLE_XET}"

bash scripts/download_gpt_oss_20b.sh
