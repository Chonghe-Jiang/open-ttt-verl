#!/bin/bash
set -euo pipefail

WORKSPACE=${WORKSPACE:-${HOME}/scratch/open-ttt-workspace}
LOG_ROOT=${LOG_ROOT:-${WORKSPACE}/logs}
IMAGE_ROOT=${IMAGE_ROOT:-${WORKSPACE}/images}
WORKSPACE_TMP=${WORKSPACE_TMP:-${WORKSPACE}/tmp}
SLIME_IMAGE=${SLIME_IMAGE:-docker://slimerl/slime:latest}
SIF_PATH=${SIF_PATH:-${IMAGE_ROOT}/slime_latest.sif}
FORCE_IMAGE_PULL=${FORCE_IMAGE_PULL:-0}

mkdir -p "${LOG_ROOT}" "${IMAGE_ROOT}" "${WORKSPACE}/apptainer-cache" "${WORKSPACE_TMP}"

LOG_FILE=${LOG_FILE:-${LOG_ROOT}/build_slime_sif_$(date +%Y%m%d_%H%M%S).log}
touch "${LOG_FILE}"
rm -f "${LOG_ROOT}/build_slime_sif.latest.log"
ln -s "${LOG_FILE}" "${LOG_ROOT}/build_slime_sif.latest.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
trap 'status=$?; echo "Exit status: ${status}"; echo "Ended at: $(date -Is)"' EXIT

echo "Logging to ${LOG_FILE}"
echo "Workspace: ${WORKSPACE}"
echo "Image source: ${SLIME_IMAGE}"
echo "SIF path: ${SIF_PATH}"
echo "Force pull: ${FORCE_IMAGE_PULL}"
echo "Host: $(hostname)"
echo "User: $(id -un)"
echo "PATH: ${PATH}"

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
  echo "Available module command: $(command -v module || true)"
  echo "Available apptainer: $(command -v apptainer || true)"
  echo "Available singularity: $(command -v singularity || true)"
  exit 1
fi

export APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-${WORKSPACE}/apptainer-cache}
export SINGULARITY_CACHEDIR=${SINGULARITY_CACHEDIR:-${WORKSPACE}/apptainer-cache}
export TMPDIR=${WORKSPACE_TMP}
export APPTAINER_TMPDIR=${APPTAINER_TMPDIR:-${WORKSPACE_TMP}}
export SINGULARITY_TMPDIR=${SINGULARITY_TMPDIR:-${WORKSPACE_TMP}}

if [ -s "${SIF_PATH}" ] && [ "${FORCE_IMAGE_PULL}" != "1" ]; then
  echo "SIF already exists; skipping pull."
  ls -lh "${SIF_PATH}"
  exit 0
fi

TMP_SIF="${SIF_PATH}.tmp.${SLURM_JOB_ID:-manual}.$$"
rm -f "${TMP_SIF}"

echo "Runtime: ${RUNTIME}"
echo "Cache: ${APPTAINER_CACHEDIR}"
echo "Tmp: ${TMPDIR}"
echo "Started at: $(date -Is)"

"${RUNTIME}" pull "${TMP_SIF}" "${SLIME_IMAGE}"
test -s "${TMP_SIF}"
mv -f "${TMP_SIF}" "${SIF_PATH}"

echo "Finished at: $(date -Is)"
ls -lh "${SIF_PATH}"
