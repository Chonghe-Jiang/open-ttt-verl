#!/bin/bash
set -euo pipefail

WORKSPACE=${WORKSPACE:-${HOME}/scratch/open-ttt-workspace}
LOG_ROOT=${LOG_ROOT:-${WORKSPACE}/logs}
IMAGE_ROOT=${IMAGE_ROOT:-${WORKSPACE}/images}
WORKSPACE_TMP=${WORKSPACE_TMP:-${WORKSPACE}/tmp}
SLIME_IMAGE=${SLIME_IMAGE:-docker://slimerl/slime:latest}
SANDBOX_PATH=${SANDBOX_PATH:-${IMAGE_ROOT}/slime_latest.sandbox}
FORCE_SANDBOX_BUILD=${FORCE_SANDBOX_BUILD:-0}

mkdir -p "${LOG_ROOT}" "${IMAGE_ROOT}" "${WORKSPACE}/apptainer-cache" "${WORKSPACE_TMP}"

LOG_FILE=${LOG_FILE:-${LOG_ROOT}/build_slime_sandbox_$(date +%Y%m%d_%H%M%S).log}
touch "${LOG_FILE}"
rm -f "${LOG_ROOT}/build_slime_sandbox.latest.log"
ln -s "${LOG_FILE}" "${LOG_ROOT}/build_slime_sandbox.latest.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
trap 'status=$?; echo "Exit status: ${status}"; echo "Ended at: $(date -Is)"' EXIT

echo "Logging to ${LOG_FILE}"
echo "Workspace: ${WORKSPACE}"
echo "Image source: ${SLIME_IMAGE}"
echo "Sandbox path: ${SANDBOX_PATH}"
echo "Force sandbox build: ${FORCE_SANDBOX_BUILD}"
echo "Host: $(hostname)"
echo "User: $(id -un)"
echo "PATH: ${PATH}"
df -h "${WORKSPACE}" /tmp || true

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

export APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-${WORKSPACE}/apptainer-cache}
export SINGULARITY_CACHEDIR=${SINGULARITY_CACHEDIR:-${WORKSPACE}/apptainer-cache}
export TMPDIR=${WORKSPACE_TMP}
export APPTAINER_TMPDIR=${APPTAINER_TMPDIR:-${WORKSPACE_TMP}}
export SINGULARITY_TMPDIR=${SINGULARITY_TMPDIR:-${WORKSPACE_TMP}}

if [ -d "${SANDBOX_PATH}" ] && [ "${FORCE_SANDBOX_BUILD}" != "1" ]; then
  echo "Sandbox already exists; skipping build."
  du -sh "${SANDBOX_PATH}"
  exit 0
fi

if [ -e "${SANDBOX_PATH}" ] && [ "${FORCE_SANDBOX_BUILD}" = "1" ]; then
  rm -rf "${SANDBOX_PATH}"
fi

TMP_SANDBOX="${SANDBOX_PATH}.tmp.${SLURM_JOB_ID:-manual}.$$"
case "${TMP_SANDBOX}" in
  "${IMAGE_ROOT}"/slime_latest.sandbox.tmp.*) rm -rf "${TMP_SANDBOX}" ;;
  *) echo "Refusing unsafe tmp sandbox path: ${TMP_SANDBOX}" >&2; exit 1 ;;
esac

echo "Runtime: ${RUNTIME}"
echo "Cache: ${APPTAINER_CACHEDIR}"
echo "Tmp: ${TMPDIR}"
echo "Started at: $(date -Is)"

"${RUNTIME}" build --sandbox "${TMP_SANDBOX}" "${SLIME_IMAGE}"
test -d "${TMP_SANDBOX}"
mv "${TMP_SANDBOX}" "${SANDBOX_PATH}"

echo "Finished sandbox build at: $(date -Is)"
du -sh "${SANDBOX_PATH}"
find "${SANDBOX_PATH}" -maxdepth 2 -type f \( -name python3 -o -name torchrun \) -print | head -20 || true
