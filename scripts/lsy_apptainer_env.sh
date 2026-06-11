#!/bin/bash
# Source this file from /work/mit/ppliang_mit/lsy/open-ttt-verl-slime-ttt before running
# the Slime Apptainer helper scripts on the local cluster workspace.

export REPO_DIR=${REPO_DIR:-/work/mit/ppliang_mit/lsy/open-ttt-verl-slime-ttt}
export WORKSPACE=${WORKSPACE:-/work/mit/ppliang_mit/lsy/open-ttt-workspace}
export LOG_ROOT=${LOG_ROOT:-${WORKSPACE}/logs}
export IMAGE_ROOT=${IMAGE_ROOT:-${WORKSPACE}/images}
export WORKSPACE_TMP=${WORKSPACE_TMP:-${WORKSPACE}/tmp}

export SIF_PATH=${SIF_PATH:-${IMAGE_ROOT}/slime_latest.sif}
export SANDBOX_PATH=${SANDBOX_PATH:-${IMAGE_ROOT}/slime_latest.sandbox}
export CONTAINER_PATH=${CONTAINER_PATH:-${SIF_PATH}}

export MODEL_ID=${MODEL_ID:-Qwen/Qwen3-8B}
export MODEL_DIR=${MODEL_DIR:-${WORKSPACE}/models/Qwen3-8B}
export MODEL_RAW=${MODEL_RAW:-${MODEL_DIR}}
export CKPT_OUT=${CKPT_OUT:-${WORKSPACE}/ckpt/Qwen3-8B_torch_dist}
export REF_CKPT=${REF_CKPT:-${CKPT_OUT}}
export SAVE_CKPT=${SAVE_CKPT:-${WORKSPACE}/ckpt/Qwen3-8B_erdos_lora}
export ARCHIVE_PATH=${ARCHIVE_PATH:-${WORKSPACE}/data/qwen3_8b_erdos_archive.json}

export APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-${WORKSPACE}/apptainer-cache}
export SINGULARITY_CACHEDIR=${SINGULARITY_CACHEDIR:-${APPTAINER_CACHEDIR}}
export TMPDIR=${TMPDIR:-${WORKSPACE_TMP}}
export APPTAINER_TMPDIR=${APPTAINER_TMPDIR:-${WORKSPACE_TMP}}
export SINGULARITY_TMPDIR=${SINGULARITY_TMPDIR:-${WORKSPACE_TMP}}

mkdir -p "${LOG_ROOT}" "${IMAGE_ROOT}" "${WORKSPACE_TMP}" \
  "${WORKSPACE}/models" "${WORKSPACE}/ckpt" "${WORKSPACE}/data" \
  "${APPTAINER_CACHEDIR}"
