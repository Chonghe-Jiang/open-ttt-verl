#!/bin/bash
set -euo pipefail

IMAGE=${IMAGE:-open-ttt-slime:latest}
REPO_DIR=${REPO_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
WORKSPACE_HOST=${WORKSPACE_HOST:-${REPO_DIR}/workspace}
CONTAINER_NAME=${CONTAINER_NAME:-open-ttt-slime}
SHM_SIZE=${SHM_SIZE:-64g}
DOCKER_GPUS=${DOCKER_GPUS:-all}

mkdir -p \
  "${WORKSPACE_HOST}/models" \
  "${WORKSPACE_HOST}/ckpt" \
  "${WORKSPACE_HOST}/logs" \
  "${WORKSPACE_HOST}/data" \
  "${WORKSPACE_HOST}/tmp"

ENV_ARGS=(
  -e WORKSPACE=/root/workspace
  -e REPO_DIR=/root/workspace/erdos
  -e HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
  -e HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
)

if [ -n "${HF_TOKEN:-}" ]; then
  ENV_ARGS+=(-e HF_TOKEN="${HF_TOKEN}")
fi
if [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  ENV_ARGS+=(-e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}")
fi

if [ "$#" -eq 0 ]; then
  CMD=(/bin/bash)
else
  CMD=("$@")
fi

if [ -t 0 ]; then
  TTY_ARGS=(-it)
else
  TTY_ARGS=(-i)
fi

docker run --rm "${TTY_ARGS[@]}" \
  --name "${CONTAINER_NAME}" \
  --gpus "${DOCKER_GPUS}" \
  --ipc=host \
  --shm-size "${SHM_SIZE}" \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  "${ENV_ARGS[@]}" \
  -v "${WORKSPACE_HOST}:/root/workspace" \
  -v "${REPO_DIR}:/root/workspace/erdos" \
  -w /root/workspace/erdos \
  "${IMAGE}" \
  "${CMD[@]}"
