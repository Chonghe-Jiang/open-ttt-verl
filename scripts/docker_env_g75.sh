#!/usr/bin/env bash
# Run guidance-ttt on FIT_G75 inside the prepared Docker image.
set -euo pipefail

IMAGE_TAG=${IMAGE_TAG:-guidance-ttt:g75-env}
GUIDANCE_TTT_ROOT=${GUIDANCE_TTT_ROOT:-/home/wangshuo/wangshuo03/lsy/guidance-ttt}
OPEN_TTT_VERL_ROOT=${OPEN_TTT_VERL_ROOT:-/home/wangshuo/wangshuo03/lsy/guidance-ttt}
MODEL_PATH=${MODEL_PATH:-/home/wangshuo/wangshuo03/lsy/models/Qwen3-8B}
HF_HOME=${HF_HOME:-/home/wangshuo/wangshuo03/.cache/huggingface}
OUTPUT_ROOT=${OUTPUT_ROOT:-${GUIDANCE_TTT_ROOT}/outputs}
DOCKER_GPUS=${DOCKER_GPUS:-all}
SHM_SIZE=${SHM_SIZE:-256g}

mkdir -p "$HF_HOME" "$OUTPUT_ROOT"

exec docker run --rm \
  --gpus "$DOCKER_GPUS" \
  --ipc=host \
  --net=host \
  --shm-size="$SHM_SIZE" \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /home/wangshuo/wangshuo03:/home/wangshuo/wangshuo03 \
  -w "$GUIDANCE_TTT_ROOT" \
  -e GUIDANCE_TTT_ROOT="$GUIDANCE_TTT_ROOT" \
  -e OPEN_TTT_VERL_ROOT="$OPEN_TTT_VERL_ROOT" \
  -e GUIDANCE_TTT_MODEL_PATH="$MODEL_PATH" \
  -e HF_HOME="$HF_HOME" \
  -e PYTHONPATH="$GUIDANCE_TTT_ROOT:$OPEN_TTT_VERL_ROOT:${PYTHONPATH:-}" \
  -e TOKENIZERS_PARALLELISM=false \
  -e HYDRA_FULL_ERROR=1 \
  -e RAY_DEDUP_LOGS=0 \
  "$IMAGE_TAG" "$@"
