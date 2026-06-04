#!/usr/bin/env bash
set -euo pipefail

# Run the portable TTT-Discover Docker image.
#
# Modes:
#   preflight  - check CUDA, torch, verl, vLLM, and flash-attn imports
#   prepare    - parse the selected TTT config with --prepare-only
#   shell      - open an interactive container shell
#   run        - default; launch the selected TTT Erdos run

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

IMAGE_TAG="${IMAGE_TAG:-open-ttt-verl:ttt-vllm}"
GPUS="${GPUS:-0,1,2,3}"
CONFIG="${CONFIG:-verl_ttt_discover/config/erdos_4gpu_b200_gptoss20b_bf16_official.yaml}"
HOST_HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf_cache}"
HOST_OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs}"
MODEL_PATH="${MODEL_PATH:-}"
ATTN_IMPL="${ATTN_IMPL:-}"
DOCKER_GPUS="${DOCKER_GPUS:-all}"
SHM_SIZE="${SHM_SIZE:-256g}"

MODE="${1:-run}"
case "${MODE}" in
  run|preflight|prepare|shell|bash)
    shift || true
    ;;
  *)
    MODE="run"
    ;;
esac

mkdir -p "${HOST_HF_HOME}" "${HOST_OUTPUT_DIR}"

if [[ -n "${MODEL_PATH}" && -e "${MODEL_PATH}" ]]; then
  if [[ "${MODEL_PATH}" != /* ]]; then
    MODEL_PATH="$(cd "$(dirname "${MODEL_PATH}")" && pwd -P)/$(basename "${MODEL_PATH}")"
  fi
fi

docker_tty=()
if [[ -t 0 && -t 1 ]]; then
  docker_tty=(-it)
fi

docker_args=(
  run
  --rm
  "${docker_tty[@]}"
  --gpus "${DOCKER_GPUS}"
  --ipc=host
  --net=host
  --shm-size="${SHM_SIZE}"
  --ulimit memlock=-1
  --ulimit stack=67108864
  -v "${HOST_HF_HOME}:/hf_cache"
  -v "${HOST_OUTPUT_DIR}:/workspace/open-ttt-verl/outputs"
  -e "HF_HOME=/hf_cache"
  -e "HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}"
  -e "HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}"
  -e "RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-0}"
  -e "GPUS=${GPUS}"
  -e "CONFIG=${CONFIG}"
  -e "MODEL_PATH=${MODEL_PATH}"
  -e "ATTN_IMPL=${ATTN_IMPL}"
  -e "NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}"
  -e "NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-0}"
  -e "NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}"
)

for optional_env in NCCL_DEBUG NCCL_DEBUG_SUBSYS NCCL_SOCKET_IFNAME NCCL_IB_HCA; do
  if [[ -n "${!optional_env:-}" ]]; then
    docker_args+=(-e "${optional_env}=${!optional_env}")
  fi
done

if [[ -n "${MODEL_PATH}" && -e "${MODEL_PATH}" ]]; then
  model_mount="${MODEL_PATH}"
  if [[ -f "${model_mount}" ]]; then
    model_mount="$(dirname "${model_mount}")"
  fi
  docker_args+=(-v "${model_mount}:${model_mount}:ro")
fi

if [[ "${CONFIG}" = /* && -f "${CONFIG}" ]]; then
  config_dir="$(dirname "${CONFIG}")"
  docker_args+=(-v "${config_dir}:${config_dir}:ro")
fi

case "${MODE}" in
  preflight)
    inner_command='
set -euo pipefail
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
python - <<'"'"'PY'"'"'
import importlib
import os

import torch

print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_device_count={torch.cuda.device_count()}")
print(f"HF_HOME={os.environ.get('"'"'HF_HOME'"'"')}")

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside the container")

for module_name in ("verl", "verl_ttt_discover", "vllm", "flash_attn"):
    importlib.import_module(module_name)
    print(f"import {module_name}: ok")
PY
'
    exec docker "${docker_args[@]}" "${IMAGE_TAG}" -lc "${inner_command}"
    ;;
  prepare)
    inner_command='scripts/ttt_discover/run_erdos_gptoss_bf16_4gpu_b200.sh --prepare-only "$@"'
    exec docker "${docker_args[@]}" "${IMAGE_TAG}" -lc "${inner_command}" bash "$@"
    ;;
  shell|bash)
    exec docker "${docker_args[@]}" "${IMAGE_TAG}" "$@"
    ;;
  run)
    inner_command='scripts/ttt_discover/run_erdos_gptoss_bf16_4gpu_b200.sh "$@"'
    exec docker "${docker_args[@]}" "${IMAGE_TAG}" -lc "${inner_command}" bash "$@"
    ;;
esac
