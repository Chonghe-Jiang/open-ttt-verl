#!/usr/bin/env bash
set -euo pipefail

# Run the portable TTT-Discover Docker image.
#
# Modes:
#   preflight  - check CUDA, torch, verl, vLLM, and flash-attn imports
#   preflight-forward - load GPT-OSS once and run a tiny actor/ref forward
#   prepare    - parse the selected TTT config with --prepare-only
#   prepare-qwen8b - parse the Qwen3-8B official TTT config
#   shell      - open an interactive container shell
#   run        - default; launch the selected TTT Erdos run
#   run-qwen8b - launch the Qwen3-8B official TTT Erdos run

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

IMAGE_TAG="${IMAGE_TAG:-open-ttt-verl:ttt-vllm}"
GPUS="${GPUS:-0,1,2,3}"
CONFIG="${CONFIG:-verl_ttt_discover/config/erdos_4gpu_b200_gptoss20b_bf16_official.yaml}"
HOST_HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf_cache}"
HOST_OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs}"
MODEL_PATH="${MODEL_PATH:-}"
ATTN_IMPL="${ATTN_IMPL:-}"
USE_HUB_KERNELS="${USE_HUB_KERNELS:-0}"
RUNTIME_LOCK="${RUNTIME_LOCK:-docker/b200_vllm017_env.lock.json}"
DOCKER_GPUS="${DOCKER_GPUS:-all}"
SHM_SIZE="${SHM_SIZE:-256g}"

MODE="${1:-run}"
case "${MODE}" in
  run|run-qwen8b|preflight|preflight-forward|prepare|prepare-qwen8b|shell|bash)
    shift || true
    ;;
  *)
    MODE="run"
    ;;
esac

if [[ "${MODE}" == "run-qwen8b" || "${MODE}" == "prepare-qwen8b" ]]; then
  if [[ "${CONFIG}" == "verl_ttt_discover/config/erdos_4gpu_b200_gptoss20b_bf16_official.yaml" ]]; then
    CONFIG="verl_ttt_discover/config/erdos_4gpu_b200_qwen3_8b_official.yaml"
  fi
fi

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
  -e "USE_HUB_KERNELS=${USE_HUB_KERNELS}"
  -e "RUNTIME_LOCK=${RUNTIME_LOCK}"
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

from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input  # noqa: F401
from flash_attn.ops.triton.rotary import apply_rotary  # noqa: F401

print("import flash_attn.bert_padding: ok")
print("import flash_attn.ops.triton.rotary: ok")
PY
python scripts/ttt_discover/docker_runtime_guard.py \
  --lock "${RUNTIME_LOCK}" \
  --require-cuda \
  --require-use-hub-kernels-zero \
  --check-flash-attn \
  --forbid-vllm-flash-attn3
'
    exec docker "${docker_args[@]}" "${IMAGE_TAG}" -lc "${inner_command}"
    ;;
  preflight-forward)
    inner_command='
set -euo pipefail
python scripts/ttt_discover/docker_runtime_guard.py \
  --lock "${RUNTIME_LOCK}" \
  --require-cuda \
  --require-use-hub-kernels-zero \
  --check-flash-attn \
  --forbid-vllm-flash-attn3
python - <<'"'"'PY'"'"'
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = os.environ.get("MODEL_PATH") or "unsloth/gpt-oss-20b-BF16"
attn_impl = os.environ.get("ATTN_IMPL") or "flash_attention_2"
print(f"Loading {model_path} with attn_implementation={attn_impl}")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    attn_implementation=attn_impl,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
inputs = tokenizer("Return the number 1.", return_tensors="pt").to(model.device)
with torch.no_grad():
    output = model(**inputs)
print(f"forward ok; logits_shape={tuple(output.logits.shape)}")
PY
'
    exec docker "${docker_args[@]}" "${IMAGE_TAG}" -lc "${inner_command}"
    ;;
  prepare)
    inner_command='scripts/ttt_discover/run_erdos_gptoss_bf16_4gpu_b200.sh --prepare-only "$@"'
    exec docker "${docker_args[@]}" "${IMAGE_TAG}" -lc "${inner_command}" bash "$@"
    ;;
  prepare-qwen8b)
    inner_command='scripts/ttt_discover/run_erdos_qwen3_8b_4gpu_b200.sh --prepare-only "$@"'
    exec docker "${docker_args[@]}" "${IMAGE_TAG}" -lc "${inner_command}" bash "$@"
    ;;
  shell|bash)
    exec docker "${docker_args[@]}" "${IMAGE_TAG}" "$@"
    ;;
  run)
    inner_command='
set -euo pipefail
python scripts/ttt_discover/docker_runtime_guard.py \
  --lock "${RUNTIME_LOCK}" \
  --require-cuda \
  --require-use-hub-kernels-zero \
  --check-flash-attn \
  --forbid-vllm-flash-attn3
scripts/ttt_discover/run_erdos_gptoss_bf16_4gpu_b200.sh "$@"
'
    exec docker "${docker_args[@]}" "${IMAGE_TAG}" -lc "${inner_command}" bash "$@"
    ;;
  run-qwen8b)
    inner_command='
set -euo pipefail
python scripts/ttt_discover/docker_runtime_guard.py \
  --lock "${RUNTIME_LOCK}" \
  --require-cuda \
  --require-use-hub-kernels-zero \
  --check-flash-attn \
  --forbid-vllm-flash-attn3
scripts/ttt_discover/run_erdos_qwen3_8b_4gpu_b200.sh "$@"
'
    exec docker "${docker_args[@]}" "${IMAGE_TAG}" -lc "${inner_command}" bash "$@"
    ;;
esac
