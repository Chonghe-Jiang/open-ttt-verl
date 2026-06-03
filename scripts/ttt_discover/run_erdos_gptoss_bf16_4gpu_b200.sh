#!/usr/bin/env bash
set -euo pipefail

# Full-scale 4-GPU TTT-Discover/Erdos launch for unsloth/gpt-oss-20b-BF16.
# Override GPUS, CONFIG, HF_HOME, MODEL_PATH, or ATTN_IMPL from the shell.

GPUS="${GPUS:-0,1,2,3}"
CONFIG="${CONFIG:-verl_ttt_discover/config/erdos_4gpu_b200_gptoss20b_bf16_official.yaml}"
HF_HOME="${HF_HOME:-${PWD}/.hf_cache}"
MODEL_PATH="${MODEL_PATH:-}"
ATTN_IMPL="${ATTN_IMPL:-}"

export CUDA_VISIBLE_DEVICES="${GPUS}"
export HF_HOME
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"

overrides=()
if [[ -n "${MODEL_PATH}" ]]; then
  overrides+=("actor_rollout_ref.model.path=${MODEL_PATH}")
fi
if [[ -n "${ATTN_IMPL}" ]]; then
  overrides+=("actor_rollout_ref.model.override_config.attn_implementation=${ATTN_IMPL}")
fi

python -m verl_ttt_discover.main_erdos \
  --config "${CONFIG}" \
  "${overrides[@]}" \
  "$@"
