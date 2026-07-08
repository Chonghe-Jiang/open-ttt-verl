#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-guidance_ttt/config/backup/erdos_gpt_oss_20b_8gpu_50step.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/guidance_ttt/erdos_gpt_oss_20b_8gpu_50step}"
export NCCL_NET="${NCCL_NET:-Socket}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export TMPDIR="${TMPDIR:-$PWD/.tmp}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PWD/.triton_cache}"
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO="${RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO:-0}"
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR"

python -m guidance_ttt.main_erdos \
  --config "$CONFIG" \
  "$@"

python -m guidance_ttt.run_summary "$OUTPUT_DIR" --config "$CONFIG"
