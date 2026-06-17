#!/usr/bin/env bash
# Source this file before running guidance-ttt on FIT_G75:
#   source scripts/env_g75.sh

set -euo pipefail

export GUIDANCE_TTT_ROOT=/home/wangshuo/wangshuo03/lsy/guidance-ttt
export OPEN_TTT_VERL_ROOT=${GUIDANCE_TTT_ROOT}
export GUIDANCE_TTT_MODEL_PATH=/home/wangshuo/wangshuo03/lsy/models/Qwen3-8B
export GUIDANCE_TTT_ATTENTION_IMPL=${GUIDANCE_TTT_ATTENTION_IMPL:-sdpa}

source /home/wangshuo/wangshuo03/anaconda3/etc/profile.d/conda.sh
conda activate spell

export PYTHONPATH="${GUIDANCE_TTT_ROOT}:${OPEN_TTT_VERL_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-/home/wangshuo/wangshuo03/.cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/hub}

cd "$GUIDANCE_TTT_ROOT"
