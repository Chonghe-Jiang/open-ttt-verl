#!/usr/bin/env bash
# Launch vLLM OpenAI-compatible server for DeepSeek-R1-Distill-Qwen-8B on a p5en compute node.
# 8B model fits on 1 H200 with room; we use TP=1 and start 8 server instances on ports 8000..8007
# so that 8 inflight requests truly use 8 GPUs in parallel (avoids the all-rollouts-on-one-GPU
# bottleneck and lets us swap in LoRA adapters per-task during TTT-Discover).
#
# Usage on a compute node (after `salloc -p p5en-odcr-queue -N 1 --gres=gpu:8`):
#   bash launch_vllm.sh
# Or:
#   GPU_ID=0 PORT=8000 bash launch_vllm.sh single
set -euo pipefail

MODEL=${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-8B}
HF_HOME=${HF_HOME:-/fsx/xuanj/ttt-discover/.hf-cache}
LOG_DIR=${LOG_DIR:-/fsx/xuanj/ttt-discover/logs/vllm}
mkdir -p "$HF_HOME" "$LOG_DIR"

source /fsx/xuanj/ttt-discover/.venv/bin/activate
export HF_HOME

mode=${1:-multi}

launch_one() {
  local gpu=$1
  local port=$2
  local logf="$LOG_DIR/vllm_gpu${gpu}_port${port}.log"
  echo "[launch_vllm] gpu=$gpu port=$port log=$logf"
  CUDA_VISIBLE_DEVICES=$gpu \
    python -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" \
      --port $port \
      --host 0.0.0.0 \
      --tensor-parallel-size 1 \
      --gpu-memory-utilization 0.85 \
      --max-model-len 32768 \
      --dtype bfloat16 \
      --enable-lora \
      --max-loras 4 \
      --max-lora-rank 64 \
      --disable-log-requests \
      > "$logf" 2>&1 &
  echo "  pid=$!"
}

if [[ "$mode" == "single" ]]; then
  launch_one "${GPU_ID:-0}" "${PORT:-8000}"
  wait
else
  for i in 0 1 2 3 4 5 6 7; do
    launch_one $i $((8000 + i))
  done
  echo "[launch_vllm] launched 8 servers on ports 8000-8007"
  echo "[launch_vllm] waiting; use 'pkill -f vllm.entrypoints.openai.api_server' to kill"
  wait
fi
