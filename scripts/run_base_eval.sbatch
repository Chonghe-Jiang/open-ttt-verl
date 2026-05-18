#!/usr/bin/env bash
#SBATCH --job-name=ttt-base-gen
#SBATCH --partition=p5en-odcr-queue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=06:00:00
#SBATCH --output=/fsx/xuanj/ttt-discover/logs/slurm/%x-%j.out
#SBATCH --error=/fsx/xuanj/ttt-discover/logs/slurm/%x-%j.err
set -euo pipefail

cd /fsx/xuanj/ttt-discover
mkdir -p logs/slurm logs/vllm results/base/meta

source /fsx/xuanj/ttt-discover/.venv/bin/activate
export HF_HOME=/fsx/xuanj/ttt-discover/.hf-cache
# Force vLLM to use FlashAttention (FA3 on Hopper) instead of flashinfer (broken import on this image)
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
# Bypass flashinfer cubin/python version mismatch (0.6.6 vs 0.5.3 in user's local env)
export FLASHINFER_DISABLE_VERSION_CHECK=1
# Lower OMP threads - we don't need full 96 threads per worker
export OMP_NUM_THREADS=4

echo "[run_base_eval] node=$(hostname) gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')"

# Launch a single TP=8 vLLM server for the 8B model -- gives max throughput, simpler for plain generation
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
nohup python -m vllm.entrypoints.openai.api_server \
  --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
  --port 8000 --host 127.0.0.1 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 32768 \
  --dtype bfloat16 \
  --disable-log-requests \
  > logs/vllm/server_${SLURM_JOB_ID}.log 2>&1 &
VLLM_PID=$!
echo "[run_base_eval] vllm pid=$VLLM_PID, waiting for /health"

# Wait up to 8 min for vLLM to be ready
for i in $(seq 1 96); do
  if curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1; then
    echo "[run_base_eval] vllm healthy after ${i}*5=$((i*5))s"
    break
  fi
  sleep 5
done

if ! curl -sf http://127.0.0.1:8000/health > /dev/null; then
  echo "[run_base_eval] FATAL: vllm did not come up in 480s"
  tail -200 logs/vllm/server_${SLURM_JOB_ID}.log
  kill $VLLM_PID 2>/dev/null || true
  exit 2
fi

# Generate 19 tasks x 8 rollouts = 152 solutions
python /fsx/xuanj/ttt-discover/scripts/gen_solutions.py \
  --frontier-root /fsx/xuanj/ttt-discover/src/frontier-cs \
  --tasks-json /fsx/xuanj/ttt-discover/bench/tasks_19.json \
  --server http://127.0.0.1:8000 \
  --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
  --model-tag dsr1q3_8b_base \
  --num-rollouts 8 \
  --concurrency 16 \
  --max-tokens 8192 \
  --temperature 1.0 \
  --top-p 0.95 \
  --results-dir /fsx/xuanj/ttt-discover/results/base/meta

echo "[run_base_eval] generation done, killing vllm"
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true
echo "[run_base_eval] all done"
