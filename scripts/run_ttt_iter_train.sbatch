#!/usr/bin/env bash
#SBATCH --job-name=no-shell
#SBATCH --partition=p5en-odcr-queue
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00
#SBATCH --output=/fsx/xuanj/ttt-discover/logs/slurm/%x-%j.out
#SBATCH --error=/fsx/xuanj/ttt-discover/logs/slurm/%x-%j.err
set -euo pipefail

cd /fsx/xuanj/ttt-discover
mkdir -p logs/slurm logs/vllm results/ttt_iter results/ttt_iter_lora scratch_iter_train

NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
NODE_ARR=($NODES)
NODE_VLLM=${NODE_ARR[0]}
NODE_TRAIN=${NODE_ARR[1]}
TRAIN_TASK="${TRAIN_TASK:-cbl__low_av_tight_dl_large_oh}"
JOBID=${SLURM_JOB_ID:-local}
PORT=8001
VLLM_URL="http://${NODE_VLLM}:${PORT}"
echo "[ttt-iter-train] vllm_node=$NODE_VLLM train_node=$NODE_TRAIN task=$TRAIN_TASK"

srun --nodes=1 --ntasks=1 --nodelist=$NODE_VLLM --exclusive --output=/dev/null \
  bash -lc "
    set -euo pipefail
    source /fsx/xuanj/ttt-discover/.venv/bin/activate
    export HF_HOME=/fsx/xuanj/ttt-discover/.hf-cache
    export FLASHINFER_DISABLE_VERSION_CHECK=1
    export VLLM_ATTENTION_BACKEND=FLASH_ATTN
    export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
    export OMP_NUM_THREADS=4
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    exec python -m vllm.entrypoints.openai.api_server \
      --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
      --port $PORT --host 0.0.0.0 \
      --tensor-parallel-size 8 \
      --gpu-memory-utilization 0.85 \
      --max-model-len 65536 \
      --dtype bfloat16 \
      --enable-lora --max-lora-rank 64 --max-loras 4 \
      --disable-log-requests \
      > /fsx/xuanj/ttt-discover/logs/vllm/ttt_iter_train_vllm_${JOBID}.log 2>&1
  " &
VLLM_SRUN_PID=$!

echo "[ttt-iter-train] waiting for vllm at $VLLM_URL"
for i in $(seq 1 360); do
  if curl -sf "${VLLM_URL}/health" > /dev/null 2>&1; then
    echo "[ttt-iter-train] vllm up after $((i*5))s"
    break
  fi
  sleep 5
done
if ! curl -sf "${VLLM_URL}/health" > /dev/null ; then
  echo "[ttt-iter-train] FATAL vllm did not start"
  kill $VLLM_SRUN_PID 2>/dev/null || true
  exit 2
fi

srun --nodes=1 --ntasks=1 --nodelist=$NODE_TRAIN --exclusive \
  bash -lc "
    set -euo pipefail
    source /fsx/xuanj/ttt-discover/.venv/bin/activate
    export HF_HOME=/fsx/xuanj/ttt-discover/.hf-cache
    export OMP_NUM_THREADS=4
    export TOKENIZERS_PARALLELISM=false
    export CUDA_VISIBLE_DEVICES=0
    python /fsx/xuanj/ttt-discover/scripts/ttt_iterative.py \
      --task $TRAIN_TASK \
      --vllm-url $VLLM_URL \
      --num-steps 5 \
      --group-size 8 \
      --lr 4e-5 \
      --lora-rank 32 \
      --kl-coef 0.1 \
      --temperature 0.9 \
      --max-tokens-per-turn 16384 \
      --max-turns 3 \
      --gen-timeout 1800 \
      --eval-timeout 1800 \
      --rollout-concurrency 4 \
      > /fsx/xuanj/ttt-discover/logs/slurm/ttt-iter-train-${TRAIN_TASK}-${JOBID}.log 2>&1
  "

kill $VLLM_SRUN_PID 2>/dev/null || true
wait $VLLM_SRUN_PID 2>/dev/null || true
echo "[ttt-iter-train] done"
