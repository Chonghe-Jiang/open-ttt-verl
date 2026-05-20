#!/usr/bin/env bash
#SBATCH --job-name=no-shell
#SBATCH --partition=p5en-odcr-queue
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00
#SBATCH --output=/fsx/xuanj/ttt-discover/logs/slurm/%x-%j.out
#SBATCH --error=/fsx/xuanj/ttt-discover/logs/slurm/%x-%j.err
set -euo pipefail

cd /fsx/xuanj/ttt-discover
mkdir -p logs/slurm logs/vllm results/ttt_iter_post_eval/eval scratch_post_ttt

NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
NODE_ARR=($NODES)
NODE_VLLM=${NODE_ARR[0]}
JOBID=${SLURM_JOB_ID:-local}
PORT=8005
URL="http://${NODE_VLLM}:${PORT}"

LORA_DIR="${LORA_DIR:-/fsx/xuanj/ttt-discover/results/ttt_iter_lora/4tasks_yuchen_seed1/step_003}"
LORA_NAME="ttt-iter-eval"
echo "[post-eval] vllm=$NODE_VLLM eval_nodes=${NODE_ARR[@]:1}"
echo "[post-eval] LoRA=$LORA_DIR"

if [[ ! -d "$LORA_DIR" ]]; then
  echo "[post-eval] FATAL LoRA dir not found: $LORA_DIR"
  exit 1
fi

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
      --tensor-parallel-size 8 --gpu-memory-utilization 0.85 \
      --max-model-len 65536 --dtype bfloat16 \
      --enable-lora --max-lora-rank 64 --max-loras 4 \
      --lora-modules ${LORA_NAME}=${LORA_DIR} \
      --disable-log-requests \
      > /fsx/xuanj/ttt-discover/logs/vllm/post_eval_vllm_${JOBID}.log 2>&1
  " &
VLLM_PID=$!

echo "[post-eval] waiting for vllm at $URL"
for i in $(seq 1 360); do
  curl -sf "${URL}/health" > /dev/null 2>&1 && { echo "vllm up after $((i*5))s"; break; }
  sleep 5
done
if ! curl -sf "${URL}/health" > /dev/null ; then
  echo "[post-eval] FATAL vllm did not start"
  kill $VLLM_PID 2>/dev/null || true
  exit 2
fi

# Higher concurrency per shard since KV cache usage is only ~1% — we can fit more in flight
for shard in 0 1 2 ; do
  EVAL_NODE=${NODE_ARR[$((shard+1))]}
  echo "[post-eval] dispatching shard $shard to $EVAL_NODE (skip-if-exists, concurrency=8)"
  srun --nodes=1 --ntasks=1 --nodelist=$EVAL_NODE --exclusive \
    bash -lc "
      set -euo pipefail
      source /fsx/xuanj/ttt-discover/.venv/bin/activate
      export HF_HOME=/fsx/xuanj/ttt-discover/.hf-cache
      export OMP_NUM_THREADS=4
      python /fsx/xuanj/ttt-discover/scripts/post_ttt_eval.py \
        --server $URL --model $LORA_NAME \
        --shard-id $shard --num-shards 3 \
        --num-rollouts 8 \
        --max-turns 3 --max-tokens 16384 --temperature 0.9 \
        --concurrency 8 --gen-timeout 1800 --eval-timeout 1800 \
        --skip-if-exists \
        > /fsx/xuanj/ttt-discover/logs/slurm/post-eval-shard${shard}-${JOBID}.log 2>&1
    " &
done

wait
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true
echo "[post-eval] all done"
