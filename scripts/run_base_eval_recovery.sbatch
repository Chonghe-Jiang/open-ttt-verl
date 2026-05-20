#!/usr/bin/env bash
#SBATCH --job-name=no-shell
#SBATCH --partition=p5en-odcr-queue
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00
#SBATCH --output=/fsx/xuanj/ttt-discover/logs/slurm/%x-%j.out
#SBATCH --error=/fsx/xuanj/ttt-discover/logs/slurm/%x-%j.err
set -euo pipefail

cd /fsx/xuanj/ttt-discover
mkdir -p logs/slurm logs/vllm results/base_iter/eval scratch_iter

NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
NODE_ARR=($NODES)
NODE_VLLM=${NODE_ARR[0]}
JOBID=${SLURM_JOB_ID:-local}
PORT=8000
VLLM_URL="http://${NODE_VLLM}:${PORT}"
echo "[base-recovery] vllm=$NODE_VLLM eval_nodes=${NODE_ARR[@]:1}"

srun --nodes=1 --ntasks=1 --nodelist=$NODE_VLLM --exclusive --output=/dev/null \
  bash -lc "
    set -euo pipefail
    source /fsx/xuanj/ttt-discover/.venv/bin/activate
    export HF_HOME=/fsx/xuanj/ttt-discover/.hf-cache
    export FLASHINFER_DISABLE_VERSION_CHECK=1
    export VLLM_ATTENTION_BACKEND=FLASH_ATTN
    export OMP_NUM_THREADS=4
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    exec python -m vllm.entrypoints.openai.api_server \
      --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
      --port $PORT --host 0.0.0.0 \
      --tensor-parallel-size 8 --gpu-memory-utilization 0.85 \
      --max-model-len 65536 --dtype bfloat16 \
      --disable-log-requests \
      > /fsx/xuanj/ttt-discover/logs/vllm/base_recovery_vllm_${JOBID}.log 2>&1
  " &
VLLM_PID=$!

echo "[base-recovery] waiting for vllm at $VLLM_URL"
for i in $(seq 1 360); do
  curl -sf "${VLLM_URL}/health" > /dev/null 2>&1 && { echo "vllm up after $((i*5))s"; break; }
  sleep 5
done
if ! curl -sf "${VLLM_URL}/health" > /dev/null ; then
  echo "[base-recovery] FATAL vllm did not start"
  kill $VLLM_PID 2>/dev/null || true
  exit 2
fi

# 3 shards across 3 nodes; each runs through ALL tasks but skip-if-exists
for shard in 0 1 2 ; do
  EVAL_NODE=${NODE_ARR[$((shard+1))]}
  echo "[base-recovery] dispatching shard $shard to $EVAL_NODE"
  srun --nodes=1 --ntasks=1 --nodelist=$EVAL_NODE --exclusive \
    bash -lc "
      set -euo pipefail
      source /fsx/xuanj/ttt-discover/.venv/bin/activate
      export HF_HOME=/fsx/xuanj/ttt-discover/.hf-cache
      export OMP_NUM_THREADS=4
      python /fsx/xuanj/ttt-discover/scripts/base_eval_iterative.py \
        --server $VLLM_URL \
        --shard-id $shard --num-shards 3 \
        --num-rollouts 8 \
        --max-turns 3 --max-tokens 16384 --temperature 0.9 \
        --concurrency 4 --gen-timeout 1800 --eval-timeout 1800 \
        --skip-if-exists \
        > /fsx/xuanj/ttt-discover/logs/slurm/base-recovery-shard${shard}-${JOBID}.log 2>&1
    " &
done

wait
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true
echo "[base-recovery] all done"
