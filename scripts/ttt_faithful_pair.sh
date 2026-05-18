#!/usr/bin/env bash
# Per-pair runner: launch vLLM on $VLLM_NODE and run TTT-Faithful for each task in $TASK_CSV
# sequentially on $TRAIN_NODE (using the shared vLLM).
set -euo pipefail
VLLM_NODE=${1:?vllm node}
TRAIN_NODE=${2:?train node}
PORT=${3:?port}
TASK_CSV=${4:?tasks csv}
JOBID=${5:-local}
PAIR_IDX=${6:-0}

VLLM_URL="http://${VLLM_NODE}:${PORT}"
echo "[pair $PAIR_IDX] vllm_url=$VLLM_URL tasks=$TASK_CSV"

# Launch vLLM on VLLM_NODE
srun --nodes=1 --ntasks=1 --nodelist=$VLLM_NODE --exclusive --output=/dev/null \
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
      --max-model-len 32768 \
      --dtype bfloat16 \
      --enable-lora --max-lora-rank 64 --max-loras 4 \
      --disable-log-requests \
      > /fsx/xuanj/ttt-discover/logs/vllm/sweep_p${PAIR_IDX}_vllm_${JOBID}.log 2>&1
  " &
VLLM_SRUN_PID=$!

# Wait for /health
for i in $(seq 1 360); do
  if curl -sf "${VLLM_URL}/health" > /dev/null 2>&1; then
    echo "[pair $PAIR_IDX] vllm up after $((i*5))s"
    break
  fi
  sleep 5
done
if ! curl -sf "${VLLM_URL}/health" > /dev/null ; then
  echo "[pair $PAIR_IDX] FATAL vllm did not start"
  kill $VLLM_SRUN_PID 2>/dev/null || true
  exit 2
fi

# Run TTT-Faithful for each task sequentially on TRAIN_NODE
IFS=',' read -ra TASKS_ARR <<< "$TASK_CSV"
for T in "${TASKS_ARR[@]}"; do
  echo "[pair $PAIR_IDX] running TTT-Faithful for $T"
  srun --nodes=1 --ntasks=1 --nodelist=$TRAIN_NODE --exclusive \
    bash -lc "
      set -euo pipefail
      source /fsx/xuanj/ttt-discover/.venv/bin/activate
      export HF_HOME=/fsx/xuanj/ttt-discover/.hf-cache
      export OMP_NUM_THREADS=4
      export TOKENIZERS_PARALLELISM=false
      export CUDA_VISIBLE_DEVICES=0
      python /fsx/xuanj/ttt-discover/scripts/ttt_faithful.py \
        --task $T \
        --vllm-url $VLLM_URL \
        --lora-name ttt-active-p${PAIR_IDX} \
        --num-steps 10 \
        --group-size 8 \
        --groups-per-batch 1 \
        --lr 4e-5 \
        --lora-rank 32 \
        --kl-coef 0.1 \
        --temperature 1.0 \
        --top-p 0.95 \
        --max-tokens 8192 \
        --gen-timeout 1800 \
        --eval-concurrency 8 \
        --eval-timeout 1200 \
        > /fsx/xuanj/ttt-discover/logs/slurm/ttt-faithful-${T}-${JOBID}.log 2>&1 \
        || echo \"[pair $PAIR_IDX] $T failed\"
    "
done

# Cleanup vLLM
kill $VLLM_SRUN_PID 2>/dev/null || true
wait $VLLM_SRUN_PID 2>/dev/null || true
echo "[pair $PAIR_IDX] done"
