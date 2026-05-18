#!/usr/bin/env bash
# One-off: resample shard 3's 4 tasks (all base-only) at temp=0.3
set -euo pipefail
PORT=${1:-8000}

source /fsx/xuanj/ttt-discover/.venv/bin/activate
export HF_HOME=/fsx/xuanj/ttt-discover/.hf-cache
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export FLASHINFER_DISABLE_VERSION_CHECK=1
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

JOBID=${SLURM_JOB_ID:-local}
LOG_DIR=/fsx/xuanj/ttt-discover/logs/vllm
mkdir -p "$LOG_DIR"

echo "[sft-shard3] starting vllm (base only)"
nohup python -m vllm.entrypoints.openai.api_server \
  --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
  --port $PORT --host 127.0.0.1 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 32768 \
  --dtype bfloat16 \
  --disable-log-requests \
  > "$LOG_DIR/sft_shard3_${JOBID}.log" 2>&1 &
VLLM_PID=$!

for i in $(seq 1 240); do
  if curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
    echo "[sft-shard3] vllm up after $((i*5))s"
    break
  fi
  sleep 5
done
if ! curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null ; then
  echo "[sft-shard3] FATAL vllm did not start"
  kill $VLLM_PID 2>/dev/null || true
  exit 2
fi

TASKS=(
  cbl_multi__high_av_loose_dl_small_oh
  gemm_opt__annoying
  gemm_opt__transformerish
  vdb_pareto__low_latency
)

for T in "${TASKS[@]}"; do
  echo "[sft-shard3] resample $T (base@temp=0.3)"
  python /fsx/xuanj/ttt-discover/scripts/sft_resample.py \
    --task "$T" \
    --vllm-url "http://127.0.0.1:${PORT}" \
    --lora-name "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B" \
    --num-rollouts 8 \
    --max-tokens 8192 \
    --temperature 0.3 --top-p 0.95 \
    --gen-timeout 1800 \
    --eval-concurrency 8 \
    --output-dir /fsx/xuanj/ttt-discover/results/sft_eval \
    > "/fsx/xuanj/ttt-discover/logs/slurm/sft-resample-${T}-${JOBID}.log" 2>&1 \
    || echo "[sft-shard3] $T resample failed"
done

kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true
echo "[sft-shard3] done"
