#!/usr/bin/env bash
# Per-node TTT shard runner. Args: SHARD_ID NUM_SHARDS PORT
set -euo pipefail
SHARD_ID=${1:?shard id}
NUM_SHARDS=${2:?num shards}
PORT=${3:?port}

source /fsx/xuanj/ttt-discover/.venv/bin/activate
export HF_HOME=/fsx/xuanj/ttt-discover/.hf-cache
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export FLASHINFER_DISABLE_VERSION_CHECK=1
export OMP_NUM_THREADS=4

NODE=$(hostname)
JOBID=${SLURM_JOB_ID:-local}
LOG_DIR=/fsx/xuanj/ttt-discover/logs/vllm
mkdir -p "$LOG_DIR"

echo "[shard $SHARD_ID node=$NODE] starting vllm on port $PORT"
nohup python -m vllm.entrypoints.openai.api_server \
  --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
  --port $PORT --host 127.0.0.1 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 32768 \
  --dtype bfloat16 \
  --disable-log-requests \
  > "$LOG_DIR/ttt_shard${SHARD_ID}_${JOBID}.log" 2>&1 &
VLLM_PID=$!

for i in $(seq 1 96); do
  if curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
    echo "[shard $SHARD_ID] vllm up after $((i*5))s"
    break
  fi
  sleep 5
done
if ! curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null ; then
  echo "[shard $SHARD_ID] FATAL vllm did not start"
  kill $VLLM_PID 2>/dev/null || true
  exit 2
fi

TASKS=(
  cbl__high_av_loose_dl_small_oh
  cbl__low_av_tight_dl_large_oh
  cbl__mixed_av_loose_dl_large_oh
  cbl_multi__high_av_loose_dl_small_oh
  cbl_multi__high_av_tight_dl_small_oh
  cbl_multi__low_av_loose_dl_small_oh
  fused_linear_ce
  gemm_opt__annoying
  gemm_opt__k_skewed
  gemm_opt__rectangles
  gemm_opt__squares
  gemm_opt__transformerish
  llm_sql__large
  poc_gen__heap_uaf
  poc_gen__uninit_value
  vdb_pareto__low_latency
  vdb_pareto__recall80_lat
)

for idx in $(seq 0 $((${#TASKS[@]} - 1))); do
  if [ $((idx % NUM_SHARDS)) -eq $SHARD_ID ]; then
    T=${TASKS[$idx]}
    echo "[shard $SHARD_ID] TTT for task $T (idx $idx)"
    LOG_F="/fsx/xuanj/ttt-discover/logs/slurm/ttt-task-${T}-${JOBID}.log"
    python /fsx/xuanj/ttt-discover/scripts/ttt_discover_minimal.py \
      --task "$T" \
      --vllm-url "http://127.0.0.1:${PORT}" \
      --num-steps 6 \
      --group-size 8 \
      --max-tokens 8192 \
      --gen-timeout 1800 \
      --eval-concurrency 8 \
      --preload-base-eval-dir /fsx/xuanj/ttt-discover/results/base/eval \
      --preload-base-solutions-root /fsx/xuanj/ttt-discover/src/frontier-cs/research/solutions \
      --preload-base-tag dsr1q3_8b_base \
      --no-train \
      --output-dir /fsx/xuanj/ttt-discover/results/ttt \
      --scratch-dir /fsx/xuanj/ttt-discover/scratch \
      > "$LOG_F" 2>&1 || echo "[shard $SHARD_ID] task $T failed (rc=$?)"
  fi
done

kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true
echo "[shard $SHARD_ID] done"
