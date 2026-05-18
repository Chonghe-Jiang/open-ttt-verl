#!/usr/bin/env bash
# Per-shard runner: SFT-pretrain + resample for the assigned tasks.
# Args: SHARD_ID NUM_SHARDS PORT
set -euo pipefail
SHARD_ID=${1:?shard id}
NUM_SHARDS=${2:?num shards}
PORT=${3:?port}

source /fsx/xuanj/ttt-discover/.venv/bin/activate
export HF_HOME=/fsx/xuanj/ttt-discover/.hf-cache
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export OMP_NUM_THREADS=4

NODE=$(hostname)
JOBID=${SLURM_JOB_ID:-local}
LOG_DIR=/fsx/xuanj/ttt-discover/logs/vllm
mkdir -p "$LOG_DIR"

# Tasks list (same 17 as TTT). Only those with reward_01 > 0.05 in base eval will produce a LoRA.
# Others are skipped by sft_pretrain (no signal to memorize).
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

# Phase 1: SFT pretrain for shard's tasks (only one task at a time loads model)
# Use only GPU 0 for SFT (saves ~7 GPUs for nothing during SFT, but vLLM starts after)
SHARD_TASKS=()
for idx in $(seq 0 $((${#TASKS[@]} - 1))); do
  if [ $((idx % NUM_SHARDS)) -eq $SHARD_ID ]; then
    SHARD_TASKS+=("${TASKS[$idx]}")
  fi
done
echo "[sft-sweep shard $SHARD_ID] tasks: ${SHARD_TASKS[@]}"

for T in "${SHARD_TASKS[@]}"; do
  if [ -f "/fsx/xuanj/ttt-discover/results/sft_lora/${T}/adapter_model.safetensors" ]; then
    echo "[sft-sweep shard $SHARD_ID] LoRA already exists for $T, skipping pretrain"
    continue
  fi
  echo "[sft-sweep shard $SHARD_ID] SFT pretrain for $T"
  CUDA_VISIBLE_DEVICES=0 python /fsx/xuanj/ttt-discover/scripts/sft_pretrain.py \
    --task "$T" \
    --epochs 8 --lr 1e-4 --lora-rank 32 \
    --output-dir /fsx/xuanj/ttt-discover/results/sft_lora \
    > "/fsx/xuanj/ttt-discover/logs/slurm/sft-pretrain-${T}-${JOBID}.log" 2>&1 \
    || echo "[sft-sweep shard $SHARD_ID] $T pretrain skipped/failed (probably no base reward >= threshold)"
done

# Phase 2: Launch vLLM with ALL shard's LoRAs at once
LORA_ARGS=""
for T in "${SHARD_TASKS[@]}"; do
  LP="/fsx/xuanj/ttt-discover/results/sft_lora/${T}"
  if [ -d "$LP" ]; then
    LORA_ARGS="$LORA_ARGS sft-${T}=${LP}"
  fi
done
if [ -z "$LORA_ARGS" ]; then
  echo "[sft-sweep shard $SHARD_ID] no LoRAs to serve, exiting"
  exit 0
fi

echo "[sft-sweep shard $SHARD_ID] starting vllm with LoRAs: $LORA_ARGS"
nohup python -m vllm.entrypoints.openai.api_server \
  --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
  --port $PORT --host 127.0.0.1 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 32768 \
  --dtype bfloat16 \
  --enable-lora --max-lora-rank 64 --max-loras 8 \
  --lora-modules $LORA_ARGS \
  --disable-log-requests \
  > "$LOG_DIR/sft_sweep_shard${SHARD_ID}_${JOBID}.log" 2>&1 &
VLLM_PID=$!

for i in $(seq 1 240); do
  if curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
    echo "[sft-sweep shard $SHARD_ID] vllm up after $((i*5))s"
    break
  fi
  sleep 5
done
if ! curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null ; then
  echo "[sft-sweep shard $SHARD_ID] FATAL vllm did not start"
  kill $VLLM_PID 2>/dev/null || true
  exit 2
fi

# Phase 3: Resample 8 rollouts at temp=0.3 for each task
for T in "${SHARD_TASKS[@]}"; do
  LP="/fsx/xuanj/ttt-discover/results/sft_lora/${T}"
  if [ ! -d "$LP" ]; then
    echo "[sft-sweep shard $SHARD_ID] no LoRA for $T (no base signal), using BASE model in resample"
    LORA_NAME="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
  else
    LORA_NAME="sft-${T}"
  fi
  echo "[sft-sweep shard $SHARD_ID] resample task=$T lora_name=$LORA_NAME"
  python /fsx/xuanj/ttt-discover/scripts/sft_resample.py \
    --task "$T" \
    --vllm-url "http://127.0.0.1:${PORT}" \
    --lora-name "$LORA_NAME" \
    --num-rollouts 8 \
    --max-tokens 8192 \
    --temperature 0.3 --top-p 0.95 \
    --gen-timeout 1800 \
    --eval-concurrency 8 \
    --output-dir /fsx/xuanj/ttt-discover/results/sft_eval \
    > "/fsx/xuanj/ttt-discover/logs/slurm/sft-resample-${T}-${JOBID}.log" 2>&1 \
    || echo "[sft-sweep shard $SHARD_ID] $T resample failed"
done

kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true
echo "[sft-sweep shard $SHARD_ID] done"
