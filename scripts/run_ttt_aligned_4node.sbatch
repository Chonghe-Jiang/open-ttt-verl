#!/usr/bin/env bash
#SBATCH --job-name=no-shell
#SBATCH --partition=p5en-odcr-queue
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=36:00:00
#SBATCH --output=/fsx/xuanj/ttt-discover/logs/slurm/%x-%j.out
#SBATCH --error=/fsx/xuanj/ttt-discover/logs/slurm/%x-%j.err
set -euo pipefail

cd /fsx/xuanj/ttt-discover
mkdir -p logs/slurm logs/vllm \
  results/ttt_iter results/ttt_iter_lora \
  results/ttt_iter_post_eval/eval \
  scratch_iter_train scratch_post_ttt

NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
NODE_ARR=($NODES)
NODE_VLLM_TRAIN=${NODE_ARR[0]}
NODE_TRAIN=${NODE_ARR[1]}
NODE_VLLM_EVAL=${NODE_ARR[2]}
NODE_EVAL=${NODE_ARR[3]}

# 4 train tasks from Yuchen seed=1 sample
TRAIN_TASKS="cant_be_late_multi__low_availability_loose_deadline_small_overhead,cant_be_late__mixed_availability_loose_deadline_large_overhead,gemm_optimization__transformerish,cant_be_late_multi__high_availability_tight_deadline_small_overhead"
RUN_TAG="${RUN_TAG:-4tasks_yuchen_seed1}"
JOBID=${SLURM_JOB_ID:-local}
PORT_TRAIN=8003
PORT_EVAL=8004
URL_TRAIN="http://${NODE_VLLM_TRAIN}:${PORT_TRAIN}"
URL_EVAL="http://${NODE_VLLM_EVAL}:${PORT_EVAL}"
LORA_NAME_TRAIN="ttt-iter"
LORA_NAME_EVAL="ttt-iter-eval"
LORA_DIR_BASE="/fsx/xuanj/ttt-discover/results/ttt_iter_lora/${RUN_TAG}"

echo "[aligned] vllm_train=$NODE_VLLM_TRAIN train=$NODE_TRAIN vllm_eval=$NODE_VLLM_EVAL eval=$NODE_EVAL"
echo "[aligned] tasks=$TRAIN_TASKS"
echo "[aligned] run_tag=$RUN_TAG"

# vLLM #1: trainer-side
srun --nodes=1 --ntasks=1 --nodelist=$NODE_VLLM_TRAIN --exclusive --output=/dev/null \
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
      --port $PORT_TRAIN --host 0.0.0.0 \
      --tensor-parallel-size 8 --gpu-memory-utilization 0.85 \
      --max-model-len 65536 --dtype bfloat16 \
      --enable-lora --max-lora-rank 64 --max-loras 4 \
      --disable-log-requests \
      > /fsx/xuanj/ttt-discover/logs/vllm/aligned_vllm_train_${JOBID}.log 2>&1
  " &
PID_VLLM_TRAIN=$!

# vLLM #2: eval-side (will hot-reload LoRA after TTT done)
srun --nodes=1 --ntasks=1 --nodelist=$NODE_VLLM_EVAL --exclusive --output=/dev/null \
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
      --port $PORT_EVAL --host 0.0.0.0 \
      --tensor-parallel-size 8 --gpu-memory-utilization 0.85 \
      --max-model-len 65536 --dtype bfloat16 \
      --enable-lora --max-lora-rank 64 --max-loras 4 \
      --disable-log-requests \
      > /fsx/xuanj/ttt-discover/logs/vllm/aligned_vllm_eval_${JOBID}.log 2>&1
  " &
PID_VLLM_EVAL=$!

echo "[aligned] waiting for both vllms"
for url in $URL_TRAIN $URL_EVAL; do
  echo "[aligned] waiting $url"
  for i in $(seq 1 480); do
    curl -sf "${url}/health" > /dev/null 2>&1 && { echo "[aligned] $url up after $((i*5))s"; break; }
    sleep 5
  done
  if ! curl -sf "${url}/health" > /dev/null ; then
    echo "[aligned] FATAL $url did not start"
    kill $PID_VLLM_TRAIN $PID_VLLM_EVAL 2>/dev/null || true
    exit 2
  fi
done

# Trainer (uses vllm-train, 4-task pool, 16 steps, group_size=8 → 2 rollouts/task)
srun --nodes=1 --ntasks=1 --nodelist=$NODE_TRAIN --exclusive \
  bash -lc "
    set -euo pipefail
    source /fsx/xuanj/ttt-discover/.venv/bin/activate
    export HF_HOME=/fsx/xuanj/ttt-discover/.hf-cache
    export OMP_NUM_THREADS=4
    export TOKENIZERS_PARALLELISM=false
    export CUDA_VISIBLE_DEVICES=0
    python /fsx/xuanj/ttt-discover/scripts/ttt_iterative_4tasks.py \
      --tasks $TRAIN_TASKS \
      --vllm-url $URL_TRAIN \
      --lora-name $LORA_NAME_TRAIN \
      --run-tag $RUN_TAG \
      --num-steps 5 --group-size 8 \
      --lr 4e-5 --lora-rank 32 --kl-coef 0.1 \
      --temperature 0.9 --max-tokens-per-turn 16384 --max-turns 3 \
      --gen-timeout 1800 --eval-timeout 1800 --rollout-concurrency 8 \
      > /fsx/xuanj/ttt-discover/logs/slurm/ttt-iter-aligned-${RUN_TAG}-${JOBID}.log 2>&1
  " &
PID_TRAINER=$!

# Wait for trainer to finish
wait $PID_TRAINER
echo "[aligned] trainer finished, killing vllm-train"
kill $PID_VLLM_TRAIN 2>/dev/null || true
wait $PID_VLLM_TRAIN 2>/dev/null || true

# Find final LoRA
LORA_DIR_FINAL="${LORA_DIR_BASE}/final"
if [ ! -d "$LORA_DIR_FINAL" ]; then
  LORA_DIR_FINAL=$(ls -td ${LORA_DIR_BASE}/step_* 2>/dev/null | head -1)
fi
echo "[aligned] final LoRA: $LORA_DIR_FINAL"

# Load final LoRA into vllm-eval
curl -sf -X POST "$URL_EVAL/v1/unload_lora_adapter" -H "Content-Type: application/json" \
  -d "{\"lora_name\":\"$LORA_NAME_EVAL\"}" || true
sleep 2
curl -sf -X POST "$URL_EVAL/v1/load_lora_adapter" -H "Content-Type: application/json" \
  -d "{\"lora_name\":\"$LORA_NAME_EVAL\",\"lora_path\":\"$LORA_DIR_FINAL\"}"
echo "[aligned] loaded final LoRA into vllm-eval"

# Post-TTT eval: 4 shards on 4 nodes (vllm-eval, eval, trainer-now-free, vllm-train-now-free)
for shard in 0 1 2 3 ; do
  case $shard in
    0) SHARD_NODE=$NODE_EVAL ;;
    1) SHARD_NODE=$NODE_TRAIN ;;
    2) SHARD_NODE=$NODE_VLLM_TRAIN ;;
    3) SHARD_NODE=$NODE_VLLM_EVAL ;;
  esac
  echo "[aligned] dispatching post-eval shard $shard to $SHARD_NODE"
  srun --nodes=1 --ntasks=1 --nodelist=$SHARD_NODE --overlap \
    bash -lc "
      set -euo pipefail
      source /fsx/xuanj/ttt-discover/.venv/bin/activate
      export HF_HOME=/fsx/xuanj/ttt-discover/.hf-cache
      export OMP_NUM_THREADS=4
      python /fsx/xuanj/ttt-discover/scripts/post_ttt_eval.py \
        --server $URL_EVAL --model $LORA_NAME_EVAL \
        --shard-id $shard --num-shards 4 \
        --num-rollouts 8 \
        --max-turns 3 --max-tokens 16384 --temperature 0.9 \
        --concurrency 4 --gen-timeout 1800 --eval-timeout 1800 \
        > /fsx/xuanj/ttt-discover/logs/slurm/post-ttt-shard${shard}-${JOBID}.log 2>&1
    " &
done

wait
kill $PID_VLLM_EVAL 2>/dev/null || true
wait $PID_VLLM_EVAL 2>/dev/null || true
echo "[aligned] all done"
