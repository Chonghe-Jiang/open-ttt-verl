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
mkdir -p logs/slurm logs/vllm results/ttt_iter results/ttt_iter_lora results/ttt_iter_zeroshot/eval scratch_iter_train scratch_iter_zs

NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
NODE_ARR=($NODES)
NODE_VLLM_TRAIN=${NODE_ARR[0]}
NODE_TRAIN=${NODE_ARR[1]}
NODE_VLLM_ZS=${NODE_ARR[2]}
NODE_ZS=${NODE_ARR[3]}

TRAIN_TASK="${TRAIN_TASK:-cbl__low_av_tight_dl_large_oh}"
JOBID=${SLURM_JOB_ID:-local}
PORT_TRAIN=8001
PORT_ZS=8002
URL_TRAIN="http://${NODE_VLLM_TRAIN}:${PORT_TRAIN}"
URL_ZS="http://${NODE_VLLM_ZS}:${PORT_ZS}"
LORA_NAME_TRAIN="ttt-iter"
LORA_NAME_ZS="ttt-iter-zs"
LORA_DIR_BASE="/fsx/xuanj/ttt-discover/results/ttt_iter_lora/${TRAIN_TASK}"

echo "[mixed] vllm_train=$NODE_VLLM_TRAIN train=$NODE_TRAIN vllm_zs=$NODE_VLLM_ZS zs=$NODE_ZS"

# vLLM #1 — for trainer rollouts (LoRA "ttt-iter")
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
      > /fsx/xuanj/ttt-discover/logs/vllm/mixed_vllm_train_${JOBID}.log 2>&1
  " &
PID_VLLM_TRAIN=$!

# vLLM #2 — for zero-shot eval (will reload LoRA "ttt-iter-zs" each new step)
srun --nodes=1 --ntasks=1 --nodelist=$NODE_VLLM_ZS --exclusive --output=/dev/null \
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
      --port $PORT_ZS --host 0.0.0.0 \
      --tensor-parallel-size 8 --gpu-memory-utilization 0.85 \
      --max-model-len 65536 --dtype bfloat16 \
      --enable-lora --max-lora-rank 64 --max-loras 4 \
      --disable-log-requests \
      > /fsx/xuanj/ttt-discover/logs/vllm/mixed_vllm_zs_${JOBID}.log 2>&1
  " &
PID_VLLM_ZS=$!

# Wait for both vLLMs
echo "[mixed] waiting for vllm-train at $URL_TRAIN"
for i in $(seq 1 360); do
  curl -sf "${URL_TRAIN}/health" > /dev/null 2>&1 && { echo "[mixed] vllm-train up after $((i*5))s"; break; }
  sleep 5
done
echo "[mixed] waiting for vllm-zs at $URL_ZS"
for i in $(seq 1 360); do
  curl -sf "${URL_ZS}/health" > /dev/null 2>&1 && { echo "[mixed] vllm-zs up after $((i*5))s"; break; }
  sleep 5
done
if ! curl -sf "${URL_TRAIN}/health" > /dev/null || ! curl -sf "${URL_ZS}/health" > /dev/null ; then
  echo "[mixed] FATAL one of vllm did not start"
  kill $PID_VLLM_TRAIN $PID_VLLM_ZS 2>/dev/null || true
  exit 2
fi

# Trainer (uses vllm-train)
srun --nodes=1 --ntasks=1 --nodelist=$NODE_TRAIN --exclusive \
  bash -lc "
    set -euo pipefail
    source /fsx/xuanj/ttt-discover/.venv/bin/activate
    export HF_HOME=/fsx/xuanj/ttt-discover/.hf-cache
    export OMP_NUM_THREADS=4
    export TOKENIZERS_PARALLELISM=false
    export CUDA_VISIBLE_DEVICES=0
    python /fsx/xuanj/ttt-discover/scripts/ttt_iterative.py \
      --task $TRAIN_TASK --vllm-url $URL_TRAIN \
      --num-steps 5 --group-size 8 \
      --lr 4e-5 --lora-rank 32 --kl-coef 0.1 \
      --temperature 0.9 --max-tokens-per-turn 16384 --max-turns 3 \
      --gen-timeout 1800 --eval-timeout 1800 --rollout-concurrency 4 \
      > /fsx/xuanj/ttt-discover/logs/slurm/ttt-iter-train-${TRAIN_TASK}-${JOBID}.log 2>&1
  " &
PID_TRAINER=$!

# ZS daemon: watches LORA_DIR_BASE/step_*, runs eval rollouts on each new step
srun --nodes=1 --ntasks=1 --nodelist=$NODE_ZS --exclusive \
  bash -lc "
    set -euo pipefail
    source /fsx/xuanj/ttt-discover/.venv/bin/activate
    export HF_HOME=/fsx/xuanj/ttt-discover/.hf-cache
    export OMP_NUM_THREADS=4
    python /fsx/xuanj/ttt-discover/scripts/concurrent_zs_eval.py \
      --vllm-url $URL_ZS --lora-name $LORA_NAME_ZS \
      --lora-base-dir $LORA_DIR_BASE --train-task $TRAIN_TASK \
      --output-dir /fsx/xuanj/ttt-discover/results/ttt_iter_zeroshot/eval \
      --rollouts-per-step 1 --final-rollouts 8 \
      > /fsx/xuanj/ttt-discover/logs/slurm/concurrent-zs-${JOBID}.log 2>&1
  " &
PID_ZS=$!

# Wait trainer to finish
wait $PID_TRAINER
echo "[mixed] trainer finished, killing vllm-train (free node $NODE_VLLM_TRAIN, $NODE_TRAIN)"
kill $PID_VLLM_TRAIN 2>/dev/null || true
wait $PID_VLLM_TRAIN 2>/dev/null || true

# Signal ZS daemon to do final pass and exit
touch /fsx/xuanj/ttt-discover/results/ttt_iter_lora/${TRAIN_TASK}/.training_done
echo "[mixed] training_done sentinel written"

# Spawn 3 more zs-shard srun's on the 3 freed nodes ($NODE_VLLM_TRAIN + $NODE_TRAIN are now idle)
# Each shard only handles a slice of tasks×rollouts
# Existing ZS daemon (NODE_ZS) is shard 0/4 — its main loop still running step eval polling.
# We need to wait for ZS daemon to enter "final" mode (sentinel detected) — easier to just
# restart the ZS evaluation from scratch as a sharded final pass.

# Wait for ZS daemon to read sentinel and enter final mode (it polls every 30s)
sleep 35

# Kill the daemon — we'll do 4-shard parallel final eval instead (faster)
kill $PID_ZS 2>/dev/null || true
wait $PID_ZS 2>/dev/null || true
echo "[mixed] step-eval daemon stopped, launching 4-shard parallel final ZS eval"

# Now do parallel final ZS eval across all 4 nodes
# vllm-zs (NODE_VLLM_ZS) still running with LoRA loaded; reuse it as the single vLLM
# All 4 nodes (incl trainer node, vllm-train node) become eval shards
LORA_DIR_FINAL="/fsx/xuanj/ttt-discover/results/ttt_iter_lora/${TRAIN_TASK}/final"
if [ ! -d "$LORA_DIR_FINAL" ]; then
  LORA_DIR_FINAL=$(ls -td /fsx/xuanj/ttt-discover/results/ttt_iter_lora/${TRAIN_TASK}/step_* 2>/dev/null | head -1)
fi
echo "[mixed] final LoRA: $LORA_DIR_FINAL"

# Make sure the LoRA name is loaded fresh
curl -sf -X POST "$URL_ZS/v1/unload_lora_adapter" -H "Content-Type: application/json" \
  -d "{\"lora_name\":\"$LORA_NAME_ZS\"}" || true
sleep 2
curl -sf -X POST "$URL_ZS/v1/load_lora_adapter" -H "Content-Type: application/json" \
  -d "{\"lora_name\":\"$LORA_NAME_ZS\",\"lora_path\":\"$LORA_DIR_FINAL\"}"
echo "[mixed] reloaded final LoRA"

# Launch 4 shards on the 4 nodes
for shard in 0 1 2 3 ; do
  case $shard in
    0) SHARD_NODE=$NODE_ZS ;;
    1) SHARD_NODE=$NODE_TRAIN ;;
    2) SHARD_NODE=$NODE_VLLM_TRAIN ;;
    3) SHARD_NODE=$NODE_VLLM_ZS ;;
  esac
  echo "[mixed] dispatching final-zs shard $shard to $SHARD_NODE"
  srun --nodes=1 --ntasks=1 --nodelist=$SHARD_NODE --overlap \
    bash -lc "
      set -euo pipefail
      source /fsx/xuanj/ttt-discover/.venv/bin/activate
      export HF_HOME=/fsx/xuanj/ttt-discover/.hf-cache
      export OMP_NUM_THREADS=4
      python /fsx/xuanj/ttt-discover/scripts/zeroshot_eval_ttt.py \
        --server $URL_ZS --model $LORA_NAME_ZS \
        --exclude-task $TRAIN_TASK \
        --shard-id $shard --num-shards 4 \
        --num-rollouts 8 \
        --max-turns 3 --max-tokens 16384 --temperature 0.9 \
        --concurrency 4 --gen-timeout 1800 --eval-timeout 1800 \
        > /fsx/xuanj/ttt-discover/logs/slurm/final-zs-shard${shard}-${JOBID}.log 2>&1
    " &
done

wait
kill $PID_VLLM_ZS 2>/dev/null || true
wait $PID_VLLM_ZS 2>/dev/null || true
echo "[mixed] all done"
