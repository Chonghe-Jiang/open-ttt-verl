#!/usr/bin/env bash
#SBATCH --job-name=no-shell
#SBATCH --partition=p5en-odcr-queue
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00
#SBATCH --output=/fsx/xuanj/ttt-discover/logs/slurm/%x-%j.out
#SBATCH --error=/fsx/xuanj/ttt-discover/logs/slurm/%x-%j.err
set -euo pipefail

cd /fsx/xuanj/ttt-discover
mkdir -p logs/slurm logs/vllm results/sft_lora results/sft_eval scratch

source /fsx/xuanj/ttt-discover/.venv/bin/activate
export HF_HOME=/fsx/xuanj/ttt-discover/.hf-cache
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
NODE_ARR=($NODES)
TASK="cbl_multi__low_av_loose_dl_small_oh"
NODE0=${NODE_ARR[0]}

echo "[sft-validate] master=$(hostname) nodes=${NODE_ARR[@]}"
echo "[sft-validate] task=$TASK using node=$NODE0"

# Step 1: SFT pretrain on shard 0
srun --nodes=1 --ntasks=1 --nodelist=$NODE0 --exclusive \
  bash -lc "
    set -euo pipefail
    source /fsx/xuanj/ttt-discover/.venv/bin/activate
    export HF_HOME=/fsx/xuanj/ttt-discover/.hf-cache
    export OMP_NUM_THREADS=4
    export TOKENIZERS_PARALLELISM=false
    export CUDA_VISIBLE_DEVICES=0
    python /fsx/xuanj/ttt-discover/scripts/sft_pretrain.py \
      --task $TASK \
      --epochs 8 \
      --lr 1e-4 \
      --lora-rank 32 \
      --output-dir /fsx/xuanj/ttt-discover/results/sft_lora \
      > /fsx/xuanj/ttt-discover/logs/slurm/sft-pretrain-${TASK}-${SLURM_JOB_ID}.log 2>&1
  "

LORA_PATH=/fsx/xuanj/ttt-discover/results/sft_lora/${TASK}
if [ ! -d "$LORA_PATH" ]; then
  echo "[sft-validate] SFT failed to produce LoRA at $LORA_PATH"
  exit 1
fi

# Step 2: Launch vLLM on shard 0 with the LoRA adapter loaded
srun --nodes=1 --ntasks=1 --nodelist=$NODE0 --exclusive \
  bash -lc "
    set -euo pipefail
    source /fsx/xuanj/ttt-discover/.venv/bin/activate
    export HF_HOME=/fsx/xuanj/ttt-discover/.hf-cache
    export FLASHINFER_DISABLE_VERSION_CHECK=1
    export VLLM_ATTENTION_BACKEND=FLASH_ATTN
    export OMP_NUM_THREADS=4
    export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    PORT=8000

    nohup python -m vllm.entrypoints.openai.api_server \
      --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
      --port \$PORT --host 127.0.0.1 \
      --tensor-parallel-size 8 \
      --gpu-memory-utilization 0.85 \
      --max-model-len 32768 \
      --dtype bfloat16 \
      --enable-lora \
      --max-lora-rank 64 \
      --lora-modules sft-${TASK}=$LORA_PATH \
      --disable-log-requests \
      > /fsx/xuanj/ttt-discover/logs/vllm/sft_vllm_\${SLURM_JOB_ID}.log 2>&1 &
    VLLM_PID=\$!

    for i in \$(seq 1 240); do
      if curl -sf \"http://127.0.0.1:\$PORT/health\" > /dev/null 2>&1; then
        echo \"[sft-validate] vllm up after \$((i*5))s\"
        break
      fi
      sleep 5
    done
    if ! curl -sf \"http://127.0.0.1:\$PORT/health\" > /dev/null ; then
      echo \"[sft-validate] FATAL vllm did not start\"
      kill \$VLLM_PID 2>/dev/null || true
      exit 2
    fi

    python /fsx/xuanj/ttt-discover/scripts/sft_resample.py \
      --task $TASK \
      --vllm-url http://127.0.0.1:\$PORT \
      --lora-name sft-${TASK} \
      --num-rollouts 8 \
      --max-tokens 8192 \
      --gen-timeout 1800 \
      --eval-concurrency 8 \
      > /fsx/xuanj/ttt-discover/logs/slurm/sft-resample-${TASK}-${SLURM_JOB_ID}.log 2>&1

    kill \$VLLM_PID 2>/dev/null || true
    wait \$VLLM_PID 2>/dev/null || true
  "

echo "[sft-validate] done"
