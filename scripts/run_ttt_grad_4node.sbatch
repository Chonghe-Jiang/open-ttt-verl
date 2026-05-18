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
mkdir -p logs/slurm results/ttt_grad scratch

source /fsx/xuanj/ttt-discover/.venv/bin/activate
export HF_HOME=/fsx/xuanj/ttt-discover/.hf-cache
export FLASHINFER_DISABLE_VERSION_CHECK=1
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

echo "[ttt-grad-4n] master=$(hostname) nodelist=$SLURM_JOB_NODELIST"
NODES=$(scontrol show hostnames $SLURM_JOB_NODELIST)
NODE_ARR=($NODES)
echo "[ttt-grad-4n] node count: ${#NODE_ARR[@]}"

# Per user rule: allocate >=2 nodes (power of 2). We use shard 0 only for the PoC.
# Other 3 nodes idle but allocated. Bigger sweep can fan out later.
# Validation phase: only run shard 0 on cbl_multi__low (the only base task with
# >0 reward). Other 3 nodes idle but allocated (per quota rule). Once we confirm
# step 0 reproduces base signal, expand to a fan-out across more tasks.
TASK_LIST=(
  "cbl_multi__low_av_loose_dl_small_oh"
)

for shard in 0 ; do
  NODE=${NODE_ARR[$shard]}
  TASK=${TASK_LIST[$shard]}
  echo "[ttt-grad-4n] dispatching shard $shard to node $NODE on task $TASK"
  srun --nodes=1 --ntasks=1 --nodelist=$NODE --exclusive \
    bash -lc "
      set -euo pipefail
      source /fsx/xuanj/ttt-discover/.venv/bin/activate
      export HF_HOME=/fsx/xuanj/ttt-discover/.hf-cache
      export OMP_NUM_THREADS=4
      export TOKENIZERS_PARALLELISM=false
      export CUDA_VISIBLE_DEVICES=0
      python /fsx/xuanj/ttt-discover/scripts/ttt_discover_grad.py \
        --task $TASK \
        --num-steps 6 \
        --group-size 8 \
        --max-new-tokens 8192 \
        --lr 4e-5 \
        --lora-rank 32 \
        --beta 2.0 \
        --eval-concurrency 4 \
        --base-eval-dir /fsx/xuanj/ttt-discover/results/base/eval \
        --base-model-tag dsr1q3_8b_base \
        --output-dir /fsx/xuanj/ttt-discover/results/ttt_grad \
        > /fsx/xuanj/ttt-discover/logs/slurm/ttt-grad-${TASK}-${SLURM_JOB_ID}.log 2>&1 || echo \"shard $shard task $TASK failed\"
    " &
done
wait
echo "[ttt-grad-4n] done"
