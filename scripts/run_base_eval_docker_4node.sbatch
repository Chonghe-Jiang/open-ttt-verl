#!/usr/bin/env bash
#SBATCH --job-name=ttt-base-eval-4n
#SBATCH --partition=p5en-odcr-queue
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=06:00:00
#SBATCH --output=/fsx/xuanj/ttt-discover/logs/slurm/%x-%j.out
#SBATCH --error=/fsx/xuanj/ttt-discover/logs/slurm/%x-%j.err
set -euo pipefail

cd /fsx/xuanj/ttt-discover
mkdir -p logs/slurm results/base/eval

source /fsx/xuanj/ttt-discover/.venv/bin/activate
export HF_HOME=/fsx/xuanj/ttt-discover/.hf-cache

# Strategy: split the 144 evaluations into 4 shards by task index.
# Each node runs its own shard via srun --nodes=1 --ntasks=1.
# Shard 0..3 each handles ~36 evals; concurrency=8 within each shard for full GPU use.
echo "[base_eval_docker_4n] master=$(hostname) total_nodes=$SLURM_NNODES"
echo "[base_eval_docker_4n] node list: $SLURM_JOB_NODELIST"

# Launch a shard on each node. Use srun --relative for per-node task mapping.
# We pass --shard-id and --num-shards into the python script so it picks its slice.
for shard in 0 1 2 3 ; do
  srun --nodes=1 --ntasks=1 --relative=$shard --exclusive \
    bash -c "
      source /fsx/xuanj/ttt-discover/.venv/bin/activate
      export HF_HOME=/fsx/xuanj/ttt-discover/.hf-cache
      echo \"[shard $shard] node=\$(hostname)\"
      python /fsx/xuanj/ttt-discover/scripts/eval_solutions.py \
        --frontier-root /fsx/xuanj/ttt-discover/src/frontier-cs \
        --tasks-json /fsx/xuanj/ttt-discover/bench/tasks_19.json \
        --model-tag dsr1q3_8b_base \
        --num-rollouts 8 \
        --concurrency 8 \
        --timeout 2400 \
        --shard-id $shard --num-shards 4 \
        --results-dir /fsx/xuanj/ttt-discover/results/base/eval \
        > /fsx/xuanj/ttt-discover/logs/slurm/ttt-base-eval-4n-$SLURM_JOB_ID.shard${shard}.log 2>&1
    " &
done
wait
echo "[base_eval_docker_4n] all 4 shards done"
