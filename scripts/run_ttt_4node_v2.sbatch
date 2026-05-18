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
mkdir -p logs/slurm logs/vllm results/ttt scratch

echo "[ttt-4n] master=$(hostname) nodelist=$SLURM_JOB_NODELIST"

# We allocated 4 nodes, but use only 4 distinct ones via --nodelist explicit
# Get the unique node list to avoid --relative collisions
NODES=$(scontrol show hostnames $SLURM_JOB_NODELIST)
echo "[ttt-4n] expanded nodes: $NODES"
NODE_ARR=($NODES)
echo "[ttt-4n] node count: ${#NODE_ARR[@]}"

for shard in 0 1 2 3 ; do
  NODE=${NODE_ARR[$shard]}
  PORT=$((8000 + shard))
  echo "[ttt-4n] dispatching shard $shard to node $NODE on port $PORT"
  srun --nodes=1 --ntasks=1 --nodelist=$NODE --exclusive \
    bash /fsx/xuanj/ttt-discover/scripts/ttt_shard.sh $shard 4 $PORT &
done
wait
echo "[ttt-4n] all shards done"
