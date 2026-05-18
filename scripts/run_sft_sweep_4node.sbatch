#!/usr/bin/env bash
#SBATCH --job-name=no-shell
#SBATCH --partition=p5en-odcr-queue
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=08:00:00
#SBATCH --output=/fsx/xuanj/ttt-discover/logs/slurm/%x-%j.out
#SBATCH --error=/fsx/xuanj/ttt-discover/logs/slurm/%x-%j.err
set -euo pipefail

cd /fsx/xuanj/ttt-discover
mkdir -p logs/slurm logs/vllm results/sft_lora results/sft_eval scratch

echo "[sft-sweep-4n] master=$(hostname) nodelist=$SLURM_JOB_NODELIST"
NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
NODE_ARR=($NODES)
echo "[sft-sweep-4n] node count: ${#NODE_ARR[@]}"

for shard in 0 1 2 3 ; do
  NODE=${NODE_ARR[$shard]}
  PORT=$((8000 + shard))
  echo "[sft-sweep-4n] dispatching shard $shard to node $NODE on port $PORT"
  srun --nodes=1 --ntasks=1 --nodelist=$NODE --exclusive \
    bash /fsx/xuanj/ttt-discover/scripts/sft_sweep_shard.sh $shard 4 $PORT &
done
wait
echo "[sft-sweep-4n] all shards done"
