#!/usr/bin/env bash
#SBATCH --job-name=no-shell
#SBATCH --partition=p5en-odcr-queue
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=02:00:00
#SBATCH --output=/fsx/xuanj/ttt-discover/logs/slurm/%x-%j.out
#SBATCH --error=/fsx/xuanj/ttt-discover/logs/slurm/%x-%j.err
set -euo pipefail

cd /fsx/xuanj/ttt-discover
NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
NODE_ARR=($NODES)
NODE0=${NODE_ARR[0]}
echo "[shard3-only] running on $NODE0"

srun --nodes=1 --ntasks=1 --nodelist=$NODE0 --exclusive \
  bash /fsx/xuanj/ttt-discover/scripts/sft_sweep_shard3_only.sh 8000

echo "[shard3-only] done"
