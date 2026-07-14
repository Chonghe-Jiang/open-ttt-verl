#!/usr/bin/env bash
#SBATCH --job-name=guidance-poly-b200-3gpu-smoke
#SBATCH --partition=b200-batch
#SBATCH --gres=gpu:b200:3
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=08:00:00
#SBATCH --account=ppliang_mit
#SBATCH --output=outputs/slurm/polyomino-b200-3gpu-smoke-%j.out
#SBATCH --error=outputs/slurm/polyomino-b200-3gpu-smoke-%j.err

set -euo pipefail

REPO_ROOT="/work/mit/ppliang_mit/chonghej/open-ttt-verl-guidance-ttt-modal"
cd "${REPO_ROOT}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

scripts/run_polyomino_b200_3gpu_smoke.sh preflight
scripts/run_polyomino_b200_3gpu_smoke.sh run
