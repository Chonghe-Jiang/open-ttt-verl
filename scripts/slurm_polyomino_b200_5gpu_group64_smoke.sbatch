#!/usr/bin/env bash
#SBATCH --job-name=guidance-poly-b200-5gpu-g64-smoke
#SBATCH --partition=b200-batch
#SBATCH --gres=gpu:b200:5
#SBATCH --cpus-per-task=96
#SBATCH --mem=768G
#SBATCH --time=08:00:00
#SBATCH --account=ppliang_mit
#SBATCH --output=outputs/slurm/polyomino-b200-5gpu-g64-smoke-%j.out
#SBATCH --error=outputs/slurm/polyomino-b200-5gpu-g64-smoke-%j.err

set -euo pipefail

REPO_ROOT="/work/mit/ppliang_mit/chonghej/open-ttt-verl-guidance-ttt-modal"
cd "${REPO_ROOT}"

export CONFIG="guidance_ttt/config/polyomino_b200_5gpu_gpt_oss_120b_batch8_group64_code_delta_8192_smoke.yaml"
export OUTPUT_DIR="outputs/guidance_ttt/polyomino_b200_5gpu_group64_smoke_${SLURM_JOB_ID}"
export EXPECTED_GPUS=5
export TRAINING_GPUS=0,1,2,3
export EXECUTION_GPU=4
export EXPECTED_GROUPS=8
export EXPECTED_GROUP_SIZE=64

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

scripts/run_polyomino_b200_3gpu_smoke.sh preflight
scripts/run_polyomino_b200_3gpu_smoke.sh run
