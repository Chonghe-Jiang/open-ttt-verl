#!/usr/bin/env bash
#SBATCH --job-name=guidance-qwen14b-b200-5gpu-g24
#SBATCH --partition=b200-batch
#SBATCH --gres=gpu:b200:5
#SBATCH --cpus-per-task=96
#SBATCH --mem=768G
#SBATCH --time=08:00:00
#SBATCH --account=ppliang_mit
#SBATCH --output=outputs/slurm/qwen14b-b200-5gpu-g24-%j.out
#SBATCH --error=outputs/slurm/qwen14b-b200-5gpu-g24-%j.err

set -euo pipefail

REPO_ROOT="/work/mit/ppliang_mit/chonghej/open-ttt-verl-guidance-ttt-modal"
cd "${REPO_ROOT}"

export CONFIG="guidance_ttt/config/polyomino_b200_5gpu_qwen3_14b_gpt_oss_120b_batch8_group24_code_delta_8192_smoke.yaml"
export OUTPUT_DIR="outputs/guidance_ttt/polyomino_b200_5gpu_qwen3_14b_group24_smoke_${SLURM_JOB_ID}"
export EXPECTED_GPUS=5
export TRAINING_GPUS=0,1,2,3
export EXECUTION_GPU=4
export EXPECTED_GROUPS=8
export EXPECTED_GROUP_SIZE=24

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

test -f models/Qwen3-14B/.download-complete
scripts/run_polyomino_b200_3gpu_smoke.sh preflight
scripts/run_polyomino_b200_3gpu_smoke.sh run
