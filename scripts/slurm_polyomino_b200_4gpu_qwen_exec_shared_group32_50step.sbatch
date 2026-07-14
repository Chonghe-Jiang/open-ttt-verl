#!/usr/bin/env bash
#SBATCH --job-name=guidance-poly-qwenexec-4g-g32-50s
#SBATCH --partition=b200-batch
#SBATCH --gres=gpu:b200:4
#SBATCH --cpus-per-task=96
#SBATCH --mem=768G
#SBATCH --time=1-00:00:00
#SBATCH --account=ppliang_mit
#SBATCH --output=outputs/slurm/polyomino-b200-4gpu-qwenexec-g32-50step-%j.out
#SBATCH --error=outputs/slurm/polyomino-b200-4gpu-qwenexec-g32-50step-%j.err

set -euo pipefail

REPO_ROOT="/work/mit/ppliang_mit/chonghej/open-ttt-verl-guidance-ttt-modal"
cd "${REPO_ROOT}"

export CONFIG="guidance_ttt/config/polyomino_b200_4gpu_qwen3_8b_exec_shared_batch8_group32_code_delta_8192_50step.yaml"
export OUTPUT_DIR="outputs/guidance_ttt/polyomino_b200_4gpu_qwen_exec_shared_g32_50step_${SLURM_JOB_ID}"
export EXPECTED_GPUS=4
export TRAINING_GPUS=0,1,2,3
export EXECUTION_GPU=3
export EXECUTION_GPU_MEMORY_UTILIZATION=0.20
export EXECUTION_MAX_MODEL_LEN=40960
export EXECUTION_MAX_NUM_SEQS=4
export EXPECTED_GROUPS=400
export EXPECTED_GROUP_SIZE=32
export EXPECTED_PROMPT_MODE=code_delta

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

scripts/run_polyomino_b200_4gpu_qwen_exec_shared.sh preflight
scripts/run_polyomino_b200_4gpu_qwen_exec_shared.sh run
