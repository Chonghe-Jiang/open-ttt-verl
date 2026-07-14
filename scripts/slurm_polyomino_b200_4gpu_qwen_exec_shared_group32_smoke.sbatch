#!/usr/bin/env bash
#SBATCH --job-name=guidance-poly-qwenexec-4g-g32-smoke
#SBATCH --partition=b200-batch
#SBATCH --gres=gpu:b200:4
#SBATCH --cpus-per-task=96
#SBATCH --mem=768G
#SBATCH --time=12:00:00
#SBATCH --account=ppliang_mit
#SBATCH --output=outputs/slurm/polyomino-b200-4gpu-qwenexec-g32-smoke-%j.out
#SBATCH --error=outputs/slurm/polyomino-b200-4gpu-qwenexec-g32-smoke-%j.err

set -euo pipefail

REPO_ROOT="/work/mit/ppliang_mit/chonghej/open-ttt-verl-guidance-ttt-modal"
cd "${REPO_ROOT}"

COLOCATION_PROFILE="${COLOCATION_PROFILE:-primary}"
case "${COLOCATION_PROFILE}" in
  primary)
    export CONFIG="guidance_ttt/config/polyomino_b200_4gpu_qwen3_8b_exec_shared_batch8_group32_code_delta_8192_smoke.yaml"
    export EXECUTION_GPU_MEMORY_UTILIZATION=0.20
    export EXECUTION_MAX_MODEL_LEN=32768
    export EXECUTION_MAX_NUM_SEQS=4
    ;;
  fallback)
    export CONFIG="guidance_ttt/config/polyomino_b200_4gpu_qwen3_8b_exec_shared_batch8_group32_code_delta_8192_fallback_smoke.yaml"
    export EXECUTION_GPU_MEMORY_UTILIZATION=0.18
    export EXECUTION_MAX_MODEL_LEN=24576
    export EXECUTION_MAX_NUM_SEQS=2
    ;;
  *)
    echo "Unsupported COLOCATION_PROFILE=${COLOCATION_PROFILE}; expected primary or fallback" >&2
    exit 2
    ;;
esac

export OUTPUT_DIR="outputs/guidance_ttt/polyomino_b200_4gpu_qwen_exec_shared_g32_${COLOCATION_PROFILE}_smoke_${SLURM_JOB_ID}"
export EXPECTED_GPUS=4
export TRAINING_GPUS=0,1,2,3
export EXECUTION_GPU=3
export EXPECTED_GROUPS=8
export EXPECTED_GROUP_SIZE=32
export EXPECTED_PROMPT_MODE=code_delta

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "COLOCATION_PROFILE=${COLOCATION_PROFILE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

scripts/run_polyomino_b200_4gpu_qwen_exec_shared.sh preflight
scripts/run_polyomino_b200_4gpu_qwen_exec_shared.sh run
