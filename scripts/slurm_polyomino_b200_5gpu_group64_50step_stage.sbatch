#!/usr/bin/env bash
#SBATCH --job-name=guidance-poly-b200-g64-50s
#SBATCH --partition=b200-batch
#SBATCH --gres=gpu:b200:5
#SBATCH --cpus-per-task=96
#SBATCH --mem=768G
#SBATCH --time=1-00:00:00
#SBATCH --account=ppliang_mit
#SBATCH --output=outputs/slurm/polyomino-b200-5gpu-g64-50step-%j.out
#SBATCH --error=outputs/slurm/polyomino-b200-5gpu-g64-50step-%j.err

set -euo pipefail

REPO_ROOT="/work/mit/ppliang_mit/chonghej/open-ttt-verl-guidance-ttt-modal"
cd "${REPO_ROOT}"

: "${SHARED_OUTPUT_DIR:?submit with SHARED_OUTPUT_DIR for checkpoint continuity}"
STAGE="${STAGE:-final}"
export CONFIG="guidance_ttt/config/polyomino_b200_5gpu_gpt_oss_120b_batch8_group64_code_delta_8192_50step.yaml"
export OUTPUT_DIR="${SHARED_OUTPUT_DIR}"
export EXPECTED_GPUS=5
export TRAINING_GPUS=0,1,2,3
export EXECUTION_GPU=4
export EXPECTED_GROUP_SIZE=64

case "${STAGE}" in
  first)
    export NUM_STEPS_OVERRIDE=20
    export TOTAL_EPOCHS_OVERRIDE=20
    export EXPECTED_GROUPS=160
    ;;
  final)
    unset NUM_STEPS_OVERRIDE TOTAL_EPOCHS_OVERRIDE
    export EXPECTED_GROUPS=400
    ;;
  *)
    echo "Unknown STAGE=${STAGE}; expected first or final" >&2
    exit 2
    ;;
esac

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "STAGE=${STAGE} OUTPUT_DIR=${OUTPUT_DIR}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

scripts/run_polyomino_b200_long.sh preflight
scripts/run_polyomino_b200_long.sh run
