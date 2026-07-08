#!/bin/bash
#SBATCH --job-name=poly-gptoss120b-g16
#SBATCH -p mit_preemptable
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:3
#SBATCH --time=24:00:00
#SBATCH -c 64
#SBATCH --mem=256G
#SBATCH --account=mit_general
#SBATCH --qos=normal
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

set -euo pipefail

cd /home/qua/code/guidance-ttt

if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi
module load apptainer/1.4.2 >/dev/null 2>&1 || true

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

export APPTAINER_IMAGE="${APPTAINER_IMAGE:-/orcd/scratch/orcd/010/dwai/vllm.sif}"
export CONFIG="${CONFIG:-guidance_ttt/config/polyomino_modal_h200_3gpu_gpt_oss_120b_group16_h200_tuned.yaml}"
export HOST_FRONTIERCS_DIR="${HOST_FRONTIERCS_DIR:-/home/qua/code/reference/Frontier-CS}"
export TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1}"
export EXECUTION_CUDA_VISIBLE_DEVICES="${EXECUTION_CUDA_VISIBLE_DEVICES:-2}"
export RESET_OUTPUT_DIR="${RESET_OUTPUT_DIR:-1}"

echo "CONFIG=$CONFIG"
echo "APPTAINER_IMAGE=$APPTAINER_IMAGE"
echo "HOST_RUNS_DIR=${HOST_RUNS_DIR:-$PWD/.modal_local_runs}"
echo "HOST_CACHE_DIR=${HOST_CACHE_DIR:-$PWD/.modal_local_cache}"
echo "TRAIN_CUDA_VISIBLE_DEVICES=$TRAIN_CUDA_VISIBLE_DEVICES"
echo "EXECUTION_CUDA_VISIBLE_DEVICES=$EXECUTION_CUDA_VISIBLE_DEVICES"

scripts/run_local_polyomino_h200_gpt_oss_120b_group16.sh "$@"
