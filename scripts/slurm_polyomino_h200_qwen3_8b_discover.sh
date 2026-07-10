#!/bin/bash
#SBATCH --job-name=poly-qwen-disc
#SBATCH -p mit_preemptable
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:4
#SBATCH --time=48:00:00
#SBATCH -c 64
#SBATCH --mem=512G
#SBATCH --account=mit_general
#SBATCH --qos=normal
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

set -Eeuo pipefail

cd /home/qua/code/guidance-ttt

if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi
module load apptainer/1.4.2 >/dev/null 2>&1 || true

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "RUN_STAGE=${RUN_STAGE:-full}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

export PYTHON_BIN="${PYTHON_BIN:-python3.12}"
export CONTAINER_PYTHON_BIN="${CONTAINER_PYTHON_BIN:-python3}"
export APPTAINER_IMAGE="${APPTAINER_IMAGE:-/orcd/scratch/orcd/010/dwai/vllm.sif}"
export CONFIG="${CONFIG:-guidance_ttt/config/polyomino_h200_4gpu_qwen3_8b_discover.yaml}"
export FRONTIERCS_DIR="${FRONTIERCS_DIR:-/home/qua/code/reference/Frontier-CS}"
export JUDGE_WORKERS="${JUDGE_WORKERS:-8}"
export RESET_OUTPUT_DIR="${RESET_OUTPUT_DIR:-1}"

echo "CONFIG=$CONFIG"
echo "APPTAINER_IMAGE=$APPTAINER_IMAGE"
echo "FRONTIERCS_DIR=$FRONTIERCS_DIR"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "CONTAINER_PYTHON_BIN=$CONTAINER_PYTHON_BIN"
echo "JUDGE_WORKERS=$JUDGE_WORKERS"
echo "RESET_OUTPUT_DIR=$RESET_OUTPUT_DIR"

scripts/run_local_polyomino_h200_qwen3_8b_discover_stage.sh "$@"
