#!/bin/bash
#SBATCH --job-name=poly-one-30k
#SBATCH -p mit_preemptable
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:4
#SBATCH --time=06:00:00
#SBATCH -c 64
#SBATCH --mem=500G
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

export START_FRONTIER_JUDGE="${START_FRONTIER_JUDGE:-1}"
export CONFIG="${CONFIG:-guidance_ttt/config/polyomino_h200_4gpu_single_summary.yaml}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/guidance_ttt/polyomino_h200_4gpu_one_step_30k_summary}"
export FRONTIERCS_DIR="${FRONTIERCS_DIR:-/home/qua/code/reference/Frontier-CS}"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
export EXECUTION_MODEL="${EXECUTION_MODEL:-claude-sub2api-opus-4-8}"
export APPTAINER_IMAGE="${APPTAINER_IMAGE:-/orcd/scratch/orcd/010/dwai/vllm.sif}"
export CONTAINER_CC="${CONTAINER_CC:-/usr/bin/gcc}"
export CONTAINER_CXX="${CONTAINER_CXX:-/usr/bin/g++}"
export ALLOW_FALLBACK_SUMMARY="${ALLOW_FALLBACK_SUMMARY:-0}"

echo "CONFIG=$CONFIG"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "EXECUTION_MODEL=$EXECUTION_MODEL"
echo "APPTAINER_IMAGE=$APPTAINER_IMAGE"
echo "CONTAINER_RUNTIME_BIN=${CONTAINER_RUNTIME_BIN:-auto}"
echo "CONTAINER_CC=$CONTAINER_CC"
echo "CONTAINER_CXX=$CONTAINER_CXX"
echo "ALLOW_FALLBACK_SUMMARY=$ALLOW_FALLBACK_SUMMARY"

scripts/run_local_polyomino_h200_single_summary.sh "$@"
