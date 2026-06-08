#!/bin/bash
#SBATCH -p pi_ppliang
#SBATCH --gres=gpu:4
#SBATCH -c 60
#SBATCH --mem=240G
#SBATCH --time=08:00:00
#SBATCH -J open-ttt-qwen3-8b-convert-4gpu
#SBATCH -o /home/qua/scratch/open-ttt-workspace/logs/slurm-qwen3-8b-convert-%j.out

set -Eeuo pipefail

cd /home/qua/code/open-ttt

export WORKSPACE=${WORKSPACE:-/home/qua/scratch/open-ttt-workspace}
export MODEL_ID=${MODEL_ID:-Qwen/Qwen3-8B}
export MODEL_DIR=${MODEL_DIR:-${WORKSPACE}/models/Qwen3-8B}
export MODEL_RAW=${MODEL_RAW:-${MODEL_DIR}}
export CKPT_OUT=${CKPT_OUT:-${WORKSPACE}/ckpt/Qwen3-8B_torch_dist}
export NPROC=4
export TENSOR_MODEL_PARALLEL_SIZE=1
export PIPELINE_MODEL_PARALLEL_SIZE=4
export EXPERT_MODEL_PARALLEL_SIZE=1
export EXPERT_TENSOR_PARALLEL_SIZE=1

mkdir -p "${WORKSPACE}/logs" "${WORKSPACE}/models" "${WORKSPACE}/ckpt"

echo "=== download and convert Qwen3-8B, 4 GPU ==="
echo "time: $(date -Is)"
echo "host: $(hostname)"
echo "job: ${SLURM_JOB_ID:-unset}"
echo "partition: ${SLURM_JOB_PARTITION:-unset}"
echo "gres: ${SLURM_JOB_GRES:-unset}"
echo "gpus_on_node: ${SLURM_GPUS_ON_NODE:-unset}"
echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "workspace: ${WORKSPACE}"
echo "model_id: ${MODEL_ID}"
echo "model_dir: ${MODEL_DIR}"
echo "ckpt_out: ${CKPT_OUT}"
echo "nproc: ${NPROC}"
echo "tp: ${TENSOR_MODEL_PARALLEL_SIZE}"
echo "pp: ${PIPELINE_MODEL_PARALLEL_SIZE}"
echo "ep: ${EXPERT_MODEL_PARALLEL_SIZE}"
echo "etp: ${EXPERT_TENSOR_PARALLEL_SIZE}"

bash scripts/download_qwen3_8b.sh
bash scripts/convert_qwen3_8b_with_apptainer.sh
