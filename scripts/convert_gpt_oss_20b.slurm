#!/bin/bash
#SBATCH -p pi_ppliang
#SBATCH --gres=gpu:2
#SBATCH -c 30
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH -J open-ttt-convert-gpt-oss-20b
#SBATCH -o /work/mit/ppliang_mit/lsy/open-ttt-workspace/logs/slurm-convert-gpt-oss-20b-%j.out

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

export NPROC=2
export WORKSPACE=${WORKSPACE:-/work/mit/ppliang_mit/lsy/open-ttt-workspace}
export TENSOR_MODEL_PARALLEL_SIZE=1
export PIPELINE_MODEL_PARALLEL_SIZE=2
export EXPERT_MODEL_PARALLEL_SIZE=1
export EXPERT_TENSOR_PARALLEL_SIZE=1

echo "=== slurm convert gpt-oss-20b ==="
echo "time: $(date -Is)"
echo "host: $(hostname)"
echo "job: ${SLURM_JOB_ID:-unset}"
echo "partition: ${SLURM_JOB_PARTITION:-unset}"
echo "gres: ${SLURM_JOB_GRES:-unset}"
echo "gpus_on_node: ${SLURM_GPUS_ON_NODE:-unset}"
echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "nproc: ${NPROC}"
echo "tp: ${TENSOR_MODEL_PARALLEL_SIZE}"
echo "pp: ${PIPELINE_MODEL_PARALLEL_SIZE}"
echo "ep: ${EXPERT_MODEL_PARALLEL_SIZE}"
echo "etp: ${EXPERT_TENSOR_PARALLEL_SIZE}"

bash scripts/ondemand_convert_gpt_oss_20b_apptainer_job.sh
