#!/bin/bash
#SBATCH -p b200-batch
#SBATCH --gres=gpu:2
#SBATCH -c 30
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH -J open-ttt-qwen3-8b-convert-2gpu
#SBATCH -o /work/mit/ppliang_mit/lsy/open-ttt-workspace/logs/slurm-qwen3-8b-convert-%j.out

set -Eeuo pipefail

cd /work/mit/ppliang_mit/lsy/open-ttt-verl-slime-ttt
source scripts/lsy_apptainer_env.sh

export NPROC=2
export TENSOR_MODEL_PARALLEL_SIZE=1
export PIPELINE_MODEL_PARALLEL_SIZE=2
export EXPERT_MODEL_PARALLEL_SIZE=1
export EXPERT_TENSOR_PARALLEL_SIZE=1

bash scripts/download_qwen3_8b.sh
bash scripts/convert_qwen3_8b_with_apptainer.sh
