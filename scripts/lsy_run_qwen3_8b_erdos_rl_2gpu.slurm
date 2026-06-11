#!/bin/bash
#SBATCH -p b200-batch
#SBATCH --gres=gpu:2
#SBATCH -c 60
#SBATCH --mem=480G
#SBATCH --time=24:00:00
#SBATCH -J open-ttt-qwen3-8b-erdos-rl-2gpu
#SBATCH -o /work/mit/ppliang_mit/lsy/open-ttt-workspace/logs/slurm-qwen3-8b-erdos-rl-2gpu-%j.out

set -Eeuo pipefail

cd /work/mit/ppliang_mit/lsy/open-ttt-verl-slime-ttt
source scripts/lsy_apptainer_env.sh

export TOTAL_GPUS=2
export NUM_GPUS_PER_NODE=2
export ACTOR_NUM_GPUS=2
export ROLLOUT_NUM_GPUS=2
export ROLLOUT_NUM_GPUS_PER_ENGINE=1
export COLOCATE=1

export TENSOR_MODEL_PARALLEL_SIZE=1
export PIPELINE_MODEL_PARALLEL_SIZE=2
export CONTEXT_PARALLEL_SIZE=1
export EXPERT_MODEL_PARALLEL_SIZE=1
export EXPERT_TENSOR_PARALLEL_SIZE=1

export NUM_ROLLOUT=${NUM_ROLLOUT:-50}
export ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-1}
export N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-8}
export LORA_RANK=${LORA_RANK:-32}
export LORA_ALPHA=${LORA_ALPHA:-32}
export SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.35}
export SGLANG_CONTEXT_LENGTH=${SGLANG_CONTEXT_LENGTH:-16384}
export MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-8192}
export KILL_STALE_PROCESSES=${KILL_STALE_PROCESSES:-0}

bash scripts/run_qwen3_8b_erdos_rl_with_apptainer.sh
