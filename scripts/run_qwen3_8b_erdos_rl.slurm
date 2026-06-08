#!/bin/bash
#SBATCH -p pi_ppliang
#SBATCH --gres=gpu:8
#SBATCH -c 120
#SBATCH --mem=480G
#SBATCH --time=24:00:00
#SBATCH -J open-ttt-qwen3-8b-erdos-rl
#SBATCH -o /home/qua/scratch/open-ttt-workspace/logs/slurm-qwen3-8b-erdos-rl-%j.out

set -Eeuo pipefail

cd /home/qua/code/open-ttt

export WORKSPACE=${WORKSPACE:-/home/qua/scratch/open-ttt-workspace}
export LOG_ROOT=${LOG_ROOT:-${WORKSPACE}/logs}
export MODEL_RAW=${MODEL_RAW:-${WORKSPACE}/models/Qwen3-8B}
export REF_CKPT=${REF_CKPT:-${WORKSPACE}/ckpt/Qwen3-8B_torch_dist}
export SAVE_CKPT=${SAVE_CKPT:-${WORKSPACE}/ckpt/Qwen3-8B_erdos_lora}
export ARCHIVE_PATH=${ARCHIVE_PATH:-${WORKSPACE}/data/qwen3_8b_erdos_archive.json}

export TOTAL_GPUS=${TOTAL_GPUS:-8}
export ACTOR_NUM_GPUS=${ACTOR_NUM_GPUS:-4}
export ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-4}
export ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-4}
export TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-1}
export PIPELINE_MODEL_PARALLEL_SIZE=${PIPELINE_MODEL_PARALLEL_SIZE:-2}
export CONTEXT_PARALLEL_SIZE=${CONTEXT_PARALLEL_SIZE:-1}

export NUM_ROLLOUT=${NUM_ROLLOUT:-50}
export ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-2}
export N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-16}
export LORA_RANK=${LORA_RANK:-128}
export LORA_ALPHA=${LORA_ALPHA:-128}
export KILL_STALE_PROCESSES=${KILL_STALE_PROCESSES:-0}

mkdir -p "${LOG_ROOT}" "${WORKSPACE}/data" "${WORKSPACE}/ckpt"

echo "=== qwen3-8b erdos rl slurm ==="
echo "time: $(date -Is)"
echo "host: $(hostname)"
echo "job: ${SLURM_JOB_ID:-unset}"
echo "partition: ${SLURM_JOB_PARTITION:-unset}"
echo "gres: ${SLURM_JOB_GRES:-unset}"
echo "gpus_on_node: ${SLURM_GPUS_ON_NODE:-unset}"
echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "workspace: ${WORKSPACE}"
echo "model_raw: ${MODEL_RAW}"
echo "ref_ckpt: ${REF_CKPT}"
echo "save_ckpt: ${SAVE_CKPT}"
echo "archive_path: ${ARCHIVE_PATH}"
echo "total_gpus: ${TOTAL_GPUS}"
echo "actor_gpus: ${ACTOR_NUM_GPUS}"
echo "rollout_gpus: ${ROLLOUT_NUM_GPUS}"
echo "tp: ${TENSOR_MODEL_PARALLEL_SIZE}"
echo "pp: ${PIPELINE_MODEL_PARALLEL_SIZE}"
echo "num_rollout: ${NUM_ROLLOUT}"
echo "rollout_batch_size: ${ROLLOUT_BATCH_SIZE}"
echo "n_samples_per_prompt: ${N_SAMPLES_PER_PROMPT}"
echo "global_batch_size: ${GLOBAL_BATCH_SIZE}"
echo "lora_rank: ${LORA_RANK}"

bash scripts/run_qwen3_8b_erdos_rl_with_apptainer.sh
