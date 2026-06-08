#!/bin/bash
set -Eeuo pipefail

export WORKSPACE=${WORKSPACE:-/root/workspace}
export REPO_DIR=${REPO_DIR:-/root/workspace/erdos}

export NPROC=${NPROC:-2}
export TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-1}
export PIPELINE_MODEL_PARALLEL_SIZE=${PIPELINE_MODEL_PARALLEL_SIZE:-2}
export EXPERT_MODEL_PARALLEL_SIZE=${EXPERT_MODEL_PARALLEL_SIZE:-1}
export EXPERT_TENSOR_PARALLEL_SIZE=${EXPERT_TENSOR_PARALLEL_SIZE:-1}

export TOTAL_GPUS=${TOTAL_GPUS:-2}
export NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-2}
export ACTOR_NUM_GPUS=${ACTOR_NUM_GPUS:-2}
export ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-2}
export ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}
export COLOCATE=${COLOCATE:-1}

cd "${REPO_DIR}"

scripts/download_qwen3_8b.sh
scripts/convert_qwen3_8b_in_container.sh
scripts/run_qwen3_8b_erdos_rl_in_container.sh
