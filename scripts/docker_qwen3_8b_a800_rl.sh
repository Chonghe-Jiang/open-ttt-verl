#!/bin/bash
set -Eeuo pipefail

export RUN_ID=${RUN_ID:-qwen3_8b_a800_$(date +%Y%m%d_%H%M%S)}

# A800 80GB validation/training profile. Keep the model, optimizer, LoRA, KL,
# and 30k response length aligned with the 8B200 script. Use TP=2 on the actor
# to reduce per-rank vocab/logit memory for occasional 30k responses.
export NUM_ROLLOUT=${NUM_ROLLOUT:-50}
export ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-1}
export N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-4}
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-4}
export MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
export ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-30000}
export SAVE_INTERVAL=${SAVE_INTERVAL:-10}

export TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-2}
export PIPELINE_MODEL_PARALLEL_SIZE=${PIPELINE_MODEL_PARALLEL_SIZE:-2}
export CONTEXT_PARALLEL_SIZE=${CONTEXT_PARALLEL_SIZE:-1}

export SAVE_CKPT=${SAVE_CKPT:-${WORKSPACE:-/root/workspace}/ckpt/Qwen3-8B_erdos_lora_${RUN_ID}}
export ARCHIVE_PATH=${ARCHIVE_PATH:-${WORKSPACE:-/root/workspace}/data/qwen3_8b_erdos_archive_${RUN_ID}.json}

echo "=== qwen3-8b erdos 8xa800 training ==="
echo "run_id: ${RUN_ID}"
echo "num_rollout: ${NUM_ROLLOUT}"
echo "rollout_batch_size: ${ROLLOUT_BATCH_SIZE}"
echo "n_samples_per_prompt: ${N_SAMPLES_PER_PROMPT}"
echo "global_batch_size: ${GLOBAL_BATCH_SIZE}"
echo "micro_batch_size: ${MICRO_BATCH_SIZE}"
echo "rollout_max_response_len: ${ROLLOUT_MAX_RESPONSE_LEN}"
echo "tp_pp_cp: ${TENSOR_MODEL_PARALLEL_SIZE}/${PIPELINE_MODEL_PARALLEL_SIZE}/${CONTEXT_PARALLEL_SIZE}"
echo "archive_path: ${ARCHIVE_PATH}"
echo "save_ckpt: ${SAVE_CKPT}"

exec scripts/docker_qwen3_8b_8b200_rl.sh
