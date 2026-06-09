#!/bin/bash
set -Eeuo pipefail

export NUM_ROLLOUT=${NUM_ROLLOUT:-1}
export ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-1}
export N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-2}
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-2}
export SAVE_INTERVAL=${SAVE_INTERVAL:-1}
export RUN_ID=${RUN_ID:-qwen3_8b_8b200_onestep_$(date +%Y%m%d_%H%M%S)}
export SAVE_CKPT=${SAVE_CKPT:-${WORKSPACE:-/root/workspace}/ckpt/Qwen3-8B_erdos_lora_${RUN_ID}}
export ARCHIVE_PATH=${ARCHIVE_PATH:-${WORKSPACE:-/root/workspace}/data/qwen3_8b_erdos_archive_${RUN_ID}.json}

echo "=== qwen3-8b erdos 8b200 one-step ==="
echo "run_id: ${RUN_ID}"
echo "num_rollout: ${NUM_ROLLOUT}"
echo "rollout_batch_size: ${ROLLOUT_BATCH_SIZE}"
echo "n_samples_per_prompt: ${N_SAMPLES_PER_PROMPT}"
echo "global_batch_size: ${GLOBAL_BATCH_SIZE}"
echo "rollout_max_response_len: ${ROLLOUT_MAX_RESPONSE_LEN:-30000}"
echo "archive_path: ${ARCHIVE_PATH}"
echo "save_ckpt: ${SAVE_CKPT}"

exec scripts/docker_qwen3_8b_8b200_rl.sh
