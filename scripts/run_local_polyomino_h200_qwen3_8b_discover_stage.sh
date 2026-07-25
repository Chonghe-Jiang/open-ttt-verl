#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."

RUN_STAGE="${RUN_STAGE:-full}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-outputs/guidance_ttt/polyomino_h200_4gpu_qwen3_8b_discover_batch8_group16_aligned}"

case "$RUN_STAGE" in
  tiny_smoke)
    export OUTPUT_DIR="${OUTPUT_DIR:-${BASE_OUTPUT_DIR}_tiny_smoke}"
    export RESET_OUTPUT_DIR="${RESET_OUTPUT_DIR:-1}"
    export EXPECTED_H200_COUNT="${EXPECTED_H200_COUNT:-1}"
    export EXPECTED_ROLLOUT_N="${EXPECTED_ROLLOUT_N:-1}"
    export EXPECTED_TENSOR_MODEL_PARALLEL_SIZE="${EXPECTED_TENSOR_MODEL_PARALLEL_SIZE:-1}"
    EXTRA_OVERRIDES=(
      "run.experiment_name=polyomino_qwen3_8b_direct_discover_h200_4gpu_tiny_smoke"
      "run.num_steps=1"
      "run.total_epochs=1"
      "run.n_gpus_per_node=1"
      "run.tensor_model_parallel_size=1"
      "run.ppo_mini_batch_size=1"
      "ttt.groups_per_batch=1"
      "ttt.group_size=1"
      "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
      "actor_rollout_ref.rollout.max_num_seqs=1"
    )
    ;;
  one_step)
    export OUTPUT_DIR="${OUTPUT_DIR:-${BASE_OUTPUT_DIR}_one_step_full_batch}"
    export RESET_OUTPUT_DIR="${RESET_OUTPUT_DIR:-1}"
    export EXPECTED_H200_COUNT="${EXPECTED_H200_COUNT:-4}"
    export EXPECTED_ROLLOUT_N="${EXPECTED_ROLLOUT_N:-16}"
    export EXPECTED_TENSOR_MODEL_PARALLEL_SIZE="${EXPECTED_TENSOR_MODEL_PARALLEL_SIZE:-4}"
    EXTRA_OVERRIDES=(
      "run.experiment_name=polyomino_qwen3_8b_direct_discover_h200_4gpu_one_step_full_batch"
      "run.num_steps=1"
      "run.total_epochs=1"
    )
    ;;
  full)
    export OUTPUT_DIR="${OUTPUT_DIR:-${BASE_OUTPUT_DIR}}"
    export RESET_OUTPUT_DIR="${RESET_OUTPUT_DIR:-1}"
    export EXPECTED_H200_COUNT="${EXPECTED_H200_COUNT:-4}"
    export EXPECTED_ROLLOUT_N="${EXPECTED_ROLLOUT_N:-16}"
    export EXPECTED_TENSOR_MODEL_PARALLEL_SIZE="${EXPECTED_TENSOR_MODEL_PARALLEL_SIZE:-4}"
    EXTRA_OVERRIDES=()
    ;;
  *)
    echo "Unknown RUN_STAGE=$RUN_STAGE. Expected tiny_smoke, one_step, or full." >&2
    exit 2
    ;;
esac

echo "RUN_STAGE=$RUN_STAGE"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "RESET_OUTPUT_DIR=$RESET_OUTPUT_DIR"

scripts/run_local_polyomino_h200_qwen3_8b_discover.sh "${EXTRA_OVERRIDES[@]}" "$@"
