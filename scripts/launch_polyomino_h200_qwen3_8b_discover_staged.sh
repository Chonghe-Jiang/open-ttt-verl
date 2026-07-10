#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."

SLURM_SCRIPT="${SLURM_SCRIPT:-scripts/slurm_polyomino_h200_qwen3_8b_discover.sh}"
TINY_TIME="${TINY_TIME:-04:00:00}"
ONE_STEP_TIME="${ONE_STEP_TIME:-12:00:00}"
FULL_TIME="${FULL_TIME:-48:00:00}"
FULL_RESET_OUTPUT_DIR="${FULL_RESET_OUTPUT_DIR:-0}"
TINY_GRES="${TINY_GRES:-gpu:h200:1}"
ONE_STEP_GRES="${ONE_STEP_GRES:-gpu:h200:4}"
FULL_GRES="${FULL_GRES:-gpu:h200:4}"

submit_stage() {
  local stage="$1"
  local job_name="$2"
  local time_limit="$3"
  local gres="$4"
  local dependency="${5:-}"
  shift 5 || true

  local args=(
    --parsable
    --job-name="$job_name"
    --time="$time_limit"
    --gres="$gres"
    --export="ALL,RUN_STAGE=$stage,RESET_OUTPUT_DIR=1"
  )
  if [[ "$stage" == "full" ]]; then
    args=(
      --parsable
      --job-name="$job_name"
      --time="$time_limit"
      --gres="$gres"
      --export="ALL,RUN_STAGE=$stage,RESET_OUTPUT_DIR=$FULL_RESET_OUTPUT_DIR"
    )
  fi
  if [[ -n "$dependency" ]]; then
    args+=(--dependency="afterok:$dependency")
  fi
  args+=("$SLURM_SCRIPT")

  local submitted
  submitted="$(sbatch "${args[@]}" "$@")"
  echo "$submitted"
}

tiny_job_raw="$(submit_stage tiny_smoke poly-qwen-disc-tiny "$TINY_TIME" "$TINY_GRES" "" "$@")"
tiny_job="${tiny_job_raw%%;*}"
echo "Submitted tiny smoke: $tiny_job_raw"

one_step_job_raw="$(submit_stage one_step poly-qwen-disc-1step "$ONE_STEP_TIME" "$ONE_STEP_GRES" "$tiny_job" "$@")"
one_step_job="${one_step_job_raw%%;*}"
echo "Submitted full-batch one-step smoke after $tiny_job: $one_step_job_raw"

full_job_raw="$(submit_stage full poly-qwen-disc-full "$FULL_TIME" "$FULL_GRES" "$one_step_job" "$@")"
echo "Submitted full training after $one_step_job: $full_job_raw"
