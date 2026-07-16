#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/work/mit/ppliang_mit/chonghej/open-ttt-verl-guidance-ttt-modal"
cd "${REPO_ROOT}"
mkdir -p outputs/slurm

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 MODE SUCCESSFUL_SMOKE_JOB_ID OUTPUT_TAG" >&2
  exit 2
fi

mode="$1"
smoke_job="$2"
tag="$3"
case "${mode}" in code_delta|summary_only) ;; *) exit 2 ;; esac

formal_output="outputs/guidance_ttt/qwen35_9b_b200_2gpu_g8_${mode}_two_day_${tag}"
day1=$(sbatch --parsable --job-name="q35-9b-g8-${mode}-d1" --dependency="afterok:${smoke_job}" \
  --export=ALL,SHARED_OUTPUT_DIR="${formal_output}",PROMPT_MODE="${mode}",STAGE=day1 \
  scripts/slurm_polyomino_b200_2gpu_qwen35_9b_group8_stage.sbatch)
day2=$(sbatch --parsable --job-name="q35-9b-g8-${mode}-d2" --dependency="afterok:${day1}" \
  --export=ALL,SHARED_OUTPUT_DIR="${formal_output}",PROMPT_MODE="${mode}",STAGE=day2 \
  scripts/slurm_polyomino_b200_2gpu_qwen35_9b_group8_stage.sbatch)
echo "${mode}: day1=${day1} day2=${day2} output=${formal_output}"
