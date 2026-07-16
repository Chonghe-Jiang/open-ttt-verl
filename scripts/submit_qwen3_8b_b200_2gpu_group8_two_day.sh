#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/work/mit/ppliang_mit/chonghej/open-ttt-verl-guidance-ttt-modal"
cd "${REPO_ROOT}"
mkdir -p outputs/slurm

tag="${1:-$(date +%Y%m%d_%H%M%S)}"
for mode in code_delta summary_only; do
  smoke_output="outputs/guidance_ttt/qwen3_8b_b200_2gpu_g8_${mode}_smoke_${tag}"
  formal_output="outputs/guidance_ttt/qwen3_8b_b200_2gpu_g8_${mode}_two_day_${tag}"

  smoke=$(sbatch --parsable --job-name="q3-8b-g8-${mode}-smoke" \
    --export=ALL,SHARED_OUTPUT_DIR="${smoke_output}",PROMPT_MODE="${mode}",STAGE=smoke \
    scripts/slurm_polyomino_b200_2gpu_qwen3_8b_group8_stage.sbatch)
  day1=$(sbatch --parsable --job-name="q3-8b-g8-${mode}-d1" --dependency="afterok:${smoke}" \
    --export=ALL,SHARED_OUTPUT_DIR="${formal_output}",PROMPT_MODE="${mode}",STAGE=day1 \
    scripts/slurm_polyomino_b200_2gpu_qwen3_8b_group8_stage.sbatch)
  day2=$(sbatch --parsable --job-name="q3-8b-g8-${mode}-d2" --dependency="afterok:${day1}" \
    --export=ALL,SHARED_OUTPUT_DIR="${formal_output}",PROMPT_MODE="${mode}",STAGE=day2 \
    scripts/slurm_polyomino_b200_2gpu_qwen3_8b_group8_stage.sbatch)
  echo "${mode}: smoke=${smoke} day1=${day1} day2=${day2} output=${formal_output}"
done
