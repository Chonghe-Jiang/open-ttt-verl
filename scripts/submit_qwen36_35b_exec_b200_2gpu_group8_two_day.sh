#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/work/mit/ppliang_mit/chonghej/open-ttt-verl-guidance-ttt-modal"
cd "${REPO_ROOT}"
mkdir -p outputs/slurm

tag="${1:-$(date +%Y%m%d_%H%M%S)}"
if [[ -f models/Qwen3.6-35B-A3B/.download-complete && \
      -f .runtime/qwen36-exec-site-packages/.qwen36-exec-complete && \
      -f .runtime/qwen36-exec-site-packages/.mistral-common-1.11.5-complete ]]; then
  setup_dependency=""
  echo "Qwen3.6 model and execution runtime are already prepared."
else
  setup_job=$(sbatch --parsable scripts/slurm_setup_qwen36_35b_exec_b200.sbatch)
  setup_dependency="afterok:${setup_job}"
  echo "setup=${setup_job}"
fi

for mode in code_delta summary_only; do
  smoke_output="outputs/guidance_ttt/qwen3_8b_qwen36_35b_exec_g8_${mode}_smoke_${tag}"
  formal_output="outputs/guidance_ttt/qwen3_8b_qwen36_35b_exec_g8_${mode}_two_day_${tag}"
  # The 1-step gate fits the two-GPU devel QOS and can backfill quickly. The
  # 23-hour continuation stages below retain the sbatch file's batch partition.
  smoke_args=(--parsable --job-name="q36exec-g8-${mode}-smoke" --partition=b200-devel --time=02:00:00)
  if [[ -n "${setup_dependency}" ]]; then
    smoke_args+=(--dependency="${setup_dependency}")
  fi
  smoke=$(sbatch "${smoke_args[@]}" \
    --export=ALL,SHARED_OUTPUT_DIR="${smoke_output}",PROMPT_MODE="${mode}",STAGE=smoke \
    scripts/slurm_polyomino_b200_2gpu_qwen36_35b_exec_group8_stage.sbatch)
  day1=$(sbatch --parsable --job-name="q36exec-g8-${mode}-d1" --dependency="afterok:${smoke}" \
    --export=ALL,SHARED_OUTPUT_DIR="${formal_output}",PROMPT_MODE="${mode}",STAGE=day1 \
    scripts/slurm_polyomino_b200_2gpu_qwen36_35b_exec_group8_stage.sbatch)
  day2=$(sbatch --parsable --job-name="q36exec-g8-${mode}-d2" --dependency="afterok:${day1}" \
    --export=ALL,SHARED_OUTPUT_DIR="${formal_output}",PROMPT_MODE="${mode}",STAGE=day2 \
    scripts/slurm_polyomino_b200_2gpu_qwen36_35b_exec_group8_stage.sbatch)
  echo "${mode}: smoke=${smoke} day1=${day1} day2=${day2} output=${formal_output}"
done
