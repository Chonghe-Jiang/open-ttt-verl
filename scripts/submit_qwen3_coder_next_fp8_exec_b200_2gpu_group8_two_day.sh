#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/work/mit/ppliang_mit/chonghej/open-ttt-verl-guidance-ttt-modal"
cd "${REPO_ROOT}"
mkdir -p outputs/slurm

tag="${1:-$(date +%Y%m%d_%H%M%S)}"
runtime_dependency=""
if [[ ! -f .runtime/qwen36-exec-site-packages/.vllm-0.19.0-complete || \
      ! -f .runtime/qwen36-exec-site-packages/.mistral-common-1.11.5-complete ]]; then
  runtime_job=$(sbatch --parsable scripts/slurm_setup_qwen36_35b_exec_b200.sbatch)
  runtime_dependency="afterok:${runtime_job}"
  echo "runtime_setup=${runtime_job}"
fi

setup_dependency="${runtime_dependency}"
if [[ -f models/Qwen3-Coder-Next-FP8/.download-complete ]]; then
  echo "Qwen3-Coder-Next-FP8 is already prepared."
else
  setup_args=(--parsable)
  if [[ -n "${runtime_dependency}" ]]; then
    setup_args+=(--dependency="${runtime_dependency}")
  fi
  setup_job=$(sbatch "${setup_args[@]}" scripts/slurm_setup_qwen3_coder_next_fp8_b200.sbatch)
  setup_dependency="afterok:${setup_job}"
  echo "model_setup=${setup_job}"
fi

for mode in code_delta summary_only; do
  smoke_output="outputs/guidance_ttt/qwen3_8b_qwen3_coder_next_fp8_exec_g8_${mode}_smoke_${tag}"
  formal_output="outputs/guidance_ttt/qwen3_8b_qwen3_coder_next_fp8_exec_g8_${mode}_two_day_${tag}"
  smoke_args=(--parsable --job-name="q3cnfp8-g8-${mode}-smoke" --partition=b200-devel --time=02:00:00)
  if [[ -n "${setup_dependency}" ]]; then
    smoke_args+=(--dependency="${setup_dependency}")
  fi
  smoke=$(sbatch "${smoke_args[@]}" \
    --export=ALL,SHARED_OUTPUT_DIR="${smoke_output}",PROMPT_MODE="${mode}",STAGE=smoke \
    scripts/slurm_polyomino_b200_2gpu_qwen3_coder_next_fp8_exec_group8_stage.sbatch)
  day1=$(sbatch --parsable --job-name="q3cnfp8-g8-${mode}-d1" --dependency="afterok:${smoke}" \
    --export=ALL,SHARED_OUTPUT_DIR="${formal_output}",PROMPT_MODE="${mode}",STAGE=day1 \
    scripts/slurm_polyomino_b200_2gpu_qwen3_coder_next_fp8_exec_group8_stage.sbatch)
  day2=$(sbatch --parsable --job-name="q3cnfp8-g8-${mode}-d2" --dependency="afterok:${day1}" \
    --export=ALL,SHARED_OUTPUT_DIR="${formal_output}",PROMPT_MODE="${mode}",STAGE=day2 \
    scripts/slurm_polyomino_b200_2gpu_qwen3_coder_next_fp8_exec_group8_stage.sbatch)
  echo "${mode}: smoke=${smoke} day1=${day1} day2=${day2} output=${formal_output}"
done
