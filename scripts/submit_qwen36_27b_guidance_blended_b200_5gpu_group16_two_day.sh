#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/work/mit/ppliang_mit/chonghej/open-ttt-verl-guidance-ttt-modal"
cd "${REPO_ROOT}"
mkdir -p outputs/slurm

tag="${1:-$(date +%Y%m%d_%H%M%S)}"
group_size=16
config="guidance_ttt/config/polyomino_b200_5gpu_qwen36_27b_gpt_oss_120b_batch8_group8_summary_only_entropic_blended_500step.yaml"

if [[ -f models/Qwen3.6-27B/.download-complete && \
      -f .runtime/qwen36-actor-site-packages/.qwen36-actor-complete ]]; then
  setup_dependency=""
  echo "Qwen3.6-27B guidance model and actor runtime are already prepared."
else
  setup_job=$(sbatch --parsable scripts/slurm_setup_qwen36_27b_guidance_b200.sbatch)
  setup_dependency="afterok:${setup_job}"
  echo "setup=${setup_job}"
fi

smoke_output="outputs/guidance_ttt/qwen36_27b_guidance_g16_summary_only_blended_smoke_${tag}"
formal_output="outputs/guidance_ttt/qwen36_27b_guidance_g16_summary_only_blended_two_day_${tag}"
common_export="ALL,SHARED_OUTPUT_DIR,STAGE,GROUP_SIZE_OVERRIDE=${group_size},CONFIG_OVERRIDE=${config},PUCT_Q_MODE_OVERRIDE=blended"

smoke_args=(--parsable --job-name="q36-27b-g16-sumblend-smoke" --time=04:00:00)
if [[ -n "${setup_dependency}" ]]; then
  smoke_args+=(--dependency="${setup_dependency}")
fi
smoke=$(SHARED_OUTPUT_DIR="${smoke_output}" STAGE=smoke \
  sbatch "${smoke_args[@]}" --export="${common_export}" \
  scripts/slurm_polyomino_b200_5gpu_qwen36_27b_guidance_group8_stage.sbatch)
day1=$(SHARED_OUTPUT_DIR="${formal_output}" STAGE=day1 \
  sbatch --parsable --job-name="q36-27b-g16-sumblend-d1" --dependency="afterok:${smoke}" \
  --export="${common_export}" \
  scripts/slurm_polyomino_b200_5gpu_qwen36_27b_guidance_group8_stage.sbatch)
day2=$(SHARED_OUTPUT_DIR="${formal_output}" STAGE=day2 \
  sbatch --parsable --job-name="q36-27b-g16-sumblend-d2" --dependency="afterok:${day1}" \
  --export="${common_export}" \
  scripts/slurm_polyomino_b200_5gpu_qwen36_27b_guidance_group8_stage.sbatch)

echo "summary_only blended: group_size=${group_size} smoke=${smoke} day1=${day1} day2=${day2} output=${formal_output}"
