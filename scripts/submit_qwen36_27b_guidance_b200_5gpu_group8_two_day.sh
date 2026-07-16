#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/work/mit/ppliang_mit/chonghej/open-ttt-verl-guidance-ttt-modal"
cd "${REPO_ROOT}"
mkdir -p outputs/slurm

tag="${1:-$(date +%Y%m%d_%H%M%S)}"
group_size="${GROUP_SIZE:-16}"
if [[ "${group_size}" != "8" && "${group_size}" != "16" ]]; then
  echo "GROUP_SIZE must be 8 or 16, got ${group_size}" >&2
  exit 2
fi
if [[ -f models/Qwen3.6-27B/.download-complete && \
      -f .runtime/qwen36-actor-site-packages/.qwen36-actor-complete ]]; then
  setup_dependency=""
  echo "Qwen3.6-27B guidance model and actor runtime are already prepared."
else
  setup_job=$(sbatch --parsable scripts/slurm_setup_qwen36_27b_guidance_b200.sbatch)
  setup_dependency="afterok:${setup_job}"
  echo "setup=${setup_job}"
fi

smoke_output="outputs/guidance_ttt/qwen36_27b_guidance_g${group_size}_summary_only_smoke_${tag}"
formal_output="outputs/guidance_ttt/qwen36_27b_guidance_g${group_size}_summary_only_two_day_${tag}"
smoke_args=(--parsable --job-name="q36-27b-g${group_size}-summary-smoke" --time=04:00:00)
if [[ -n "${setup_dependency}" ]]; then
  smoke_args+=(--dependency="${setup_dependency}")
fi
smoke=$(sbatch "${smoke_args[@]}" \
  --export=ALL,SHARED_OUTPUT_DIR="${smoke_output}",STAGE=smoke,GROUP_SIZE_OVERRIDE="${group_size}" \
  scripts/slurm_polyomino_b200_5gpu_qwen36_27b_guidance_group8_stage.sbatch)
day1=$(sbatch --parsable --job-name="q36-27b-g${group_size}-summary-d1" --dependency="afterok:${smoke}" \
  --export=ALL,SHARED_OUTPUT_DIR="${formal_output}",STAGE=day1,GROUP_SIZE_OVERRIDE="${group_size}" \
  scripts/slurm_polyomino_b200_5gpu_qwen36_27b_guidance_group8_stage.sbatch)
day2=$(sbatch --parsable --job-name="q36-27b-g${group_size}-summary-d2" --dependency="afterok:${day1}" \
  --export=ALL,SHARED_OUTPUT_DIR="${formal_output}",STAGE=day2,GROUP_SIZE_OVERRIDE="${group_size}" \
  scripts/slurm_polyomino_b200_5gpu_qwen36_27b_guidance_group8_stage.sbatch)

echo "summary_only: group_size=${group_size} smoke=${smoke} day1=${day1} day2=${day2} output=${formal_output}"
