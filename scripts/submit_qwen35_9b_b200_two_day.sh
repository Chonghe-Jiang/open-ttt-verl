#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/work/mit/ppliang_mit/chonghej/open-ttt-verl-guidance-ttt-modal"
cd "${REPO_ROOT}"
mkdir -p outputs/slurm

tag="${1:-$(date +%Y%m%d_%H%M%S)}"
setup_job=$(sbatch --parsable scripts/slurm_setup_qwen35_9b_b200.sbatch)
echo "setup=${setup_job}"

previous_smoke=""
for setting in 8x32 16x16; do
  groups="${setting%x*}"
  size="${setting#*x}"
  smoke_output="outputs/guidance_ttt/qwen35_9b_b200_${setting}_smoke_${tag}"
  formal_output="outputs/guidance_ttt/qwen35_9b_b200_${setting}_two_day_${tag}"

  smoke_dependency="${setup_job}"
  if [[ -n "${previous_smoke}" ]]; then
    smoke_dependency="${previous_smoke}"
  fi
  smoke=$(sbatch --parsable --dependency="afterok:${smoke_dependency}" \
    --export=ALL,SHARED_OUTPUT_DIR="${smoke_output}",GROUPS_PER_BATCH="${groups}",GROUP_SIZE="${size}",STAGE=smoke \
    scripts/slurm_polyomino_b200_3gpu_qwen35_9b_256samples_stage.sbatch)
  day1=$(sbatch --parsable --dependency="afterok:${smoke}" \
    --export=ALL,SHARED_OUTPUT_DIR="${formal_output}",GROUPS_PER_BATCH="${groups}",GROUP_SIZE="${size}",STAGE=day1 \
    scripts/slurm_polyomino_b200_3gpu_qwen35_9b_256samples_stage.sbatch)
  day2=$(sbatch --parsable --dependency="afterok:${day1}" \
    --export=ALL,SHARED_OUTPUT_DIR="${formal_output}",GROUPS_PER_BATCH="${groups}",GROUP_SIZE="${size}",STAGE=day2 \
    scripts/slurm_polyomino_b200_3gpu_qwen35_9b_256samples_stage.sbatch)
  echo "${setting}: smoke=${smoke} day1=${day1} day2=${day2} output=${formal_output}"
  previous_smoke="${smoke}"
done
