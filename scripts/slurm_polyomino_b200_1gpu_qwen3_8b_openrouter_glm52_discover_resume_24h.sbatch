#!/usr/bin/env bash
#SBATCH --job-name=q3-8b-glm52-resume
#SBATCH --partition=b200-batch
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=384G
#SBATCH --time=1-00:00:00
#SBATCH --account=ppliang_mit
#SBATCH --output=outputs/slurm/qwen3-8b-openrouter-glm52-resume-24h-%j.out
#SBATCH --error=outputs/slurm/qwen3-8b-openrouter-glm52-resume-24h-%j.err

set -euo pipefail

REPO_ROOT="/work/mit/ppliang_mit/chonghej/open-ttt-verl-guidance-ttt-modal"
cd "${REPO_ROOT}"

export CONFIG="guidance_ttt/config/polyomino_b200_1gpu_qwen3_8b_openrouter_glm52_batch8_group16_summary_only_discover_1day_latest_ckpt.yaml"
export OUTPUT_DIR="outputs/guidance_ttt/qwen3_8b_openrouter_glm52_g8x16_discover_1day_latest_ckpt_184021"
export GUIDANCE_MODEL_DIR="models/Qwen3-8B"
export EXPECTED_GPUS=1
export TRAINING_GPUS=0
export START_EXECUTION_SERVER=0
export BOOTSTRAP_BEFORE_RUN=0
export EXPECTED_GROUPS=4000
export EXPECTED_GROUP_SIZE=16
export EXPECTED_PROMPT_MODE=summary_only

SECRET_FILE=".secrets/openrouter_api_key"
CHECKPOINT_ROOT="${REPO_ROOT}/${OUTPUT_DIR}/checkpoints"
LIBRARY_PATH="${REPO_ROOT}/${OUTPUT_DIR}/library.json"
test -s "${SECRET_FILE}"
test "$(stat -c '%a' "${SECRET_FILE}")" = "600"
test -s "${LIBRARY_PATH}"
test -s "${CHECKPOINT_ROOT}/latest_checkpointed_iteration.txt"
test -f models/Qwen3-8B/config.json

grep -Eq '^[[:space:]]+save_freq:[[:space:]]+1[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+discover_compat:[[:space:]]+true[[:space:]]*$' "${CONFIG}"
grep -Fq 'trainer.resume_mode=auto' "${CONFIG}"
grep -Fq 'trainer.max_actor_ckpt_to_keep=1' "${CONFIG}"
grep -Fq 'trainer.max_critic_ckpt_to_keep=1' "${CONFIG}"

prune_to_latest_checkpoint() {
  local tracker="${CHECKPOINT_ROOT}/latest_checkpointed_iteration.txt"
  local latest_step
  local latest_dir
  local candidate

  case "${CHECKPOINT_ROOT}" in
    "${REPO_ROOT}"/outputs/guidance_ttt/*/checkpoints) ;;
    *)
      echo "Refusing checkpoint cleanup outside the run output: ${CHECKPOINT_ROOT}" >&2
      return 1
      ;;
  esac
  if [[ ! -s "${tracker}" ]]; then
    return 0
  fi
  latest_step="$(tr -d '[:space:]' < "${tracker}")"
  [[ "${latest_step}" =~ ^[1-9][0-9]*$ ]]
  latest_dir="${CHECKPOINT_ROOT}/global_step_${latest_step}"
  test -d "${latest_dir}/actor"

  while IFS= read -r -d '' candidate; do
    if [[ "${candidate}" != "${latest_dir}" ]]; then
      rm -rf -- "${candidate}"
    fi
  done < <(find "${CHECKPOINT_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' -print0)
  test "$(find "${CHECKPOINT_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' | wc -l)" -eq 1
  echo "Retained latest checkpoint only: ${latest_dir}"
}
trap prune_to_latest_checkpoint EXIT

latest_step="$(tr -d '[:space:]' < "${CHECKPOINT_ROOT}/latest_checkpointed_iteration.txt")"
[[ "${latest_step}" =~ ^[1-9][0-9]*$ ]]
test -d "${CHECKPOINT_ROOT}/global_step_${latest_step}/actor"
test -s "${CHECKPOINT_ROOT}/global_step_${latest_step}/data.pt"

backup_path="${REPO_ROOT}/${OUTPUT_DIR}/library.before_resume_${SLURM_JOB_ID}_step${latest_step}.json"
cp --reflink=auto --preserve=mode,timestamps "${LIBRARY_PATH}" "${backup_path}"
PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  python scripts/prepare_guidance_ttt_resume.py --output-dir "${OUTPUT_DIR}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID} RESUME_FROM=global_step_${latest_step}"
echo "OUTPUT_DIR=${OUTPUT_DIR} LIBRARY=${LIBRARY_PATH}"
echo "Allocation=24h planned training boundary=23h"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

scripts/run_polyomino_b200_long.sh preflight

set +e
timeout --foreground --signal=TERM --kill-after=5m 23h scripts/run_polyomino_b200_long.sh run
status=$?
set -e
if [[ ${status} -eq 0 || ${status} -eq 124 ]]; then
  echo "Resume run completed or reached the planned 23-hour training boundary."
  exit 0
fi
exit "${status}"
