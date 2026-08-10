#!/usr/bin/env bash
#SBATCH --job-name=trimul-g8x32-s30
#SBATCH --partition=b200-batch
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=384G
#SBATCH --time=1-00:00:00
#SBATCH --account=ppliang_mit
#SBATCH --output=outputs/slurm/trimul-glm52-thinking-cache-g8x32-30step-day-%j.out
#SBATCH --error=outputs/slurm/trimul-glm52-thinking-cache-g8x32-30step-day-%j.err

set -euo pipefail

REPO_ROOT="/work/mit/ppliang_mit/chonghej/open-ttt-verl-guidance-ttt-modal"
cd "${REPO_ROOT}"

export CONFIG="guidance_ttt/config/trimul_b200_1gpu_qwen3_8b_evolvent_glm52_thinking_cache_batch8_group32_30step.yaml"
: "${OUTPUT_DIR:?Submit with a shared OUTPUT_DIR for the daily chain}"
export EXPECTED_GPUS=1
TARGET_STEPS=30
CHECKPOINT_ROOT="${REPO_ROOT}/${OUTPUT_DIR}/checkpoints"
TRACKER="${CHECKPOINT_ROOT}/latest_checkpointed_iteration.txt"
LIBRARY_PATH="${REPO_ROOT}/${OUTPUT_DIR}/library.json"

test -s .secrets/evolvent_api_key
test -s .secrets/trimul_judge_token
test -s .runtime/trimul_judge_url
test "$(stat -c '%a' .secrets/evolvent_api_key)" = "600"
test "$(stat -c '%a' .secrets/trimul_judge_token)" = "600"
test -s guidance_ttt/seeds/trimul/glm52_scratch_bootstrap_library.json
test -f models/Qwen3-8B/config.json

grep -Eq '^[[:space:]]+num_steps:[[:space:]]+30[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+total_epochs:[[:space:]]+30[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+max_prompt_length:[[:space:]]+6144[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+max_response_length:[[:space:]]+10240[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+n_gpus_per_node:[[:space:]]+1[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+ppo_mini_batch_size:[[:space:]]+8[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+ppo_micro_batch_size_per_gpu:[[:space:]]+2[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+learning_rate:[[:space:]]+4\.0e-5[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+use_kl_loss:[[:space:]]+false[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+discover_compat:[[:space:]]+true[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+groups_per_batch:[[:space:]]+8[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+group_size:[[:space:]]+32[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+reasoning_effort:[[:space:]]+high[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+-[[:space:]]+reasoning_effort[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+max_tokens:[[:space:]]+81920[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+concurrency:[[:space:]]+16[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+concurrency:[[:space:]]+32[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+cache_prime_first:[[:space:]]+true[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+save_freq:[[:space:]]+1[[:space:]]*$' "${CONFIG}"
grep -Fq 'actor_rollout_ref.rollout.max_model_len=16384' "${CONFIG}"
grep -Fq 'trainer.resume_mode=auto' "${CONFIG}"
grep -Fq 'trainer.max_actor_ckpt_to_keep=1' "${CONFIG}"
grep -Fq 'trainer.max_critic_ckpt_to_keep=1' "${CONFIG}"

mkdir -p "${REPO_ROOT}/${OUTPUT_DIR}"
git rev-parse HEAD > "${REPO_ROOT}/${OUTPUT_DIR}/launch_git_commit.txt"

prune_to_latest_checkpoint() {
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
  if [[ ! -s "${TRACKER}" ]]; then
    return 0
  fi
  latest_step="$(tr -d '[:space:]' < "${TRACKER}")"
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

latest_step=0
if [[ -s "${TRACKER}" ]]; then
  latest_step="$(tr -d '[:space:]' < "${TRACKER}")"
  [[ "${latest_step}" =~ ^[1-9][0-9]*$ ]]
  test -d "${CHECKPOINT_ROOT}/global_step_${latest_step}/actor"
  test -s "${CHECKPOINT_ROOT}/global_step_${latest_step}/data.pt"
  if (( latest_step >= TARGET_STEPS )); then
    echo "Target already complete at global_step_${latest_step}; exiting chained allocation."
    exit 0
  fi
  test -s "${LIBRARY_PATH}"
  cp --reflink=auto --preserve=mode,timestamps \
    "${LIBRARY_PATH}" \
    "${REPO_ROOT}/${OUTPUT_DIR}/library.before_resume_${SLURM_JOB_ID}_step${latest_step}.json"
  PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    python scripts/prepare_guidance_ttt_resume.py --output-dir "${OUTPUT_DIR}"
elif [[ -s "${LIBRARY_PATH}" ]]; then
  cp --reflink=auto --preserve=mode,timestamps \
    "${LIBRARY_PATH}" \
    "${REPO_ROOT}/${OUTPUT_DIR}/library.before_restart_${SLURM_JOB_ID}_step0.json"
  PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" python - "${LIBRARY_PATH}" <<'PY'
import sys
from guidance_ttt.library import GuidanceLibrary

rollback = GuidanceLibrary(sys.argv[1]).rollback_incomplete_steps_after(0)
print(f"Restart state ready at step 0: {rollback}")
PY
fi

echo "SLURM_JOB_ID=${SLURM_JOB_ID} RESUME_FROM=${latest_step} TARGET_STEPS=${TARGET_STEPS}"
echo "OUTPUT_DIR=${OUTPUT_DIR} GPU=B200 MODEL=glm-5.2 REASONING_EFFORT=high"
echo "SHAPE=8x32 EXEC_CONCURRENCY=32 EVAL_CONCURRENCY=16 PROMPT=6144 RESPONSE=10240"
echo "Allocation=24h planned training boundary=23h SAVE_EVERY_STEP=true KEEP_LATEST=1"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
scripts/run_trimul_b200_long.sh preflight

set +e
timeout --foreground --signal=TERM --kill-after=5m 23h scripts/run_trimul_b200_long.sh run
status=$?
set -e
if [[ ${status} -eq 0 ]]; then
  test -s "${TRACKER}"
  completed_step="$(tr -d '[:space:]' < "${TRACKER}")"
  if (( completed_step < TARGET_STEPS )); then
    echo "Trainer exited successfully before target: step=${completed_step}, target=${TARGET_STEPS}" >&2
    exit 1
  fi
  echo "TriMul 30-step experiment completed at global_step_${completed_step}."
  exit 0
fi
if [[ ${status} -eq 124 ]]; then
  test -s "${TRACKER}"
  echo "Reached the planned 23-hour boundary; next daily job will resume from step $(<"${TRACKER}")."
  exit 0
fi
exit "${status}"
