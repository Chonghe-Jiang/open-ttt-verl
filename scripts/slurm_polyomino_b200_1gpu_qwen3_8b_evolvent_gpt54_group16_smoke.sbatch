#!/usr/bin/env bash
#SBATCH --job-name=q3-8b-gpt54-g16-smoke
#SBATCH --partition=b200-batch
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=384G
#SBATCH --time=08:00:00
#SBATCH --account=ppliang_mit
#SBATCH --output=outputs/slurm/qwen3-8b-evolvent-gpt54-g16-smoke-%j.out
#SBATCH --error=outputs/slurm/qwen3-8b-evolvent-gpt54-g16-smoke-%j.err

set -euo pipefail

REPO_ROOT="/work/mit/ppliang_mit/chonghej/open-ttt-verl-guidance-ttt-modal"
cd "${REPO_ROOT}"

export CONFIG="guidance_ttt/config/polyomino_b200_1gpu_qwen3_8b_evolvent_gpt54_batch8_group16_summary_only_entropic_best_child_500step.yaml"
export OUTPUT_DIR="outputs/guidance_ttt/qwen3_8b_evolvent_gpt54_g8x16_smoke_${SLURM_JOB_ID}"
export GUIDANCE_MODEL_DIR="models/Qwen3-8B"
export EXPECTED_GPUS=1
export TRAINING_GPUS=0
export START_EXECUTION_SERVER=0
export BOOTSTRAP_BEFORE_RUN=1
export NUM_STEPS_OVERRIDE=1
export TOTAL_EPOCHS_OVERRIDE=1
export SAVE_FREQ_OVERRIDE=-1
export GROUPS_PER_BATCH_OVERRIDE=8
export GROUP_SIZE_OVERRIDE=16
export EXPECTED_GROUPS=8
export EXPECTED_GROUP_SIZE=16
export EXPECTED_PROMPT_MODE=summary_only

CONCURRENCY_FILE=".runtime/evolvent_gpt54_concurrency"
SECRET_FILE=".secrets/evolvent_api_key"
test -s "${SECRET_FILE}"
test "$(stat -c '%a' "${SECRET_FILE}")" = "600"
test -s "${CONCURRENCY_FILE}"
export EXECUTION_CONCURRENCY_OVERRIDE="$(tr -d '[:space:]' < "${CONCURRENCY_FILE}")"
[[ "${EXECUTION_CONCURRENCY_OVERRIDE}" =~ ^[1-9][0-9]*$ ]]

test -f models/Qwen3-8B/config.json
grep -Eq '^[[:space:]]+adv_estimator:[[:space:]]+entropic_adaptive_beta[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+prompt_mode:[[:space:]]+summary_only[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+puct_q_mode:[[:space:]]+best_child[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+model:[[:space:]]+sub2api-gpt-5\.4[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+base_url:[[:space:]]+http://llmapi\.evolventapi\.com/v1[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+prompt_style:[[:space:]]+gpt_api_brief_thinking[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+reasoning_effort:[[:space:]]+low[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+temperature:[[:space:]]+1\.0[[:space:]]*$' "${CONFIG}"
if grep -Eq '^[[:space:]]+seed_library_path:' "${CONFIG}"; then
  echo "GPT-5.4 smoke must generate its own bootstrap seed" >&2
  exit 1
fi

echo "SLURM_JOB_ID=${SLURM_JOB_ID} MODE=summary_only GROUPS=8 GROUP_SIZE=16"
echo "OUTPUT_DIR=${OUTPUT_DIR} EVOLVENT_CONCURRENCY=${EXECUTION_CONCURRENCY_OVERRIDE}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

scripts/run_polyomino_b200_long.sh preflight
scripts/run_polyomino_b200_long.sh run

python scripts/validate_evolvent_gpt54_smoke.py "${OUTPUT_DIR}"
python scripts/validate_rollout_actor_parity.py \
  "outputs/slurm/qwen3-8b-evolvent-gpt54-g16-smoke-${SLURM_JOB_ID}.out"

echo "SMOKE_ONLY_COMPLETE: no formal experiment was submitted"
