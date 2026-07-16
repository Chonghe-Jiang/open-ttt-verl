#!/usr/bin/env bash
#SBATCH --job-name=q35-9b-256-b200
#SBATCH --partition=b200-batch
#SBATCH --gres=gpu:b200:3
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=1-00:00:00
#SBATCH --account=ppliang_mit
#SBATCH --output=outputs/slurm/qwen35-9b-b200-256-%j.out
#SBATCH --error=outputs/slurm/qwen35-9b-b200-256-%j.err

set -euo pipefail

REPO_ROOT="/work/mit/ppliang_mit/chonghej/open-ttt-verl-guidance-ttt-modal"
cd "${REPO_ROOT}"

: "${SHARED_OUTPUT_DIR:?submit with SHARED_OUTPUT_DIR}"
: "${GROUPS_PER_BATCH:?submit with GROUPS_PER_BATCH (8 or 16)}"
: "${GROUP_SIZE:?submit with GROUP_SIZE (32 or 16)}"

if (( GROUPS_PER_BATCH * GROUP_SIZE != 256 )); then
  echo "Expected 256 samples/step, got ${GROUPS_PER_BATCH}x${GROUP_SIZE}" >&2
  exit 2
fi

export CONFIG="guidance_ttt/config/polyomino_b200_3gpu_qwen35_9b_gpt_oss_120b_256samples_code_delta_entropic_best_child_50step.yaml"
export OUTPUT_DIR="${SHARED_OUTPUT_DIR}"
export GUIDANCE_MODEL_DIR="models/Qwen3.5-9B"
export EXTRA_CONTAINER_PYTHONPATH="/workspace/guidance/.runtime/qwen35-site-packages"
export EXPECTED_GPUS=3
export TRAINING_GPUS=0,1
export EXECUTION_GPU=2
export EXPECTED_GROUP_SIZE="${GROUP_SIZE}"
export EXPECTED_PROMPT_MODE=code_delta
export GROUPS_PER_BATCH_OVERRIDE="${GROUPS_PER_BATCH}"
export GROUP_SIZE_OVERRIDE="${GROUP_SIZE}"

STAGE="${STAGE:-smoke}"
case "${STAGE}" in
  smoke)
    export NUM_STEPS_OVERRIDE=1
    export TOTAL_EPOCHS_OVERRIDE=1
    export SAVE_FREQ_OVERRIDE=-1
    export EXPECTED_GROUPS="${GROUPS_PER_BATCH}"
    ;;
  day1)
    unset SAVE_FREQ_OVERRIDE
    export NUM_STEPS_OVERRIDE=20
    export TOTAL_EPOCHS_OVERRIDE=20
    export EXPECTED_GROUPS="$((GROUPS_PER_BATCH * 20))"
    ;;
  day2)
    unset NUM_STEPS_OVERRIDE TOTAL_EPOCHS_OVERRIDE SAVE_FREQ_OVERRIDE
    export EXPECTED_GROUPS="$((GROUPS_PER_BATCH * 50))"
    ;;
  *)
    echo "Unknown STAGE=${STAGE}; expected smoke, day1, or day2" >&2
    exit 2
    ;;
esac

test -f models/Qwen3.5-9B/.download-complete
test -f .runtime/qwen35-site-packages/.transformers-main-complete
test -f .runtime/qwen35-site-packages/.vllm-0.18.0-complete
grep -Eq '^[[:space:]]+adv_estimator:[[:space:]]+entropic_adaptive_beta[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+puct_q_mode:[[:space:]]+best_child[[:space:]]*$' "${CONFIG}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID} STAGE=${STAGE} SETTING=${GROUPS_PER_BATCH}x${GROUP_SIZE}"
echo "OUTPUT_DIR=${OUTPUT_DIR} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

scripts/run_polyomino_b200_long.sh preflight
if [[ "${STAGE}" == "smoke" ]]; then
  scripts/run_polyomino_b200_long.sh run
  exit 0
fi

# Stop before Slurm's hard 24-hour limit. Per-step checkpoints make the timeout
# an expected successful stage boundary, allowing the dependent day to resume.
set +e
timeout --foreground --signal=TERM --kill-after=5m 23h scripts/run_polyomino_b200_long.sh run
status=$?
set -e
if [[ ${status} -eq 0 || ${status} -eq 124 ]]; then
  echo "Stage ${STAGE} completed or reached the planned 23-hour continuation boundary."
  exit 0
fi
exit "${status}"
