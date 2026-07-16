#!/usr/bin/env bash
#SBATCH --job-name=q36-27b-guidance-g8
#SBATCH --partition=b200-batch
#SBATCH --gres=gpu:b200:5
#SBATCH --cpus-per-task=96
#SBATCH --mem=768G
#SBATCH --time=1-00:00:00
#SBATCH --account=ppliang_mit
#SBATCH --output=outputs/slurm/qwen36-27b-guidance-g8-%j.out
#SBATCH --error=outputs/slurm/qwen36-27b-guidance-g8-%j.err

set -euo pipefail

REPO_ROOT="/work/mit/ppliang_mit/chonghej/open-ttt-verl-guidance-ttt-modal"
cd "${REPO_ROOT}"

: "${SHARED_OUTPUT_DIR:?submit with SHARED_OUTPUT_DIR}"

export CONFIG="guidance_ttt/config/polyomino_b200_5gpu_qwen36_27b_gpt_oss_120b_batch8_group8_summary_only_entropic_best_child_500step.yaml"
export OUTPUT_DIR="${SHARED_OUTPUT_DIR}"
export GUIDANCE_MODEL_DIR="models/Qwen3.6-27B"
export EXTRA_CONTAINER_PYTHONPATH="/workspace/guidance/.runtime/qwen36-actor-site-packages"
export EXPECTED_GPUS=5
export TRAINING_GPUS=0,1,2,3
export EXECUTION_GPU=4
export EXPECTED_GROUP_SIZE="${GROUP_SIZE_OVERRIDE:-8}"
export EXPECTED_PROMPT_MODE=summary_only

STAGE="${STAGE:-smoke}"
case "${STAGE}" in
  smoke)
    export NUM_STEPS_OVERRIDE=1
    export TOTAL_EPOCHS_OVERRIDE=1
    export SAVE_FREQ_OVERRIDE=-1
    export EXPECTED_GROUPS=8
    ;;
  day1|day2)
    unset NUM_STEPS_OVERRIDE TOTAL_EPOCHS_OVERRIDE SAVE_FREQ_OVERRIDE
    export EXPECTED_GROUPS=4000
    ;;
  *)
    echo "Unknown STAGE=${STAGE}; expected smoke, day1, or day2" >&2
    exit 2
    ;;
esac

test -f models/Qwen3.6-27B/.download-complete
test -f models/gpt-oss-120b/config.json
test -f .runtime/qwen36-actor-site-packages/.qwen36-actor-complete
grep -q 'def expand_packed_lora' .runtime/qwen36-actor-site-packages/vllm/lora/layers/column_parallel_linear.py
grep -q 'set_triton_allocator(tensor.device)' .runtime/qwen36-actor-site-packages/vllm/model_executor/layers/fla/ops/utils.py
grep -Eq '^[[:space:]]+adv_estimator:[[:space:]]+entropic_adaptive_beta[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+prompt_mode:[[:space:]]+summary_only[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+puct_q_mode:[[:space:]]+best_child[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+save_freq:[[:space:]]+1[[:space:]]*$' "${CONFIG}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID} STAGE=${STAGE} MODE=summary_only"
echo "OUTPUT_DIR=${OUTPUT_DIR} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

scripts/run_polyomino_b200_long.sh preflight
if [[ "${STAGE}" == "smoke" ]]; then
  scripts/run_polyomino_b200_long.sh run
  exit 0
fi

set +e
timeout --foreground --signal=TERM --kill-after=5m 23h scripts/run_polyomino_b200_long.sh run
status=$?
set -e
if [[ ${status} -eq 0 || ${status} -eq 124 ]]; then
  echo "Stage ${STAGE} completed or reached the planned 23-hour continuation boundary."
  exit 0
fi
exit "${status}"
