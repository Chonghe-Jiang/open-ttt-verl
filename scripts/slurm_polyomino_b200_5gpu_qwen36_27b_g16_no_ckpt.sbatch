#!/usr/bin/env bash
#SBATCH --job-name=q36-27b-g16-no-ckpt
#SBATCH --partition=b200-batch
#SBATCH --gres=gpu:b200:5
#SBATCH --cpus-per-task=96
#SBATCH --mem=768G
#SBATCH --time=1-00:00:00
#SBATCH --account=ppliang_mit
#SBATCH --output=outputs/slurm/qwen36-27b-g16-no-ckpt-%j.out
#SBATCH --error=outputs/slurm/qwen36-27b-g16-no-ckpt-%j.err

set -euo pipefail

REPO_ROOT="/work/mit/ppliang_mit/chonghej/open-ttt-verl-guidance-ttt-modal"
cd "${REPO_ROOT}"

: "${CONFIG_OVERRIDE:?submit with CONFIG_OVERRIDE}"
: "${OUTPUT_DIR_OVERRIDE:?submit with OUTPUT_DIR_OVERRIDE}"
: "${PROMPT_MODE:?submit with PROMPT_MODE=summary_only or code_delta}"
: "${PUCT_Q_MODE:?submit with PUCT_Q_MODE=best_child or blended}"

case "${PROMPT_MODE}" in
  summary_only|code_delta) ;;
  *) echo "Unsupported PROMPT_MODE=${PROMPT_MODE}" >&2; exit 2 ;;
esac
case "${PUCT_Q_MODE}" in
  best_child|blended) ;;
  *) echo "Unsupported PUCT_Q_MODE=${PUCT_Q_MODE}" >&2; exit 2 ;;
esac

export CONFIG="${CONFIG_OVERRIDE}"
export OUTPUT_DIR="${OUTPUT_DIR_OVERRIDE}"
export GUIDANCE_MODEL_DIR="models/Qwen3.6-27B"
export EXTRA_CONTAINER_PYTHONPATH="/workspace/guidance/.runtime/qwen36-actor-site-packages"
export EXPECTED_GPUS=5
export TRAINING_GPUS=0,1,2,3
export EXECUTION_GPU=4
export EXPECTED_GROUPS=4000
export EXPECTED_GROUP_SIZE=16
export EXPECTED_PROMPT_MODE="${PROMPT_MODE}"
export GROUP_SIZE_OVERRIDE=16
export SAVE_FREQ_OVERRIDE=-1
export START_EXECUTION_SERVER=1
export BOOTSTRAP_BEFORE_RUN=0

test -f models/Qwen3.6-27B/.download-complete
test -f models/gpt-oss-120b/config.json
test -f .runtime/qwen36-actor-site-packages/.qwen36-actor-fa2-nopad-complete
grep -q 'def expand_packed_lora' .runtime/qwen36-actor-site-packages/vllm/lora/layers/column_parallel_linear.py
grep -q 'set_triton_allocator(tensor.device)' .runtime/qwen36-actor-site-packages/vllm/model_executor/layers/fla/ops/utils.py
grep -Eq '^[[:space:]]+num_steps:[[:space:]]+500[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+total_epochs:[[:space:]]+500[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+adv_estimator:[[:space:]]+entropic_adaptive_beta[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+groups_per_batch:[[:space:]]+8[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+group_size:[[:space:]]+8[[:space:]]*$' "${CONFIG}"
grep -Eq "^[[:space:]]+prompt_mode:[[:space:]]+${PROMPT_MODE}[[:space:]]*$" "${CONFIG}"
grep -Eq "^[[:space:]]+puct_q_mode:[[:space:]]+${PUCT_Q_MODE}[[:space:]]*$" "${CONFIG}"
grep -Fq 'seed_library_path: guidance_ttt/seeds/polyomino_packing/gpt_oss_120b_bootstrap_library.json' "${CONFIG}"
grep -Fq '+actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2' "${CONFIG}"
grep -Fq 'actor_rollout_ref.model.use_remove_padding=False' "${CONFIG}"
grep -Fq 'actor_rollout_ref.actor.use_remove_padding=False' "${CONFIG}"
if [[ "${PROMPT_MODE}" == "code_delta" ]]; then
  grep -Eq '^[[:space:]]+max_prompt_length:[[:space:]]+8192[[:space:]]*$' "${CONFIG}"
  grep -Eq 'actor_rollout_ref\.rollout\.max_model_len=16384$' "${CONFIG}"
fi

echo "SLURM_JOB_ID=${SLURM_JOB_ID} MODE=${PROMPT_MODE} PUCT_Q_MODE=${PUCT_Q_MODE} GROUPS_PER_STEP=8 GROUP_SIZE=16"
echo "OUTPUT_DIR=${OUTPUT_DIR} MAX_STEPS=500 SAVE_FREQ_OVERRIDE=-1"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

scripts/run_polyomino_b200_long.sh preflight

set +e
timeout --foreground --signal=TERM --kill-after=5m 23h scripts/run_polyomino_b200_long.sh run
status=$?
set -e

if find "${OUTPUT_DIR}" -type d \( -name 'global_step_*' -o -name 'checkpoint-*' \) -print -quit | grep -q .; then
  echo "Unexpected checkpoint directory found despite SAVE_FREQ_OVERRIDE=-1" >&2
  exit 1
fi

LOG_PATH="outputs/slurm/qwen36-27b-g16-no-ckpt-${SLURM_JOB_ID}.out"
if grep -q 'training/rollout_actor_probs_pearson_corr:' "${LOG_PATH}"; then
  python scripts/validate_rollout_actor_parity.py "${LOG_PATH}"
fi

if [[ ${status} -eq 0 || ${status} -eq 124 ]]; then
  echo "Run completed or reached the planned 23-hour boundary without checkpoints."
  exit 0
fi
exit "${status}"
