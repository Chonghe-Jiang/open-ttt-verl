#!/usr/bin/env bash
#SBATCH --job-name=q36-27b-oss-hi-2phase
#SBATCH --partition=b200-batch
#SBATCH --gres=gpu:b200:5
#SBATCH --cpus-per-task=96
#SBATCH --mem=768G
#SBATCH --time=1-00:00:00
#SBATCH --account=ppliang_mit
#SBATCH --output=outputs/slurm/q36-27b-oss-hi-2phase-%j.out
#SBATCH --error=outputs/slurm/q36-27b-oss-hi-2phase-%j.err

set -euo pipefail

REPO_ROOT="/work/mit/ppliang_mit/chonghej/open-ttt-verl-guidance-ttt-modal"
cd "${REPO_ROOT}"

export CONFIG="guidance_ttt/config/polyomino_b200_5gpu_qwen36_27b_gpt_oss_120b_high_reasoning_batch8_group16_summary_only_discover_two_phase_1day_no_ckpt.yaml"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/guidance_ttt/qwen36_27b_gpt_oss_120b_high_reasoning_discover_two_phase_1day_${SLURM_JOB_ID}}"
export GUIDANCE_MODEL_DIR="models/Qwen3.6-27B"
export EXTRA_CONTAINER_PYTHONPATH="/workspace/guidance/.runtime/qwen36-actor-site-packages"
export EXPECTED_GPUS=5
export TRAINING_GPUS=0,1,2,3
export EXECUTION_GPU=4
export EXPECTED_GROUPS=4000
export EXPECTED_GROUP_SIZE=16
export EXPECTED_PROMPT_MODE=summary_only
export EXECUTION_VLLM_MAX_MODEL_LEN=49152
export EXECUTION_VLLM_MAX_NUM_SEQS=16
export START_EXECUTION_SERVER=1

test -f models/Qwen3.6-27B/.download-complete
test -f models/gpt-oss-120b/config.json
test -f .runtime/qwen36-actor-site-packages/.qwen36-actor-fa2-nopad-complete
grep -q 'def expand_packed_lora' .runtime/qwen36-actor-site-packages/vllm/lora/layers/column_parallel_linear.py
grep -q 'set_triton_allocator(tensor.device)' .runtime/qwen36-actor-site-packages/vllm/model_executor/layers/fla/ops/utils.py
grep -Eq '^[[:space:]]+discover_compat:[[:space:]]+true[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+prompt_mode:[[:space:]]+summary_only[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+groups_per_batch:[[:space:]]+8[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+group_size:[[:space:]]+16[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+reasoning_effort:[[:space:]]+high[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+prompt_style:[[:space:]]+qwen_native_thinking[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+phase1_max_tokens:[[:space:]]+30000[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+context_window:[[:space:]]+49152[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+n_gpus_per_node:[[:space:]]+4[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+save_freq:[[:space:]]+-1[[:space:]]*$' "${CONFIG}"
grep -Fq '+actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2' "${CONFIG}"
grep -Fq 'actor_rollout_ref.model.use_remove_padding=False' "${CONFIG}"
grep -Fq 'actor_rollout_ref.actor.use_remove_padding=False' "${CONFIG}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID} OUTPUT_DIR=${OUTPUT_DIR}"
echo "Qwen3.6-27B guidance: 8 groups x 16 rollouts on GPUs 0-3; GPT-OSS-120B high-reasoning execution on GPU 4"
echo "Discover two-phase execution: phase1 total context 30000, full context 49152, buffer 50"
echo "Discover-compatible search/RL enabled; checkpoint saving disabled"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

scripts/run_polyomino_b200_long.sh preflight

set +e
timeout --foreground --signal=TERM --kill-after=5m 23h scripts/run_polyomino_b200_long.sh run
status=$?
set -e

if find "${OUTPUT_DIR}" -type d \( -name 'global_step_*' -o -name 'checkpoint-*' \) -print -quit | grep -q .; then
  echo "Unexpected checkpoint directory found despite save_freq=-1" >&2
  exit 1
fi

LOG_PATH="outputs/slurm/q36-27b-oss-hi-2phase-${SLURM_JOB_ID}.out"
if grep -q 'training/rollout_actor_probs_pearson_corr:' "${LOG_PATH}"; then
  python scripts/validate_rollout_actor_parity.py "${LOG_PATH}"
fi

if [[ ${status} -eq 0 || ${status} -eq 124 ]]; then
  echo "Experiment completed or reached the planned 23-hour boundary."
  exit 0
fi
exit "${status}"
