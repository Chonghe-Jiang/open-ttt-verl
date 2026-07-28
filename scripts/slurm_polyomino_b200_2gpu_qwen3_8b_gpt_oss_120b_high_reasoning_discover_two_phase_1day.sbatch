#!/usr/bin/env bash
#SBATCH --job-name=q3-8b-oss-hi-2phase
#SBATCH --partition=b200-batch
#SBATCH --gres=gpu:b200:2
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=1-00:00:00
#SBATCH --account=ppliang_mit
#SBATCH --output=outputs/slurm/q3-8b-oss-hi-2phase-%j.out
#SBATCH --error=outputs/slurm/q3-8b-oss-hi-2phase-%j.err

set -euo pipefail

REPO_ROOT="/work/mit/ppliang_mit/chonghej/open-ttt-verl-guidance-ttt-modal"
cd "${REPO_ROOT}"

export CONFIG="guidance_ttt/config/polyomino_b200_2gpu_qwen3_8b_gpt_oss_120b_high_reasoning_batch8_group16_summary_only_discover_two_phase_1day_no_ckpt.yaml"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/guidance_ttt/qwen3_8b_gpt_oss_120b_high_reasoning_discover_two_phase_1day_${SLURM_JOB_ID}}"
export EXPECTED_GPUS=2
export TRAINING_GPUS=0
export EXECUTION_GPU=1
export EXPECTED_GROUPS=4000
export EXPECTED_GROUP_SIZE=16
export EXPECTED_PROMPT_MODE=summary_only
export EXECUTION_VLLM_MAX_MODEL_LEN=32768
export EXECUTION_VLLM_MAX_NUM_SEQS=16

test -f models/Qwen3-8B/config.json
test -f models/gpt-oss-120b/config.json
grep -Eq '^[[:space:]]+discover_compat:[[:space:]]+true[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+prompt_mode:[[:space:]]+summary_only[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+groups_per_batch:[[:space:]]+8[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+group_size:[[:space:]]+16[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+reasoning_effort:[[:space:]]+high[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+prompt_style:[[:space:]]+qwen_native_thinking[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+phase1_max_tokens:[[:space:]]+16500[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+context_window:[[:space:]]+32768[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+save_freq:[[:space:]]+-1[[:space:]]*$' "${CONFIG}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID} OUTPUT_DIR=${OUTPUT_DIR}"
echo "Qwen3-8B guidance: 8 groups x 16 rollouts; GPT-OSS-120B execution: high reasoning"
echo "Discover two-phase execution: phase1 total context 16500, full context 32768, buffer 50"
echo "Discover-compatible search/RL enabled; checkpoint saving disabled"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

scripts/run_polyomino_b200_long.sh preflight

set +e
timeout --foreground --signal=TERM --kill-after=5m 23h scripts/run_polyomino_b200_long.sh run
status=$?
set -e
if [[ ${status} -eq 0 || ${status} -eq 124 ]]; then
  echo "Experiment completed or reached the planned 23-hour boundary."
  exit 0
fi
exit "${status}"
