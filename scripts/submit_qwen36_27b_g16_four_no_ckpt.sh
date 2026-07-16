#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/work/mit/ppliang_mit/chonghej/open-ttt-verl-guidance-ttt-modal"
cd "${REPO_ROOT}"

TAG="${1:-$(date +%Y%m%d_%H%M%S)}"
STAGE="scripts/slurm_polyomino_b200_5gpu_qwen36_27b_g16_no_ckpt.sbatch"
CONFIG_DIR="guidance_ttt/config"

submit_variant() {
  local label="$1"
  local prompt_mode="$2"
  local puct_q_mode="$3"
  local config="$4"
  local output_dir="outputs/guidance_ttt/qwen36_27b_g8x16_${label}_no_ckpt_${TAG}"

  local job_id
  job_id=$(sbatch --parsable \
    --job-name="q36-27b-${label}-nc" \
    --export="ALL,CONFIG_OVERRIDE=${config},OUTPUT_DIR_OVERRIDE=${output_dir},PROMPT_MODE=${prompt_mode},PUCT_Q_MODE=${puct_q_mode}" \
    "${STAGE}")
  printf '%-24s job=%s output=%s\n' "${label}" "${job_id}" "${output_dir}"
}

submit_variant \
  "sum-best" \
  "summary_only" \
  "best_child" \
  "${CONFIG_DIR}/polyomino_b200_5gpu_qwen36_27b_gpt_oss_120b_batch8_group8_summary_only_entropic_best_child_500step.yaml"
submit_variant \
  "sum-blend" \
  "summary_only" \
  "blended" \
  "${CONFIG_DIR}/polyomino_b200_5gpu_qwen36_27b_gpt_oss_120b_batch8_group8_summary_only_entropic_blended_500step.yaml"
submit_variant \
  "code-best" \
  "code_delta" \
  "best_child" \
  "${CONFIG_DIR}/polyomino_b200_5gpu_qwen36_27b_gpt_oss_120b_batch8_group8_code_delta_prompt8192_entropic_best_child_500step.yaml"
submit_variant \
  "code-blend" \
  "code_delta" \
  "blended" \
  "${CONFIG_DIR}/polyomino_b200_5gpu_qwen36_27b_gpt_oss_120b_batch8_group8_code_delta_prompt8192_entropic_blended_500step.yaml"
