#!/usr/bin/env bash
#SBATCH --job-name=q3-8b-glm52-discover
#SBATCH --partition=b200-batch
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=384G
#SBATCH --time=1-00:00:00
#SBATCH --account=ppliang_mit
#SBATCH --output=outputs/slurm/qwen3-8b-openrouter-glm52-discover-1day-%j.out
#SBATCH --error=outputs/slurm/qwen3-8b-openrouter-glm52-discover-1day-%j.err

set -euo pipefail

REPO_ROOT="/work/mit/ppliang_mit/chonghej/open-ttt-verl-guidance-ttt-modal"
cd "${REPO_ROOT}"

export CONFIG="guidance_ttt/config/polyomino_b200_1gpu_qwen3_8b_openrouter_glm52_batch8_group16_summary_only_discover_1day_latest_ckpt.yaml"
export OUTPUT_DIR="outputs/guidance_ttt/qwen3_8b_openrouter_glm52_g8x16_discover_1day_latest_ckpt_${SLURM_JOB_ID}"
export GUIDANCE_MODEL_DIR="models/Qwen3-8B"
export EXPECTED_GPUS=1
export TRAINING_GPUS=0
export START_EXECUTION_SERVER=0
export BOOTSTRAP_BEFORE_RUN=0
export EXPECTED_GROUPS=4000
export EXPECTED_GROUP_SIZE=16
export EXPECTED_PROMPT_MODE=summary_only

SECRET_FILE=".secrets/openrouter_api_key"
SEED_FILE="guidance_ttt/seeds/polyomino_packing/gpt_oss_120b_bootstrap_library.json"
test -s "${SECRET_FILE}"
test "$(stat -c '%a' "${SECRET_FILE}")" = "600"
test -s "${SEED_FILE}"
test -f models/Qwen3-8B/config.json

grep -Eq '^[[:space:]]+num_steps:[[:space:]]+500[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+total_epochs:[[:space:]]+500[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+save_freq:[[:space:]]+1[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+discover_compat:[[:space:]]+true[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+adv_estimator:[[:space:]]+entropic_adaptive_beta[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+groups_per_batch:[[:space:]]+8[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+group_size:[[:space:]]+16[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+prompt_mode:[[:space:]]+summary_only[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+puct_q_mode:[[:space:]]+best_child[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+max_buffer_size:[[:space:]]+1000[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+topk_children:[[:space:]]+2[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+model:[[:space:]]+z-ai/glm-5\.2[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+concurrency:[[:space:]]+16[[:space:]]*$' "${CONFIG}"
grep -Fq 'trainer.max_actor_ckpt_to_keep=1' "${CONFIG}"
grep -Fq 'trainer.max_critic_ckpt_to_keep=1' "${CONFIG}"

prune_to_latest_checkpoint() {
  local checkpoint_root="${REPO_ROOT}/${OUTPUT_DIR}/checkpoints"
  local tracker="${checkpoint_root}/latest_checkpointed_iteration.txt"
  local latest_step
  local latest_dir
  local candidate

  case "${checkpoint_root}" in
    "${REPO_ROOT}"/outputs/guidance_ttt/*/checkpoints) ;;
    *)
      echo "Refusing checkpoint cleanup outside the run output: ${checkpoint_root}" >&2
      return 1
      ;;
  esac
  if [[ ! -s "${tracker}" ]]; then
    return 0
  fi
  latest_step="$(tr -d '[:space:]' < "${tracker}")"
  [[ "${latest_step}" =~ ^[1-9][0-9]*$ ]]
  latest_dir="${checkpoint_root}/global_step_${latest_step}"
  test -d "${latest_dir}/actor"

  while IFS= read -r -d '' candidate; do
    if [[ "${candidate}" != "${latest_dir}" ]]; then
      rm -rf -- "${candidate}"
    fi
  done < <(find "${checkpoint_root}" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' -print0)
  test "$(find "${checkpoint_root}" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' | wc -l)" -eq 1
  echo "Retained latest checkpoint only: ${latest_dir}"
}
trap prune_to_latest_checkpoint EXIT

python - <<'PY'
import json
from pathlib import Path

payload = json.loads(Path("guidance_ttt/seeds/polyomino_packing/gpt_oss_120b_bootstrap_library.json").read_text())
roots = [entry for entry in (payload.get("entries") or {}).values() if int(entry.get("timestep") or 0) == 0]
assert len(roots) == 1, len(roots)
entry = roots[0]
assert entry.get("verifier_status") == "valid"
assert abs(float(entry.get("verifier_raw_score")) - 27.810517627142843) < 1e-9
print("Validated OSS bootstrap seed: score=27.810517627142843")
PY

echo "SLURM_JOB_ID=${SLURM_JOB_ID} MODEL=z-ai/glm-5.2 DISCOVER_COMPAT=true"
echo "OUTPUT_DIR=${OUTPUT_DIR} SHAPE=8x16 CONCURRENCY=16 SAVE_EVERY_STEP=true KEEP_LATEST=1"
echo "Allocation=24h planned training boundary=23h"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

scripts/run_polyomino_b200_long.sh preflight

set +e
timeout --foreground --signal=TERM --kill-after=5m 23h scripts/run_polyomino_b200_long.sh run
status=$?
set -e
if [[ ${status} -eq 0 || ${status} -eq 124 ]]; then
  echo "Experiment completed or reached the planned 23-hour training boundary."
  exit 0
fi
exit "${status}"
