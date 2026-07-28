#!/usr/bin/env bash
#SBATCH --job-name=q3-8b-glm52-g16-s1
#SBATCH --partition=b200-batch
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=384G
#SBATCH --time=08:00:00
#SBATCH --account=ppliang_mit
#SBATCH --output=outputs/slurm/qwen3-8b-openrouter-glm52-g16-1step-%j.out
#SBATCH --error=outputs/slurm/qwen3-8b-openrouter-glm52-g16-1step-%j.err

set -euo pipefail

REPO_ROOT="/work/mit/ppliang_mit/chonghej/open-ttt-verl-guidance-ttt-modal"
cd "${REPO_ROOT}"

export CONFIG="guidance_ttt/config/polyomino_b200_1gpu_qwen3_8b_openrouter_glm52_batch8_group16_summary_only_entropic_best_child_1step_oss_seed_no_ckpt.yaml"
export OUTPUT_DIR="outputs/guidance_ttt/qwen3_8b_openrouter_glm52_g8x16_1step_oss_seed_no_ckpt_${SLURM_JOB_ID}"
export GUIDANCE_MODEL_DIR="models/Qwen3-8B"
export EXPECTED_GPUS=1
export TRAINING_GPUS=0
export START_EXECUTION_SERVER=0
export BOOTSTRAP_BEFORE_RUN=0
export EXPECTED_GROUPS=8
export EXPECTED_GROUP_SIZE=16
export EXPECTED_PROMPT_MODE=summary_only

SECRET_FILE=".secrets/openrouter_api_key"
SEED_FILE="guidance_ttt/seeds/polyomino_packing/gpt_oss_120b_bootstrap_library.json"
test -s "${SECRET_FILE}"
test "$(stat -c '%a' "${SECRET_FILE}")" = "600"
test -s "${SEED_FILE}"
test -f models/Qwen3-8B/config.json

grep -Eq '^[[:space:]]+num_steps:[[:space:]]+1[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+total_epochs:[[:space:]]+1[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+save_freq:[[:space:]]+-1[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+adv_estimator:[[:space:]]+entropic_adaptive_beta[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+groups_per_batch:[[:space:]]+8[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+group_size:[[:space:]]+16[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+prompt_mode:[[:space:]]+summary_only[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+puct_q_mode:[[:space:]]+best_child[[:space:]]*$' "${CONFIG}"
grep -Fq 'seed_library_path: guidance_ttt/seeds/polyomino_packing/gpt_oss_120b_bootstrap_library.json' "${CONFIG}"
grep -Eq '^[[:space:]]+model:[[:space:]]+z-ai/glm-5\.2[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+base_url:[[:space:]]+https://openrouter\.ai/api/v1[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+prompt_style:[[:space:]]+gpt_api_brief_thinking[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+effort:[[:space:]]+high[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+concurrency:[[:space:]]+16[[:space:]]*$' "${CONFIG}"

python - <<'PY'
import json
from pathlib import Path

payload = json.loads(Path("guidance_ttt/seeds/polyomino_packing/gpt_oss_120b_bootstrap_library.json").read_text())
roots = [entry for entry in (payload.get("entries") or {}).values() if int(entry.get("timestep") or 0) == 0]
assert len(roots) == 1, len(roots)
entry = roots[0]
assert (entry.get("metadata") or {}).get("execution_model") == "openai/gpt-oss-120b"
assert entry.get("verifier_status") == "valid"
assert abs(float(entry.get("verifier_raw_score")) - 27.810517627142843) < 1e-9
print("Validated OSS bootstrap seed: score=27.810517627142843")
PY

echo "SLURM_JOB_ID=${SLURM_JOB_ID} MODEL=z-ai/glm-5.2 MODE=summary_only GROUPS=8 GROUP_SIZE=16"
echo "OUTPUT_DIR=${OUTPUT_DIR} OPENROUTER_CONCURRENCY=16 SAVE_FREQ=-1"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

scripts/run_polyomino_b200_long.sh preflight
scripts/run_polyomino_b200_long.sh run

python scripts/validate_openrouter_glm52_smoke.py "${OUTPUT_DIR}" --expected-children 128
python scripts/validate_rollout_actor_parity.py \
  "outputs/slurm/qwen3-8b-openrouter-glm52-g16-1step-${SLURM_JOB_ID}.out"

if find "${OUTPUT_DIR}" -type d \( -name 'global_step_*' -o -name 'checkpoint-*' \) -print -quit | grep -q .; then
  echo "Unexpected checkpoint directory found despite save_freq=-1" >&2
  exit 1
fi
echo "OPENROUTER_GLM52_SMOKE_COMPLETE"
