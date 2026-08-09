#!/usr/bin/env bash
#SBATCH --job-name=trimul-glm52-cache-s1
#SBATCH --partition=b200-devel
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=384G
#SBATCH --time=02:00:00
#SBATCH --account=ppliang_mit
#SBATCH --output=outputs/slurm/trimul-glm52-thinking-cache-group4-smoke-%j.out
#SBATCH --error=outputs/slurm/trimul-glm52-thinking-cache-group4-smoke-%j.err

set -euo pipefail

REPO_ROOT="/work/mit/ppliang_mit/chonghej/open-ttt-verl-guidance-ttt-modal"
cd "${REPO_ROOT}"

export CONFIG="guidance_ttt/config/trimul_b200_1gpu_qwen3_8b_evolvent_glm52_thinking_cache_group4_1step.yaml"
export OUTPUT_DIR="outputs/guidance_ttt/trimul_b200_glm52_thinking_cache_group4_smoke_${SLURM_JOB_ID}"
export EXPECTED_GPUS=1
export EXPECTED_CHILDREN=4

grep -Eq '^[[:space:]]+num_steps:[[:space:]]+1[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+groups_per_batch:[[:space:]]+1[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+group_size:[[:space:]]+4[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+adv_estimator:[[:space:]]+entropic_adaptive_beta[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+enable_thinking:[[:space:]]+true[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+cache_prime_first:[[:space:]]+true[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+cache_prime_max_tokens:[[:space:]]+1[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+cache_warm_first:[[:space:]]+false[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+cache_warm_delay_s:[[:space:]]+4\.0[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+concurrency:[[:space:]]+4[[:space:]]*$' "${CONFIG}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID} GPU=B200 MODEL=glm-5.2 THINKING=true CACHE_PRIME=true SHAPE=1x4"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
scripts/run_trimul_b200.sh preflight
scripts/run_trimul_b200.sh run
