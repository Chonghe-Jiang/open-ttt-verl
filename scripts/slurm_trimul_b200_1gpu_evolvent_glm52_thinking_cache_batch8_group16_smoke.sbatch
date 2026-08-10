#!/usr/bin/env bash
#SBATCH --job-name=trimul-g8x16-s1
#SBATCH --partition=b200-devel
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=384G
#SBATCH --time=02:00:00
#SBATCH --account=ppliang_mit
#SBATCH --output=outputs/slurm/trimul-glm52-thinking-cache-g8x16-smoke-%j.out
#SBATCH --error=outputs/slurm/trimul-glm52-thinking-cache-g8x16-smoke-%j.err

set -euo pipefail

REPO_ROOT="/work/mit/ppliang_mit/chonghej/open-ttt-verl-guidance-ttt-modal"
cd "${REPO_ROOT}"

export CONFIG="guidance_ttt/config/trimul_b200_1gpu_qwen3_8b_evolvent_glm52_thinking_cache_batch8_group16_1step.yaml"
export OUTPUT_DIR="outputs/guidance_ttt/trimul_b200_glm52_thinking_cache_g8x16_smoke_${SLURM_JOB_ID}"
export EXPECTED_GPUS=1
export EXPECTED_GROUPS=8
export EXPECTED_GROUP_SIZE=16
export EXPECTED_CHILDREN=128
export MINIMUM_CACHE_HITS=16
export REQUIRE_TRAINING_UPDATE=1

grep -Eq '^[[:space:]]+num_steps:[[:space:]]+1[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+total_epochs:[[:space:]]+1[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+n_gpus_per_node:[[:space:]]+1[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+ppo_mini_batch_size:[[:space:]]+8[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+ppo_micro_batch_size_per_gpu:[[:space:]]+2[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+learning_rate:[[:space:]]+4\.0e-5[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+use_kl_loss:[[:space:]]+false[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+discover_compat:[[:space:]]+true[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+groups_per_batch:[[:space:]]+8[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+group_size:[[:space:]]+16[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+reasoning_effort:[[:space:]]+high[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+-[[:space:]]+reasoning_effort[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+max_tokens:[[:space:]]+81920[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+concurrency:[[:space:]]+16[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+concurrency:[[:space:]]+32[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+cache_prime_first:[[:space:]]+true[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+save_freq:[[:space:]]+-1[[:space:]]*$' "${CONFIG}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID} GPU=B200 MODEL=glm-5.2 REASONING_EFFORT=high SHAPE=8x16 EXEC_CONCURRENCY=32 EVAL_CONCURRENCY=16"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
scripts/run_trimul_b200.sh preflight
scripts/run_trimul_b200.sh run
