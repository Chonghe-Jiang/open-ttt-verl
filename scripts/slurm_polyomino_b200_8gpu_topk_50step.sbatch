#!/usr/bin/env bash
#SBATCH --job-name=topk-guidance-8gpu
#SBATCH --partition=b200-batch
#SBATCH --gres=gpu:b200:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=1024G
#SBATCH --time=1-00:00:00
#SBATCH --account=ppliang_mit
#SBATCH --output=outputs/slurm/topk-guidance-8gpu-%j.out
#SBATCH --error=outputs/slurm/topk-guidance-8gpu-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/work/mit/ppliang_mit/lsy/topk-guidance}"
cd "${REPO_ROOT}"

: "${CONFIG:?submit with CONFIG pointing to the common 8-GPU recipe}"
: "${OUTPUT_DIR:?submit with a fresh OUTPUT_DIR}"
: "${EXPECTED_GROUP_SIZE:?submit with EXPECTED_GROUP_SIZE}"
: "${EXPECTED_PROMPT_MODE:?submit with EXPECTED_PROMPT_MODE}"

export EXPECTED_GPUS="${EXPECTED_GPUS:-8}"
export TRAINING_GPUS="${TRAINING_GPUS:-0,1,2,3,4,5,6}"
export EXECUTION_GPU="${EXECUTION_GPU:-7}"
export STAGE="${STAGE:-direct}"
export EXPECTED_GROUPS="${EXPECTED_GROUPS:-400}"
export WRITE_TOPK_METRICS="${WRITE_TOPK_METRICS:-true}"

if [[ "${EXPECTED_GPUS}" != "8" || "${TRAINING_GPUS}" != "0,1,2,3,4,5,6" || "${EXECUTION_GPU}" != "7" ]]; then
  echo "This recipe reserves 7 B200 GPUs for TTT and GPU 7 for GPT-OSS execution." >&2
  exit 2
fi

grep -Eq '^[[:space:]]+adv_estimator:[[:space:]]+entropic_adaptive_beta[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+puct_q_mode:[[:space:]]+best_child[[:space:]]*$' "${CONFIG}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "CONFIG=${CONFIG} OUTPUT_DIR=${OUTPUT_DIR}"
echo "EXTRA_RECIPE_OVERRIDES=${EXTRA_RECIPE_OVERRIDES:-}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

scripts/run_polyomino_b200_long.sh preflight
scripts/run_polyomino_b200_long.sh run
