#!/usr/bin/env bash
#SBATCH --job-name=guidance-entropic-best-child
#SBATCH --partition=b200-batch
#SBATCH --gres=gpu:b200:5
#SBATCH --cpus-per-task=96
#SBATCH --mem=768G
#SBATCH --time=1-00:00:00
#SBATCH --account=ppliang_mit
#SBATCH --output=outputs/slurm/entropic-best-child-%j.out
#SBATCH --error=outputs/slurm/entropic-best-child-%j.err

set -euo pipefail

REPO_ROOT="/work/mit/ppliang_mit/chonghej/open-ttt-verl-guidance-ttt-modal"
cd "${REPO_ROOT}"

: "${CONFIG:?submit with CONFIG pointing to one entropic best-child recipe}"
: "${OUTPUT_DIR:?submit with a fresh OUTPUT_DIR}"
: "${EXPECTED_GROUP_SIZE:?submit with EXPECTED_GROUP_SIZE}"
: "${EXPECTED_PROMPT_MODE:?submit with EXPECTED_PROMPT_MODE}"

export EXPECTED_GPUS=5
export TRAINING_GPUS=0,1,2,3
export EXECUTION_GPU=4
STAGE="${STAGE:-direct}"

case "${STAGE}" in
  first)
    export NUM_STEPS_OVERRIDE=20
    export TOTAL_EPOCHS_OVERRIDE=20
    export EXPECTED_GROUPS=160
    ;;
  final|direct)
    unset NUM_STEPS_OVERRIDE TOTAL_EPOCHS_OVERRIDE
    export EXPECTED_GROUPS=400
    ;;
  *)
    echo "Unknown STAGE=${STAGE}; expected first, final, or direct" >&2
    exit 2
    ;;
esac

grep -Eq '^[[:space:]]+adv_estimator:[[:space:]]+entropic_adaptive_beta[[:space:]]*$' "${CONFIG}"
grep -Eq '^[[:space:]]+puct_q_mode:[[:space:]]+best_child[[:space:]]*$' "${CONFIG}"

if [[ "${CONFIG}" == *qwen3_14b* ]]; then
  test -f models/Qwen3-14B/.download-complete
fi

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "STAGE=${STAGE} CONFIG=${CONFIG} OUTPUT_DIR=${OUTPUT_DIR}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

scripts/run_polyomino_b200_long.sh preflight
scripts/run_polyomino_b200_long.sh run
