#!/usr/bin/env bash
#SBATCH --job-name=guidance-qwen3-14b-download
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --account=ppliang_mit
#SBATCH --output=outputs/slurm/qwen3-14b-download-%j.out
#SBATCH --error=outputs/slurm/qwen3-14b-download-%j.err

set -euo pipefail

REPO_ROOT="/work/mit/ppliang_mit/chonghej/open-ttt-verl-guidance-ttt-modal"
SIF_PATH="/work/mit/ppliang_mit/chonghej/open-ttt-verl/containers/open-ttt-verl-ttt-vllm.sif"
cd "${REPO_ROOT}"

export APPTAINER_CACHEDIR="${REPO_ROOT}/.apptainer_cache"
export APPTAINER_TMPDIR="${REPO_ROOT}/.apptainer_tmp"
mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}" models/Qwen3-14B .hf_cache

if [[ ! -f models/Qwen3-14B/.download-complete ]]; then
  apptainer exec --cleanenv \
    --bind "${REPO_ROOT}:/workspace/guidance" \
    --pwd /workspace/guidance \
    --env HF_HOME=/workspace/guidance/.hf_cache \
    --env HUGGINGFACE_HUB_CACHE=/workspace/guidance/.hf_cache/hub \
    --env HF_XET_CACHE=/workspace/guidance/.hf_cache/xet \
    --env HF_HUB_DISABLE_XET=1 \
    --env HF_HUB_ENABLE_HF_TRANSFER=0 \
    "${SIF_PATH}" hf download Qwen/Qwen3-14B \
    --local-dir models/Qwen3-14B --max-workers 1
  touch models/Qwen3-14B/.download-complete
fi

test -s models/Qwen3-14B/config.json
echo "Qwen3-14B ready at ${REPO_ROOT}/models/Qwen3-14B"
