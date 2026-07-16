#!/usr/bin/env bash
#SBATCH --job-name=q3-coder-next-fp8-setup
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --account=ppliang_mit
#SBATCH --output=outputs/slurm/qwen3-coder-next-fp8-setup-%j.out
#SBATCH --error=outputs/slurm/qwen3-coder-next-fp8-setup-%j.err

set -euo pipefail

REPO_ROOT="/work/mit/ppliang_mit/chonghej/open-ttt-verl-guidance-ttt-modal"
SIF_PATH="${SIF_PATH:-/work/mit/ppliang_mit/chonghej/open-ttt-verl/containers/open-ttt-verl-ttt-vllm.sif}"
MODEL_DIR="${REPO_ROOT}/models/Qwen3-Coder-Next-FP8"
SITE_PACKAGES="${REPO_ROOT}/.runtime/qwen36-exec-site-packages"
cd "${REPO_ROOT}"
mkdir -p outputs/slurm models .hf_cache .apptainer_cache .apptainer_tmp

export APPTAINER_CACHEDIR="${REPO_ROOT}/.apptainer_cache"
export APPTAINER_TMPDIR="${REPO_ROOT}/.apptainer_tmp"

test -f "${SITE_PACKAGES}/.vllm-0.19.0-complete"
test -f "${SITE_PACKAGES}/.mistral-common-1.11.5-complete"

if [[ ! -f "${MODEL_DIR}/.download-complete" ]]; then
  apptainer exec --cleanenv \
    --bind "${REPO_ROOT}:/workspace/guidance" \
    --pwd /workspace/guidance \
    --env HF_HOME=/workspace/guidance/.hf_cache \
    --env HUGGINGFACE_HUB_CACHE=/workspace/guidance/.hf_cache/hub \
    --env HF_XET_CACHE=/workspace/guidance/.hf_cache/xet \
    --env HF_XET_HIGH_PERFORMANCE=0 \
    --env HF_XET_NUM_CONCURRENT_RANGE_GETS=4 \
    "${SIF_PATH}" hf download Qwen/Qwen3-Coder-Next-FP8 \
    --local-dir models/Qwen3-Coder-Next-FP8 --max-workers 2 \
    --exclude '*.gguf' 'original/*'
fi

apptainer exec --cleanenv \
  --bind "${REPO_ROOT}:/workspace/guidance" \
  --pwd /workspace/guidance \
  --env PYTHONPATH=/workspace/guidance/.runtime/qwen36-exec-site-packages \
  "${SIF_PATH}" python - <<'PY'
from transformers import AutoConfig
from vllm.model_executor.models.registry import ModelRegistry

config = AutoConfig.from_pretrained("models/Qwen3-Coder-Next-FP8", trust_remote_code=True)
architectures = set(config.architectures or [])
assert "Qwen3NextForCausalLM" in architectures, architectures
assert "Qwen3NextForCausalLM" in ModelRegistry.get_supported_archs()
quantization = getattr(config, "quantization_config", {}) or {}
assert quantization.get("quant_method") == "fp8", quantization
print("Qwen3-Coder-Next-FP8 config and vLLM 0.19 registry validation OK")
PY

touch "${MODEL_DIR}/.download-complete"
echo "Qwen3-Coder-Next-FP8 execution model is ready."
