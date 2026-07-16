#!/usr/bin/env bash
#SBATCH --job-name=qwen35-9b-b200-setup
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --account=ppliang_mit
#SBATCH --output=outputs/slurm/qwen35-9b-b200-setup-%j.out
#SBATCH --error=outputs/slurm/qwen35-9b-b200-setup-%j.err

set -euo pipefail

REPO_ROOT="/work/mit/ppliang_mit/chonghej/open-ttt-verl-guidance-ttt-modal"
SIF_PATH="${SIF_PATH:-/work/mit/ppliang_mit/chonghej/open-ttt-verl/containers/open-ttt-verl-ttt-vllm.sif}"
MODEL_DIR="${REPO_ROOT}/models/Qwen3.5-9B"
SITE_PACKAGES="${REPO_ROOT}/.runtime/qwen35-site-packages"
PACKED_LORA_PATCH="${REPO_ROOT}/scripts/vllm-0.18.0-qwen35-packed-lora.patch"
cd "${REPO_ROOT}"
mkdir -p outputs/slurm models .runtime .hf_cache .apptainer_cache .apptainer_tmp

export APPTAINER_CACHEDIR="${REPO_ROOT}/.apptainer_cache"
export APPTAINER_TMPDIR="${REPO_ROOT}/.apptainer_tmp"

if [[ ! -f "${MODEL_DIR}/.download-complete" ]]; then
  apptainer exec --cleanenv \
    --bind "${REPO_ROOT}:/workspace/guidance" \
    --pwd /workspace/guidance \
    --env HF_HOME=/workspace/guidance/.hf_cache \
    --env HUGGINGFACE_HUB_CACHE=/workspace/guidance/.hf_cache/hub \
    --env HF_XET_CACHE=/workspace/guidance/.hf_cache/xet \
    --env HF_HUB_ENABLE_HF_TRANSFER=1 \
    --env HF_XET_HIGH_PERFORMANCE=0 \
    --env HF_XET_NUM_CONCURRENT_RANGE_GETS=2 \
    "${SIF_PATH}" hf download Qwen/Qwen3.5-9B \
    --local-dir models/Qwen3.5-9B --max-workers 1 \
    --exclude 'metal/*' 'original/*'
  touch "${MODEL_DIR}/.download-complete"
fi

# Keep the Qwen3.5-specific Python stack isolated from the shared image.  The
# image's vLLM 0.17 predates Qwen3.5's language-model-only and LoRA fixes.
if [[ ! -f "${SITE_PACKAGES}/.transformers-main-complete" ]]; then
  rm -rf "${SITE_PACKAGES}"
  mkdir -p "${SITE_PACKAGES}"
  apptainer exec --cleanenv \
    --bind "${REPO_ROOT}:/workspace/guidance" \
    --pwd /workspace/guidance \
    "${SIF_PATH}" python -m pip install --no-deps \
    --target /workspace/guidance/.runtime/qwen35-site-packages \
    'transformers @ git+https://github.com/huggingface/transformers.git@cd7456ea43677e7936b2429365eda59a09a51e51'
  apptainer exec --cleanenv \
    --bind "${REPO_ROOT}:/workspace/guidance" \
    --pwd /workspace/guidance \
    "${SIF_PATH}" python -m pip install --no-deps --upgrade \
    --target /workspace/guidance/.runtime/qwen35-site-packages \
    'huggingface-hub>=1.5.0,<2.0' \
    'regex>=2025.10.22' \
    'safetensors>=0.8.0' \
    'tokenizers>=0.22.0,<=0.23.0'
  apptainer exec --cleanenv \
    --bind "${REPO_ROOT}:/workspace/guidance" \
    --env PYTHONPATH=/workspace/guidance/.runtime/qwen35-site-packages \
    "${SIF_PATH}" python -c \
    'from transformers import Qwen3_5ForConditionalGeneration; print("Qwen3.5 Transformers import OK")'
  touch "${SITE_PACKAGES}/.transformers-main-complete"
fi

if [[ ! -f "${SITE_PACKAGES}/.vllm-0.18.0-complete" ]]; then
  apptainer exec --cleanenv \
    --bind "${REPO_ROOT}:/workspace/guidance" \
    --pwd /workspace/guidance \
    "${SIF_PATH}" python -m pip install --no-deps --upgrade \
    --target /workspace/guidance/.runtime/qwen35-site-packages \
    'vllm==0.18.0'
  apptainer exec --cleanenv \
    --bind "${REPO_ROOT}:/workspace/guidance" \
    --env PYTHONPATH=/workspace/guidance/.runtime/qwen35-site-packages \
    "${SIF_PATH}" python - <<'PY'
import inspect
import vllm
from vllm.engine.arg_utils import EngineArgs

assert vllm.__version__ == "0.18.0", vllm.__version__
assert "language_model_only" in inspect.signature(EngineArgs).parameters
print(f"vLLM {vllm.__version__} language-model-only support OK")
PY
  touch "${SITE_PACKAGES}/.vllm-0.18.0-complete"
fi

# vLLM 0.18.0 declares Qwen3.5's two logical GDN adapters but its fused layer
# has four physical slices. Backport the upstream expansion fix so all-linear
# LoRA remains enabled instead of dropping the recurrent-attention adapters.
LORA_LAYER="${SITE_PACKAGES}/vllm/lora/layers/column_parallel_linear.py"
LORA_MANAGER="${SITE_PACKAGES}/vllm/lora/model_manager.py"
if ! grep -q 'def expand_packed_lora' "${LORA_LAYER}" || \
   ! grep -q 'n_slices != len(replacements)' "${LORA_MANAGER}"; then
  patch --batch --forward -d "${SITE_PACKAGES}" -p1 < "${PACKED_LORA_PATCH}" || true
fi
grep -q 'def expand_packed_lora' "${LORA_LAYER}"
grep -q 'n_slices != len(replacements)' "${LORA_MANAGER}"
touch "${SITE_PACKAGES}/.qwen35-packed-lora-complete"

echo "Qwen3.5-9B B200 runtime is ready."
