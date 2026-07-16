#!/usr/bin/env bash
#SBATCH --job-name=qwen36-27b-guidance-setup
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --account=ppliang_mit
#SBATCH --output=outputs/slurm/qwen36-27b-guidance-setup-%j.out
#SBATCH --error=outputs/slurm/qwen36-27b-guidance-setup-%j.err

set -euo pipefail

REPO_ROOT="/work/mit/ppliang_mit/chonghej/open-ttt-verl-guidance-ttt-modal"
SIF_PATH="${SIF_PATH:-/work/mit/ppliang_mit/chonghej/open-ttt-verl/containers/open-ttt-verl-ttt-vllm.sif}"
MODEL_DIR="${REPO_ROOT}/models/Qwen3.6-27B"
SITE_PACKAGES="${REPO_ROOT}/.runtime/qwen36-actor-site-packages"
PACKED_LORA_PATCH="${REPO_ROOT}/scripts/vllm-0.18.0-qwen35-packed-lora.patch"
FLA_ALLOCATOR_PATCH="${REPO_ROOT}/scripts/vllm-0.19.0-fla-triton-allocator.patch"
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
    --env HF_XET_HIGH_PERFORMANCE=0 \
    --env HF_XET_NUM_CONCURRENT_RANGE_GETS=2 \
    "${SIF_PATH}" hf download Qwen/Qwen3.6-27B \
    --local-dir models/Qwen3.6-27B --max-workers 1 \
    --exclude 'metal/*' 'original/*'
  touch "${MODEL_DIR}/.download-complete"
fi

if [[ ! -f "${SITE_PACKAGES}/.vllm-0.19.0-complete" ]]; then
  rm -rf "${SITE_PACKAGES}"
  mkdir -p "${SITE_PACKAGES}"
  apptainer exec --cleanenv \
    --bind "${REPO_ROOT}:/workspace/guidance" \
    --pwd /workspace/guidance \
    "${SIF_PATH}" python -m pip install --no-deps \
    --target /workspace/guidance/.runtime/qwen36-actor-site-packages \
    'transformers @ git+https://github.com/huggingface/transformers.git@cd7456ea43677e7936b2429365eda59a09a51e51'
  apptainer exec --cleanenv \
    --bind "${REPO_ROOT}:/workspace/guidance" \
    --pwd /workspace/guidance \
    "${SIF_PATH}" python -m pip install --no-deps --upgrade \
    --target /workspace/guidance/.runtime/qwen36-actor-site-packages \
    'huggingface-hub>=1.5.0,<2.0' \
    'regex>=2025.10.22' \
    'safetensors>=0.8.0' \
    'tokenizers>=0.22.0,<=0.23.0' \
    'vllm==0.19.0'
  touch "${SITE_PACKAGES}/.vllm-0.19.0-complete"
fi

# Qwen3.6's GDN projection is physically split into four vLLM slices while
# PEFT exposes two logical adapters.  The same backport used by the validated
# Qwen3.5 actor remains necessary for vLLM 0.19.
LORA_LAYER="${SITE_PACKAGES}/vllm/lora/layers/column_parallel_linear.py"
LORA_MANAGER="${SITE_PACKAGES}/vllm/lora/model_manager.py"
if ! grep -q 'def expand_packed_lora' "${LORA_LAYER}" || \
   ! grep -q 'n_slices != len(replacements)' "${LORA_MANAGER}"; then
  patch --batch --forward -d "${SITE_PACKAGES}" -p1 < "${PACKED_LORA_PATCH}"
fi

# Install a Triton allocator before the first GDN prefill.  Without this,
# solve_tril aborts at runtime when its autotuned kernel requests scratch.
FLA_UTILS="${SITE_PACKAGES}/vllm/model_executor/layers/fla/ops/utils.py"
if ! grep -q 'set_triton_allocator(tensor.device)' "${FLA_UTILS}"; then
  patch --batch --forward -d "${SITE_PACKAGES}" -p1 < "${FLA_ALLOCATOR_PATCH}"
fi

apptainer exec --cleanenv \
  --bind "${REPO_ROOT}:/workspace/guidance" \
  --env PYTHONPATH=/workspace/guidance/.runtime/qwen36-actor-site-packages \
  "${SIF_PATH}" python - <<'PY'
import inspect
import flash_attn
import vllm
from transformers import Qwen3_5ForConditionalGeneration
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.fla.ops.utils import input_guard

assert vllm.__version__ == "0.19.0", vllm.__version__
assert tuple(int(part) for part in flash_attn.__version__.split(".")[:2]) >= (2, 0), flash_attn.__version__
assert "language_model_only" in inspect.signature(EngineArgs).parameters
assert Qwen3_5ForConditionalGeneration is not None
assert input_guard is not None
print(f"Qwen3.6-27B actor runtime imports OK (flash-attn {flash_attn.__version__})")
PY

grep -q 'def expand_packed_lora' "${LORA_LAYER}"
grep -q 'n_slices != len(replacements)' "${LORA_MANAGER}"
grep -q 'set_triton_allocator(tensor.device)' "${FLA_UTILS}"
touch "${SITE_PACKAGES}/.qwen36-actor-complete"
touch "${SITE_PACKAGES}/.qwen36-actor-fa2-nopad-complete"
echo "Qwen3.6-27B guidance model and actor runtime are ready."
