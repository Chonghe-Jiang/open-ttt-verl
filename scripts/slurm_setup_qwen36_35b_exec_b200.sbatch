#!/usr/bin/env bash
#SBATCH --job-name=qwen36-35b-exec-setup
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --account=ppliang_mit
#SBATCH --output=outputs/slurm/qwen36-35b-exec-setup-%j.out
#SBATCH --error=outputs/slurm/qwen36-35b-exec-setup-%j.err

set -euo pipefail

REPO_ROOT="/work/mit/ppliang_mit/chonghej/open-ttt-verl-guidance-ttt-modal"
SIF_PATH="${SIF_PATH:-/work/mit/ppliang_mit/chonghej/open-ttt-verl/containers/open-ttt-verl-ttt-vllm.sif}"
MODEL_DIR="${REPO_ROOT}/models/Qwen3.6-35B-A3B"
SITE_PACKAGES="${REPO_ROOT}/.runtime/qwen36-exec-site-packages"
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
    "${SIF_PATH}" hf download Qwen/Qwen3.6-35B-A3B \
    --local-dir models/Qwen3.6-35B-A3B --max-workers 1 \
    --exclude 'metal/*' 'original/*'
  touch "${MODEL_DIR}/.download-complete"
fi

# Keep the execution-only serving stack isolated. Qwen3.6 officially recommends
# vLLM >= 0.19.0; the guidance actor continues using the image's verl-compatible
# vLLM instead of importing this directory.
if [[ ! -f "${SITE_PACKAGES}/.vllm-0.19.0-complete" ]]; then
  rm -rf "${SITE_PACKAGES}"
  mkdir -p "${SITE_PACKAGES}"
  apptainer exec --cleanenv \
    --bind "${REPO_ROOT}:/workspace/guidance" \
    --pwd /workspace/guidance \
    "${SIF_PATH}" python -m pip install --no-deps \
    --target /workspace/guidance/.runtime/qwen36-exec-site-packages \
    'transformers @ git+https://github.com/huggingface/transformers.git@cd7456ea43677e7936b2429365eda59a09a51e51'
  apptainer exec --cleanenv \
    --bind "${REPO_ROOT}:/workspace/guidance" \
    --pwd /workspace/guidance \
    "${SIF_PATH}" python -m pip install --no-deps --upgrade \
    --target /workspace/guidance/.runtime/qwen36-exec-site-packages \
    'huggingface-hub>=1.5.0,<2.0' \
    'regex>=2025.10.22' \
    'safetensors>=0.8.0' \
    'tokenizers>=0.22.0,<=0.23.0' \
    'vllm==0.19.0'
  apptainer exec --cleanenv \
    --bind "${REPO_ROOT}:/workspace/guidance" \
    --env PYTHONPATH=/workspace/guidance/.runtime/qwen36-exec-site-packages \
    "${SIF_PATH}" python - <<'PY'
import inspect
import vllm
from vllm.engine.arg_utils import EngineArgs

assert vllm.__version__ == "0.19.0", vllm.__version__
assert "language_model_only" in inspect.signature(EngineArgs).parameters
print(f"vLLM {vllm.__version__} Qwen3.6 execution runtime import OK")
PY
  touch "${SITE_PACKAGES}/.vllm-0.19.0-complete"
fi

# vLLM 0.19 imports ReasoningEffort while formatting every OpenAI chat
# completion.  The base image ships mistral-common 1.9.1, while the pinned
# Transformers Qwen3.6 revision requires >=1.11.5.  Keep the compatible
# override in the execution-only runtime so the verl actor stays unchanged.
if [[ ! -f "${SITE_PACKAGES}/.mistral-common-1.11.5-complete" ]]; then
  # pip --target --upgrade leaves older *.dist-info directories behind. That
  # makes importlib.metadata select the stale version even when newer package
  # code was copied, so remove both code and metadata before reinstalling.
  rm -rf "${SITE_PACKAGES}/mistral_common" "${SITE_PACKAGES}"/mistral_common-*.dist-info
  apptainer exec --cleanenv \
    --bind "${REPO_ROOT}:/workspace/guidance" \
    --pwd /workspace/guidance \
    "${SIF_PATH}" python -m pip install --no-deps --upgrade \
    --target /workspace/guidance/.runtime/qwen36-exec-site-packages \
    'mistral-common==1.11.5'
  apptainer exec --cleanenv \
    --bind "${REPO_ROOT}:/workspace/guidance" \
    --env PYTHONPATH=/workspace/guidance/.runtime/qwen36-exec-site-packages \
    "${SIF_PATH}" python - <<'PY'
import importlib.metadata
from pathlib import Path
from mistral_common.protocol.instruct.request import ReasoningEffort
from transformers.tokenization_mistral_common import MistralCommonBackend
from transformers.utils.import_utils import is_mistral_common_available
from vllm.tokenizers.mistral import MistralTokenizer

assert importlib.metadata.version("mistral-common") == "1.11.5"
site = Path("/workspace/guidance/.runtime/qwen36-exec-site-packages")
assert [p.name for p in site.glob("mistral_common-*.dist-info")] == ["mistral_common-1.11.5.dist-info"]
assert ReasoningEffort is not None
assert MistralCommonBackend is not None
assert MistralTokenizer is not None
assert is_mistral_common_available()
print("mistral-common 1.11.5 chat-completion imports OK")
PY
  touch "${SITE_PACKAGES}/.mistral-common-1.11.5-complete"
fi

touch "${SITE_PACKAGES}/.qwen36-exec-complete"
echo "Qwen3.6-35B-A3B execution runtime is ready."
