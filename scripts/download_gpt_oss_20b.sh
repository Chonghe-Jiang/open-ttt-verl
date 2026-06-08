#!/bin/bash
set -euo pipefail

WORKSPACE=${WORKSPACE:-${HOME}/scratch/open-ttt-workspace}
MODEL_ROOT=${MODEL_ROOT:-${WORKSPACE}/models}
LOG_ROOT=${LOG_ROOT:-${WORKSPACE}/logs}
MODEL_ID=${MODEL_ID:-openai/gpt-oss-20b}
MODEL_DIR=${MODEL_DIR:-${MODEL_ROOT}/gpt-oss-20b}
FORCE_DOWNLOAD=${FORCE_DOWNLOAD:-0}
HF_VENV=${HF_VENV:-${WORKSPACE}/venvs/hf-download}
HF_PYTHON=${HF_PYTHON:-}
HF_MAX_WORKERS=${HF_MAX_WORKERS:-1}

has_model_weights() {
  find "${MODEL_DIR}" -maxdepth 1 -type f \( -name '*.safetensors' -o -name 'pytorch_model*.bin' \) -print -quit 2>/dev/null | grep -q .
}

mkdir -p "${MODEL_ROOT}" "${LOG_ROOT}"
LOG_FILE=${LOG_FILE:-${LOG_ROOT}/download_gpt_oss_20b_$(date +%Y%m%d_%H%M%S).log}
touch "${LOG_FILE}"
rm -f "${LOG_ROOT}/download_gpt_oss_20b.latest.log"
ln -s "${LOG_FILE}" "${LOG_ROOT}/download_gpt_oss_20b.latest.log"
exec >> "${LOG_FILE}" 2>&1

echo "Logging to ${LOG_FILE}"
echo "Model id: ${MODEL_ID}"
echo "Model dir: ${MODEL_DIR}"

if [ -n "${HF_TOKEN:-}" ]; then
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

if [ "${FORCE_DOWNLOAD}" != "1" ] && [ -f "${MODEL_DIR}/config.json" ] && has_model_weights; then
  echo "Raw model already exists at ${MODEL_DIR}; set FORCE_DOWNLOAD=1 to download again."
  exit 0
fi

if [ -f "${MODEL_DIR}/config.json" ] && ! has_model_weights; then
  echo "Found partial model directory without weight shards; resuming download."
fi

if command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "${MODEL_ID}" \
    --local-dir "${MODEL_DIR}"
else
  if [ -z "${HF_PYTHON}" ]; then
    if [ -x "${HF_VENV}/bin/python" ]; then
      HF_PYTHON="${HF_VENV}/bin/python"
    elif command -v python3.12 >/dev/null 2>&1; then
      python3.12 -m venv "${HF_VENV}"
      HF_PYTHON="${HF_VENV}/bin/python"
    elif [ -x "${HOME}/miniconda3/bin/python" ]; then
      HF_PYTHON="${HOME}/miniconda3/bin/python"
    else
      HF_PYTHON="$(command -v python3)"
    fi
  fi

  if ! "${HF_PYTHON}" -c "import huggingface_hub" >/dev/null 2>&1; then
    "${HF_PYTHON}" -m pip install --upgrade pip
    "${HF_PYTHON}" -m pip install huggingface_hub
  fi

  export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
  MODEL_ID="${MODEL_ID}" MODEL_DIR="${MODEL_DIR}" HF_MAX_WORKERS="${HF_MAX_WORKERS}" "${HF_PYTHON}" - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["MODEL_ID"],
    local_dir=os.environ["MODEL_DIR"],
    token=os.environ.get("HF_TOKEN") or None,
    max_workers=int(os.environ.get("HF_MAX_WORKERS", "1")),
    resume_download=True,
)
PY
fi

test -f "${MODEL_DIR}/config.json"
test -f "${MODEL_DIR}/tokenizer_config.json" -o -f "${MODEL_DIR}/tokenizer.json"
has_model_weights

echo "Downloaded ${MODEL_ID} to ${MODEL_DIR}"
