#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_REPO="${MODEL_REPO:-openai/gpt-oss-20b}"
LOCAL_DIR="${LOCAL_DIR:-models/gpt-oss-20b}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

mkdir -p "$(dirname "$LOCAL_DIR")"
hf download "$MODEL_REPO" \
  --local-dir "$LOCAL_DIR" \
  "$@"

echo "Downloaded $MODEL_REPO to $LOCAL_DIR"
