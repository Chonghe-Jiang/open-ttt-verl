#!/usr/bin/env bash
set -euo pipefail

# Copies datasets for llm_sql_large problem to local datasets folder

PROBLEM_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# Go up from problem_dir to problems/, then to repo root
BASE_DIR=$(cd "$PROBLEM_DIR/../../.." && pwd)
DATASETS_DIR="$BASE_DIR/datasets/llm_sql/large"

mkdir -p "$DATASETS_DIR"

echo "[llm_sql/large download] Checking for datasets..."

# Check if dataset already exists
if [[ -d "$DATASETS_DIR" ]] && [[ -n $(ls -A "$DATASETS_DIR" 2>/dev/null) ]]; then
  echo "[llm_sql/large download] Dataset already exists at $DATASETS_DIR"
  exit 0
fi

# Copy datasets from problem resources
SRC_DIR="$PROBLEM_DIR/resources/datasets"

if [[ ! -d "$SRC_DIR" ]]; then
  echo "Error: source dataset directory not found at $SRC_DIR" >&2
  exit 1
fi

echo "[llm_sql/large download] Copying datasets..."
cp -r "$SRC_DIR"/* "$DATASETS_DIR/"

echo "[llm_sql/large download] Dataset ready at $DATASETS_DIR"


