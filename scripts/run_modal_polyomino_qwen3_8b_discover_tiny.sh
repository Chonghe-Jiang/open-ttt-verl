#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

modal run --detach scripts/modal_polyomino_h200_smoke.py --action train_qwen3_8b_discover_tiny "$@"
