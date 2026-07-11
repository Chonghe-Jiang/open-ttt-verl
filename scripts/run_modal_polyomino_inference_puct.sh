#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

modal run --detach scripts/modal_polyomino_h200_smoke.py --action run_gpt_oss_120b_inference_puct "$@"
