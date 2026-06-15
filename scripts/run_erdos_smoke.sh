#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -m guidance_ttt.main_erdos \
  --config guidance_ttt/config/erdos_smoke.yaml \
  "$@"

