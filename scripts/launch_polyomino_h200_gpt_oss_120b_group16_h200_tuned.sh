#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SMOKE_ONLY=1 scripts/run_local_polyomino_h200_gpt_oss_120b_group16.sh
sbatch scripts/slurm_polyomino_h200_gpt_oss_120b_group16_h200_tuned.sh "$@"
