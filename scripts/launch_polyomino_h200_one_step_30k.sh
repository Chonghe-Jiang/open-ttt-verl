#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

scripts/smoke_polyomino_h200_one_step_cpu.sh
sbatch scripts/slurm_polyomino_h200_4gpu_single_summary.sh "$@"
