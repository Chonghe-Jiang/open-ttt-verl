#!/usr/bin/env bash
#SBATCH --job-name=guidance-poly-b200-setup
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --account=ppliang_mit
#SBATCH --output=outputs/slurm/polyomino-b200-setup-%j.out
#SBATCH --error=outputs/slurm/polyomino-b200-setup-%j.err

set -euo pipefail

REPO_ROOT="/work/mit/ppliang_mit/chonghej/open-ttt-verl-guidance-ttt-modal"
cd "${REPO_ROOT}"
scripts/setup_polyomino_b200.sh
