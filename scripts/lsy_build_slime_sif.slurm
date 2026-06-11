#!/bin/bash
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH -J open-ttt-build-slime-sif
#SBATCH -o /work/mit/ppliang_mit/lsy/open-ttt-workspace/logs/slurm-build-slime-sif-%j.out

set -Eeuo pipefail

cd /work/mit/ppliang_mit/lsy/open-ttt-verl-slime-ttt
source scripts/lsy_apptainer_env.sh

bash scripts/build_slime_sif.sh
