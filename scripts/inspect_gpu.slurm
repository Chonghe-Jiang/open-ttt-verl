#!/bin/bash
#SBATCH -p pi_ppliang
#SBATCH --gres=gpu:1
#SBATCH -c 15
#SBATCH --mem=65G
#SBATCH --time=00:30:00
#SBATCH -J open-ttt-inspect-gpu
#SBATCH -o /work/mit/ppliang_mit/lsy/open-ttt-workspace/logs/slurm-inspect-gpu-%j.out

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

echo "=== slurm inspect gpu ==="
echo "time: $(date -Is)"
echo "host: $(hostname)"
echo "job: ${SLURM_JOB_ID:-unset}"
echo "partition: ${SLURM_JOB_PARTITION:-unset}"
echo "gres: ${SLURM_JOB_GRES:-unset}"
echo "gpus_on_node: ${SLURM_GPUS_ON_NODE:-unset}"
echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-unset}"

bash scripts/ondemand_inspect_slime_container_job.sh
