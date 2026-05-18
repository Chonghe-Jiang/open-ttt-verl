#!/usr/bin/env bash
#SBATCH --job-name=ttt-base-eval
#SBATCH --partition=p5en-odcr-queue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00
#SBATCH --output=/fsx/xuanj/ttt-discover/logs/slurm/%x-%j.out
#SBATCH --error=/fsx/xuanj/ttt-discover/logs/slurm/%x-%j.err
set -euo pipefail

cd /fsx/xuanj/ttt-discover
mkdir -p logs/slurm results/base/eval

source /fsx/xuanj/ttt-discover/.venv/bin/activate
export HF_HOME=/fsx/xuanj/ttt-discover/.hf-cache

echo "[base_eval_docker] node=$(hostname) gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')"
echo "[base_eval_docker] docker ps:"
docker ps 2>&1 | head -3 || true

# Run Docker evals. Some tasks (gemm_optimization, fused_linear_ce, vdb_pareto, llm_sql) need GPU.
# Frontier-CS docker runner detects nvidia-smi and adds --gpus.
python /fsx/xuanj/ttt-discover/scripts/eval_solutions.py \
  --frontier-root /fsx/xuanj/ttt-discover/src/frontier-cs \
  --tasks-json /fsx/xuanj/ttt-discover/bench/tasks_19.json \
  --model-tag dsr1q3_8b_base \
  --num-rollouts 8 \
  --concurrency 4 \
  --timeout 2400 \
  --results-dir /fsx/xuanj/ttt-discover/results/base/eval

echo "[base_eval_docker] all done"
