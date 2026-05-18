#!/usr/bin/env bash
#SBATCH --job-name=no-shell
#SBATCH --partition=p5en-odcr-queue
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=14:00:00
#SBATCH --output=/fsx/xuanj/ttt-discover/logs/slurm/%x-%j.out
#SBATCH --error=/fsx/xuanj/ttt-discover/logs/slurm/%x-%j.err
set -euo pipefail

cd /fsx/xuanj/ttt-discover
mkdir -p logs/slurm logs/vllm results/ttt_faithful results/ttt_faithful_lora scratch

NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
NODE_ARR=($NODES)
JOBID=${SLURM_JOB_ID:-local}
echo "[ttt-sweep-4pair] master=$(hostname) nodes=${NODE_ARR[@]}"

# 16 tasks (excluding cbl_multi__low which we already validated). Run 4 pairs in parallel.
# Each pair: 1 vLLM node + 1 training node.
TASKS=(
  cbl__high_av_loose_dl_small_oh        # ID, base 0
  cbl__low_av_tight_dl_large_oh         # ID, base 0.16
  cbl__mixed_av_loose_dl_large_oh       # ID, base 0.33
  cbl_multi__high_av_loose_dl_small_oh  # ID, base 0
  cbl_multi__high_av_tight_dl_small_oh  # ID, base 0.27
  fused_linear_ce
  gemm_opt__annoying
  gemm_opt__k_skewed
  gemm_opt__rectangles
  gemm_opt__squares
  gemm_opt__transformerish
  llm_sql__large
  poc_gen__heap_uaf
  poc_gen__uninit_value
  vdb_pareto__low_latency
  vdb_pareto__recall80_lat
)

# 4 pairs (uses 8 nodes), each pair processes ~4 tasks sequentially
NUM_PAIRS=4
for pair in $(seq 0 $((NUM_PAIRS - 1))); do
  VLLM_NODE=${NODE_ARR[$((pair * 2))]}
  TRAIN_NODE=${NODE_ARR[$((pair * 2 + 1))]}
  PORT=$((8000 + pair))
  echo "[ttt-sweep-4pair] pair $pair: vllm=$VLLM_NODE train=$TRAIN_NODE port=$PORT"

  # Per-pair tasks: tasks where (idx % NUM_PAIRS) == pair
  PAIR_TASKS=()
  for idx in $(seq 0 $((${#TASKS[@]} - 1))); do
    if [ $((idx % NUM_PAIRS)) -eq $pair ]; then
      PAIR_TASKS+=("${TASKS[$idx]}")
    fi
  done
  PAIR_TASK_LIST=$(IFS=, ; echo "${PAIR_TASKS[*]}")

  # Launch the pair (vLLM + sequential trainer over PAIR_TASKS)
  bash /fsx/xuanj/ttt-discover/scripts/ttt_faithful_pair.sh \
    "$VLLM_NODE" "$TRAIN_NODE" "$PORT" "$PAIR_TASK_LIST" "$JOBID" "$pair" &
done

wait
echo "[ttt-sweep-4pair] all pairs done"
