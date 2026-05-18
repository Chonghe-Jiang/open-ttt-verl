#!/usr/bin/env bash
#SBATCH --job-name=ttt-discover-4n
#SBATCH --partition=p5en-odcr-queue
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00
#SBATCH --output=/fsx/xuanj/ttt-discover/logs/slurm/%x-%j.out
#SBATCH --error=/fsx/xuanj/ttt-discover/logs/slurm/%x-%j.err
set -euo pipefail

cd /fsx/xuanj/ttt-discover
mkdir -p logs/slurm logs/vllm results/ttt scratch

source /fsx/xuanj/ttt-discover/.venv/bin/activate
export HF_HOME=/fsx/xuanj/ttt-discover/.hf-cache
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export FLASHINFER_DISABLE_VERSION_CHECK=1
export OMP_NUM_THREADS=4

# Each node runs:
#   1. its own vLLM server on TP=8 for the base model
#   2. assigned subset of the 19 tasks via TTT-Discover (--no-train context-only first pass for budget)
#
# Task assignment: shard mod 4 by index in the 19-task list.

# Build task name list
TASKS=(
  cbl__high_av_loose_dl_small_oh
  cbl__low_av_tight_dl_large_oh
  cbl__mixed_av_loose_dl_large_oh
  cbl_multi__high_av_loose_dl_small_oh
  cbl_multi__high_av_tight_dl_small_oh
  cbl_multi__low_av_loose_dl_small_oh
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

echo "[ttt-4n] master=$(hostname) nodelist=$SLURM_JOB_NODELIST"

for shard in 0 1 2 3 ; do
  srun --nodes=1 --ntasks=1 --relative=$shard --exclusive \
    bash -c "
      set -euo pipefail
      source /fsx/xuanj/ttt-discover/.venv/bin/activate
      export HF_HOME=/fsx/xuanj/ttt-discover/.hf-cache
      export VLLM_ATTENTION_BACKEND=FLASH_ATTN
      export FLASHINFER_DISABLE_VERSION_CHECK=1
      export OMP_NUM_THREADS=4
      NODE=\$(hostname)
      echo \"[shard $shard node=\$NODE] starting vllm\"
      LOG_DIR=/fsx/xuanj/ttt-discover/logs/vllm
      mkdir -p \$LOG_DIR
      nohup python -m vllm.entrypoints.openai.api_server \
        --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
        --port $((8000 + $shard)) --host 127.0.0.1 \
        --tensor-parallel-size 8 \
        --gpu-memory-utilization 0.85 \
        --max-model-len 32768 \
        --dtype bfloat16 \
        --disable-log-requests \
        > \$LOG_DIR/ttt_shard${shard}_\${SLURM_JOB_ID}.log 2>&1 &
      VLLM_PID=\$!
      for i in \$(seq 1 96); do
        if curl -sf http://127.0.0.1:$((8000 + $shard))/health > /dev/null 2>&1; then
          echo \"[shard $shard] vllm up after \$((i*5))s\"
          break
        fi
        sleep 5
      done
      if ! curl -sf http://127.0.0.1:$((8000 + $shard))/health > /dev/null ; then
        echo \"[shard $shard] FATAL vllm failed to start\"
        kill \$VLLM_PID 2>/dev/null || true
        exit 2
      fi
      # Run TTT-Discover for tasks where (idx mod 4) == $shard
      TASKS_ALL=(${TASKS[@]})
      for idx in \$(seq 0 \$((\${#TASKS_ALL[@]} - 1))); do
        if [ \$((idx % 4)) -eq $shard ]; then
          T=\${TASKS_ALL[\$idx]}
          echo \"[shard $shard] TTT for task \$T\"
          python /fsx/xuanj/ttt-discover/scripts/ttt_discover_minimal.py \
            --task \$T \
            --vllm-url http://127.0.0.1:$((8000 + $shard)) \
            --num-steps 10 \
            --group-size 8 \
            --no-train \
            --output-dir /fsx/xuanj/ttt-discover/results/ttt \
            --scratch-dir /fsx/xuanj/ttt-discover/scratch \
            > /fsx/xuanj/ttt-discover/logs/slurm/ttt-task-\${T}-\${SLURM_JOB_ID}.log 2>&1 || echo \"[shard $shard] task \$T failed\"
        fi
      done
      kill \$VLLM_PID 2>/dev/null || true
      echo \"[shard $shard] done\"
    " &
done
wait
echo "[ttt-4n] all shards done"
