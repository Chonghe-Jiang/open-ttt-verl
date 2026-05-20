#!/bin/bash
# Wait for base eval (32227) to finish, then submit 4-node aligned pipeline.
# Aligned pipeline trains TTT on 4 Yuchen-seed-1 tasks and evaluates 17 ID/OOD tasks.
set -u
export PATH="/opt/slurm/bin:$PATH"
BASE_JOBID="${1:-32227}"
LOG_FILE="/fsx/xuanj/ttt-discover/logs/slurm/chain_aligned.log"

log() {
  echo "[chain-aligned] $(date +%H:%M:%S) $*" | tee -a "$LOG_FILE"
}

log "starting chain_aligned: base_jobid=$BASE_JOBID"

while squeue -j "$BASE_JOBID" -h 2>/dev/null | grep -q "$BASE_JOBID"; do
  bc=$(ls /fsx/xuanj/ttt-discover/results/base_iter/eval/*.eval.json 2>/dev/null | wc -l)
  log "waiting base eval (job $BASE_JOBID): $bc/136"
  sleep 600
done
bc=$(ls /fsx/xuanj/ttt-discover/results/base_iter/eval/*.eval.json 2>/dev/null | wc -l)
log "base eval done: $bc/136"

# Wait for nodes to clean up
sleep 30

# Submit aligned pipeline (4 nodes, 36h time limit)
cd /fsx/xuanj/ttt-discover
sbatch_out=$(sbatch scripts/run_ttt_aligned_4node.sbatch)
ALIGNED_JOBID=$(echo "$sbatch_out" | grep -oE '[0-9]+$')
log "aligned pipeline submitted: jobid=$ALIGNED_JOBID"

if [ -z "$ALIGNED_JOBID" ]; then
  log "FATAL: failed to submit aligned sbatch"
  exit 1
fi

while squeue -j "$ALIGNED_JOBID" -h 2>/dev/null | grep -q "$ALIGNED_JOBID"; do
  ttl=$(wc -l < /fsx/xuanj/ttt-discover/logs/slurm/ttt-iter-aligned-4tasks_yuchen_seed1-${ALIGNED_JOBID}.log 2>/dev/null || echo 0)
  pe=$(ls /fsx/xuanj/ttt-discover/results/ttt_iter_post_eval/eval/*.eval.json 2>/dev/null | wc -l)
  log "waiting aligned (job $ALIGNED_JOBID): ttt_log=$ttl post_eval=$pe/136"
  sleep 600
done
log "aligned pipeline done"

# Generate report
log "generating Yuchen-aligned report"
source /fsx/xuanj/ttt-discover/.venv/bin/activate
python /fsx/xuanj/ttt-discover/scripts/yuchen_aligned_report.py \
  --our-base-dir /fsx/xuanj/ttt-discover/results/base_iter/eval \
  --our-ttt-dir /fsx/xuanj/ttt-discover/results/ttt_iter_post_eval/eval \
  --output /fsx/xuanj/ttt-discover/REPORT_YUCHEN_ALIGNED.md 2>&1 | tee -a "$LOG_FILE"

log "ALL DONE — report at /fsx/xuanj/ttt-discover/REPORT_YUCHEN_ALIGNED.md"
