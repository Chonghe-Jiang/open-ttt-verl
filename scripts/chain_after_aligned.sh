#!/bin/bash
# Wait for aligned pipeline (32270 or other) to finish, then run report.
set -u
export PATH="/opt/slurm/bin:$PATH"
ALIGNED_JOBID="${1:-32270}"
LOG_FILE="/fsx/xuanj/ttt-discover/logs/slurm/chain_after_aligned.log"

log() {
  echo "[chain-aligned] $(date +%H:%M:%S) $*" | tee -a "$LOG_FILE"
}

log "starting chain_after_aligned: aligned_jobid=$ALIGNED_JOBID"

while squeue -j "$ALIGNED_JOBID" -h 2>/dev/null | grep -q "$ALIGNED_JOBID"; do
  ttl=$(wc -l < /fsx/xuanj/ttt-discover/logs/slurm/ttt-iter-aligned-4tasks_yuchen_seed1-${ALIGNED_JOBID}.log 2>/dev/null || echo 0)
  pe=$(ls /fsx/xuanj/ttt-discover/results/ttt_iter_post_eval/eval/*.eval.json 2>/dev/null | wc -l)
  log "waiting aligned (job $ALIGNED_JOBID): ttt_log=$ttl post_eval=$pe/136"
  sleep 600
done
log "aligned pipeline done"

log "generating Yuchen-aligned report"
source /fsx/xuanj/ttt-discover/.venv/bin/activate
python /fsx/xuanj/ttt-discover/scripts/yuchen_aligned_report.py \
  --our-base-dir /fsx/xuanj/ttt-discover/results/base_iter/eval \
  --our-ttt-dir /fsx/xuanj/ttt-discover/results/ttt_iter_post_eval/eval \
  --output /fsx/xuanj/ttt-discover/REPORT_YUCHEN_ALIGNED.md 2>&1 | tee -a "$LOG_FILE"

log "ALL DONE — report at /fsx/xuanj/ttt-discover/REPORT_YUCHEN_ALIGNED.md"
