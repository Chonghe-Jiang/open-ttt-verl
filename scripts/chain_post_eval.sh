#!/bin/bash
set -u
export PATH="/opt/slurm/bin:$PATH"
JOBID="${1:-32578}"
LOG="/fsx/xuanj/ttt-discover/logs/slurm/chain_post_eval.log"
log() { echo "[chain-post] $(date +%H:%M:%S) $*" | tee -a "$LOG"; }
log "starting chain_post_eval: jobid=$JOBID"
while squeue -j "$JOBID" -h 2>/dev/null | grep -q "$JOBID"; do
  pe=$(ls /fsx/xuanj/ttt-discover/results/ttt_iter_post_eval/eval/*.eval.json 2>/dev/null | wc -l)
  log "waiting (job $JOBID): post_eval=$pe/136"
  sleep 600
done
log "post-eval done"

source /fsx/xuanj/ttt-discover/.venv/bin/activate
python /fsx/xuanj/ttt-discover/scripts/yuchen_aligned_report.py \
  --our-base-dir /fsx/xuanj/ttt-discover/results/base_iter/eval \
  --our-ttt-dir /fsx/xuanj/ttt-discover/results/ttt_iter_post_eval/eval \
  --output /fsx/xuanj/ttt-discover/REPORT_YUCHEN_ALIGNED.md 2>&1 | tee -a "$LOG"

log "ALL DONE — report at /fsx/xuanj/ttt-discover/REPORT_YUCHEN_ALIGNED.md"
