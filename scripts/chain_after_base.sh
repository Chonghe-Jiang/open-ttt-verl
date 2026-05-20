#!/bin/bash
# Wait for base eval (32227) to finish, then submit mixed 4-node TTT+ZS pipeline
set -u
BASE_JOBID="${1:-32227}"
TRAIN_TASK="${2:-cbl__low_av_tight_dl_large_oh}"
LOG_FILE="/fsx/xuanj/ttt-discover/logs/slurm/chain_after_base.log"

log() {
  echo "[chain] $(date +%H:%M:%S) $*" | tee -a "$LOG_FILE"
}

log "starting chain_after_base: base=$BASE_JOBID task=$TRAIN_TASK"

while squeue -j "$BASE_JOBID" -h 2>/dev/null | grep -q "$BASE_JOBID"; do
  bc=$(ls /fsx/xuanj/ttt-discover/results/base_iter/eval/*.eval.json 2>/dev/null | wc -l)
  log "waiting base eval (job $BASE_JOBID): $bc/136"
  sleep 600
done
bc=$(ls /fsx/xuanj/ttt-discover/results/base_iter/eval/*.eval.json 2>/dev/null | wc -l)
log "base eval done: final $bc/136"

# Wait a bit for nodes to clean up
sleep 30

# Verify no leaked python on freed nodes
log "checking for leaked processes on freed nodes"
for n in $(scontrol show job $BASE_JOBID 2>/dev/null | grep -oE 'NodeList=[^ ]+' | head -1 | sed 's/NodeList=//' | sed 's/p5en-odcr-queue-dy-p5en48xlarge-//' | tr ',' ' ' | tr -d '[]'); do
  log "node $n: skipping leak check (job done, slurm should clean)"
done

# Submit mixed 4-node pipeline (TTT train 2n + concurrent zs eval 2n, then 4-shard final ZS)
cd /fsx/xuanj/ttt-discover
TRAIN_TASK="$TRAIN_TASK" sbatch_out=$(TRAIN_TASK="$TRAIN_TASK" sbatch scripts/run_ttt_mixed_4node.sbatch)
MIXED_JOBID=$(echo "$sbatch_out" | grep -oE '[0-9]+$')
log "mixed pipeline submitted: jobid=$MIXED_JOBID"

if [ -z "$MIXED_JOBID" ]; then
  log "FATAL: failed to submit mixed sbatch"
  exit 1
fi

# Monitor mixed job until done
while squeue -j "$MIXED_JOBID" -h 2>/dev/null | grep -q "$MIXED_JOBID"; do
  ttl=$(wc -l < "/fsx/xuanj/ttt-discover/logs/slurm/ttt-iter-train-${TRAIN_TASK}-${MIXED_JOBID}.log" 2>/dev/null || echo 0)
  zc=$(ls /fsx/xuanj/ttt-discover/results/ttt_iter_zeroshot/eval/*.eval.json 2>/dev/null | wc -l)
  step_runs=$(ls -d /fsx/xuanj/ttt-discover/results/ttt_iter_zeroshot/step_eval/step_* 2>/dev/null | wc -l)
  log "waiting mixed (job $MIXED_JOBID): ttt_log_lines=$ttl step_evals=$step_runs final_zs=$zc/128"
  sleep 600
done
log "mixed pipeline done"

# Generate report
log "generating Yuchen-aligned report"
source /fsx/xuanj/ttt-discover/.venv/bin/activate
python /fsx/xuanj/ttt-discover/scripts/yuchen_aligned_report.py \
  --train-task "$TRAIN_TASK" 2>&1 | tee -a "$LOG_FILE"

log "ALL DONE — report at /fsx/xuanj/ttt-discover/REPORT_YUCHEN_ALIGNED.md"
