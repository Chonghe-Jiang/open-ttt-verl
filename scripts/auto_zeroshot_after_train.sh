#!/bin/bash
# Wait for TTT train job to finish, then submit zero-shot eval
set -u
JOBID="${1:-32228}"
TRAIN_TASK="${2:-cbl__low_av_tight_dl_large_oh}"
LORA_DIR="/fsx/xuanj/ttt-discover/results/ttt_iter_lora/${TRAIN_TASK}/final"

while squeue -j "$JOBID" -h 2>/dev/null | grep -q "$JOBID"; do
  echo "[watcher] $(date +%H:%M:%S) ttt train job $JOBID still running"
  sleep 300
done
echo "[watcher] $(date +%H:%M:%S) ttt train job $JOBID finished"

if [ ! -d "$LORA_DIR" ]; then
  echo "[watcher] no final LoRA dir — checking step ckpts"
  LORA_DIR=$(ls -td /fsx/xuanj/ttt-discover/results/ttt_iter_lora/${TRAIN_TASK}/step* 2>/dev/null | head -1)
  if [ -z "$LORA_DIR" ]; then
    echo "[watcher] no step ckpts either, abort"
    exit 1
  fi
  echo "[watcher] using last step ckpt: $LORA_DIR"
fi

cd /fsx/xuanj/ttt-discover
LORA_DIR="$LORA_DIR" TRAIN_TASK="$TRAIN_TASK" sbatch scripts/run_ttt_zeroshot_eval.sbatch
