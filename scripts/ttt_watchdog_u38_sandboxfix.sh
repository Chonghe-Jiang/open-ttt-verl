#!/usr/bin/env bash
set -u

ROOT="/workspace/open-ttt-verl"
STATE_DIR="$ROOT/outputs/ttt_erdos/watchdog_u38_sandboxfix"
WATCH_LOG="$STATE_DIR/watchdog.log"
ACTIVE_FILE="$STATE_DIR/active_run.tsv"

mkdir -p "$STATE_DIR"
cd "$ROOT"

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$WATCH_LOG"
}

run_spec() {
  case "$1" in
    0) echo "16 0.38 256" ;;
    1) echo "8 0.38 256" ;;
    2) echo "4 0.38 128" ;;
    3) echo "4 0.34 128" ;;
    4) echo "2 0.34 64" ;;
    *) return 1 ;;
  esac
}

tag_from_util() {
  printf '%s' "$1" | sed 's/^0\.//; s/\.//g'
}

session_name() {
  local group_size="$1"
  local gpu_util="$2"
  printf 'ttt_gptoss20b_a100_len4096_g1n%s_u%s_sandboxfix' "$group_size" "$(tag_from_util "$gpu_util")"
}

run_name() {
  local group_size="$1"
  local gpu_util="$2"
  local max_num_seqs="$3"
  printf 'a100_gptoss20b_len4096_g1_n%s_seq%s_u%s_officialmatch_sandboxfix_longrun' \
    "$group_size" "$max_num_seqs" "$(tag_from_util "$gpu_util")"
}

config_path_for() {
  local group_size="$1"
  local gpu_util="$2"
  local max_num_seqs="$3"
  printf '%s/outputs/ttt_erdos/sweep_len4096/%s.yaml' "$ROOT" "$(run_name "$group_size" "$gpu_util" "$max_num_seqs")"
}

output_dir_for() {
  local group_size="$1"
  local gpu_util="$2"
  local max_num_seqs="$3"
  printf '%s/outputs/ttt_erdos/%s' "$ROOT" "$(run_name "$group_size" "$gpu_util" "$max_num_seqs")"
}

write_config() {
  local group_size="$1"
  local gpu_util="$2"
  local max_num_seqs="$3"
  local cfg out rel_out run_name_value
  cfg="$(config_path_for "$group_size" "$gpu_util" "$max_num_seqs")"
  out="$(output_dir_for "$group_size" "$gpu_util" "$max_num_seqs")"
  rel_out="${out#$ROOT/}"
  run_name_value="$(run_name "$group_size" "$gpu_util" "$max_num_seqs")"
  mkdir -p "$(dirname "$cfg")" "$out"
  cat > "$cfg" <<EOF
run:
  output_dir: $rel_out
  project_name: ttt_discover
  experiment_name: $run_name_value
  model_path: /workspace/models/unsloth-gpt-oss-20b-BF16
  seed: 0
  num_initial_states: 1
  num_steps: 200
  total_epochs: 200
  max_prompt_length: 4096
  max_response_length: 4096
  learning_rate: 1.0e-05
  ppo_mini_batch_size: 1
  ppo_micro_batch_size_per_gpu: 1
  use_remove_padding: true
  use_kl_loss: true
  kl_loss_coef: 0.01
  rollout_engine: vllm
  tensor_model_parallel_size: 2
  gpu_memory_utilization: $gpu_util
  n_gpus_per_node: 2
  nnodes: 1
  save_freq: 20
  test_freq: -1
  val_before_train: false
  prepare_only: false
ttt:
  groups_per_batch: 1
  group_size: $group_size
  eval_timeout: 300
  cpus: 1
  puct_c: 1.0
  topk_children: 2
  max_buffer_size: 1000
  max_construction_len: 1000
  target_c5: 0.3808
verl_overrides:
- trainer.use_legacy_worker_impl=disable
- actor_rollout_ref.model.external_lib=verl_ttt_discover.verl_ext
- +actor_rollout_ref.model.override_config.attn_implementation=eager
- actor_rollout_ref.model.lora_rank=32
- actor_rollout_ref.model.lora_alpha=32
- actor_rollout_ref.model.enable_gradient_checkpointing=True
- actor_rollout_ref.actor.fsdp_config.model_dtype=bf16
- actor_rollout_ref.actor.fsdp_config.param_offload=True
- actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
- actor_rollout_ref.ref.fsdp_config.model_dtype=bf16
- actor_rollout_ref.ref.fsdp_config.param_offload=True
- actor_rollout_ref.rollout.agent.num_workers=$group_size
- actor_rollout_ref.rollout.load_format=auto
- actor_rollout_ref.rollout.layered_summon=True
- actor_rollout_ref.rollout.enforce_eager=True
- actor_rollout_ref.rollout.free_cache_engine=True
- actor_rollout_ref.rollout.max_model_len=8192
- actor_rollout_ref.rollout.max_num_batched_tokens=8192
- actor_rollout_ref.rollout.max_num_seqs=$max_num_seqs
- actor_rollout_ref.rollout.checkpoint_engine.backend=naive
- actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=3072
- actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
- actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
EOF
}

start_run() {
  local idx="$1"
  local group_size gpu_util max_num_seqs sess cfg out rel_cfg
  read -r group_size gpu_util max_num_seqs <<<"$(run_spec "$idx")"
  sess="$(session_name "$group_size" "$gpu_util")"
  cfg="$(config_path_for "$group_size" "$gpu_util" "$max_num_seqs")"
  out="$(output_dir_for "$group_size" "$gpu_util" "$max_num_seqs")"
  rel_cfg="${cfg#$ROOT/}"

  write_config "$group_size" "$gpu_util" "$max_num_seqs"
  mkdir -p "$out"
  log "Starting fallback index=$idx session=$sess group_size=$group_size gpu_util=$gpu_util max_num_seqs=$max_num_seqs"

  PATH=/venv/main/bin:$PATH ray stop --force >>"$WATCH_LOG" 2>&1 || true
  tmux kill-session -t "$sess" 2>/dev/null || true
  tmux new-session -d -s "$sess" \
    "cd '$ROOT' && PATH=/venv/main/bin:\$PATH HF_HOME=/workspace/.cache/huggingface CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=0 NCCL_SHM_DISABLE=0 NCCL_IB_DISABLE=1 HYDRA_FULL_ERROR=1 RAY_DEDUP_LOGS=0 /venv/main/bin/python -m verl_ttt_discover.main_erdos --config '$rel_cfg' 2>&1 | tee '$out/train.log'"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$idx" "$sess" "$group_size" "$gpu_util" "$max_num_seqs" "$out" > "$ACTIVE_FILE"
}

active_from_file_or_default() {
  if [[ -f "$ACTIVE_FILE" ]]; then
    cat "$ACTIVE_FILE"
  else
    local idx=0 group_size=16 gpu_util=0.38 max_num_seqs=256 sess out
    sess="$(session_name "$group_size" "$gpu_util")"
    out="$(output_dir_for "$group_size" "$gpu_util" "$max_num_seqs")"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$idx" "$sess" "$group_size" "$gpu_util" "$max_num_seqs" "$out" > "$ACTIVE_FILE"
    cat "$ACTIVE_FILE"
  fi
}

log_best() {
  local out="$1"
  local best="$out/best_state.json"
  if [[ -f "$best" ]]; then
    /venv/main/bin/python - "$best" >>"$WATCH_LOG" 2>&1 <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
d = json.loads(p.read_text())
raw = d.get("raw_score")
reward = 1 / (1e-8 + raw) if raw else None
print(f"best_raw={raw} best_reward={reward} timestep={d.get('timestep')} construction_len={len(d.get('construction') or [])}")
PY
  fi
}

log "Watchdog started."

while true; do
  IFS=$'\t' read -r idx sess group_size gpu_util max_num_seqs out <<<"$(active_from_file_or_default)"
  train_log="$out/train.log"

  if [[ -f "$train_log" ]] && grep -Eq "CUDA out of memory|torch.OutOfMemoryError|Engine core initialization failed|WorkerProc initialization failed|RayTaskError" "$train_log"; then
    log "Detected failure in $sess at $out"
    log_best "$out"
    tmux kill-session -t "$sess" 2>/dev/null || true
    next_idx=$((idx + 1))
    if run_spec "$next_idx" >/dev/null; then
      start_run "$next_idx"
    else
      log "No more fallback configs. Leaving watchdog active without restart."
      sleep 600
    fi
  elif [[ -f "$train_log" ]] && grep -q "Training Progress: 100%" "$train_log"; then
    log "Training appears complete for $sess."
    log_best "$out"
    exit 0
  elif tmux has-session -t "$sess" 2>/dev/null; then
    log "OK session=$sess idx=$idx group_size=$group_size gpu_util=$gpu_util max_num_seqs=$max_num_seqs"
    log_best "$out"
  else
    log "Session $sess is not running."
    if [[ -f "$train_log" ]]; then
      log_best "$out"
    fi
    next_idx=$((idx + 1))
    if run_spec "$next_idx" >/dev/null; then
      start_run "$next_idx"
    else
      log "No more fallback configs after missing session."
      sleep 600
    fi
  fi

  sleep 120
done
