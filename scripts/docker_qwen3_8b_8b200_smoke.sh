#!/bin/bash
set -Eeuo pipefail

export WORKSPACE=${WORKSPACE:-/root/workspace}
export REPO_DIR=${REPO_DIR:-/root/workspace/erdos}
export LOG_ROOT=${LOG_ROOT:-${WORKSPACE}/logs}
export MODEL_RAW=${MODEL_RAW:-${WORKSPACE}/models/Qwen3-8B}
export REF_CKPT=${REF_CKPT:-${WORKSPACE}/ckpt/Qwen3-8B_torch_dist}
export SAVE_CKPT=${SAVE_CKPT:-${WORKSPACE}/ckpt/Qwen3-8B_erdos_lora_8b200_smoke}
export ARCHIVE_PATH=${ARCHIVE_PATH:-${WORKSPACE}/data/qwen3_8b_erdos_archive_8b200_smoke.json}

export TOTAL_GPUS=${TOTAL_GPUS:-8}
export NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}
export ACTOR_NUM_GPUS=${ACTOR_NUM_GPUS:-4}
export ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-4}
export ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}
export COLOCATE=${COLOCATE:-0}

export TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-1}
export PIPELINE_MODEL_PARALLEL_SIZE=${PIPELINE_MODEL_PARALLEL_SIZE:-2}
export CONTEXT_PARALLEL_SIZE=${CONTEXT_PARALLEL_SIZE:-1}
export EXPERT_MODEL_PARALLEL_SIZE=${EXPERT_MODEL_PARALLEL_SIZE:-1}
export EXPERT_TENSOR_PARALLEL_SIZE=${EXPERT_TENSOR_PARALLEL_SIZE:-1}

export NUM_ROLLOUT=${NUM_ROLLOUT:-0}
export ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-8}
export N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-64}
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-512}
export MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
export ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-30000}

export LORA_RANK=${LORA_RANK:-32}
export LORA_ALPHA=${LORA_ALPHA:-32}
export LR=${LR:-4e-5}
export KL_LOSS_COEF=${KL_LOSS_COEF:-0.1}
export TTT_ENTROPIC_TARGET_KL=${TTT_ENTROPIC_TARGET_KL:-0.6931471805599453}
export ADAM_BETA1=${ADAM_BETA1:-0.9}
export ADAM_BETA2=${ADAM_BETA2:-0.95}
export ADAM_EPS=${ADAM_EPS:-1e-8}
export WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}

export SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.75}
export SGLANG_CONTEXT_LENGTH=${SGLANG_CONTEXT_LENGTH:-32768}
export MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-8192}
export KILL_STALE_PROCESSES=${KILL_STALE_PROCESSES:-0}

mkdir -p "${LOG_ROOT}" "${WORKSPACE}/data" "${WORKSPACE}/ckpt"

echo "=== qwen3-8b erdos rl docker 8b200 smoke ==="
echo "workspace: ${WORKSPACE}"
echo "repo_dir: ${REPO_DIR}"
echo "model_raw: ${MODEL_RAW}"
echo "ref_ckpt: ${REF_CKPT}"
echo "save_ckpt: ${SAVE_CKPT}"
echo "archive_path: ${ARCHIVE_PATH}"
echo "total_gpus: ${TOTAL_GPUS}"
echo "actor_gpus: ${ACTOR_NUM_GPUS}"
echo "rollout_gpus: ${ROLLOUT_NUM_GPUS}"
echo "rollout_gpus_per_engine: ${ROLLOUT_NUM_GPUS_PER_ENGINE}"
echo "colocate: ${COLOCATE}"
echo "smoke_num_rollout: ${NUM_ROLLOUT}"
echo "paper_rollout_groups: ${ROLLOUT_BATCH_SIZE}"
echo "paper_rollouts_per_group: ${N_SAMPLES_PER_PROMPT}"
echo "paper_global_batch_size: ${GLOBAL_BATCH_SIZE}"
echo "lora_rank: ${LORA_RANK}"
echo "lr: ${LR}"
echo "adam_beta1: ${ADAM_BETA1}"
echo "adam_beta2: ${ADAM_BETA2}"
echo "adam_eps: ${ADAM_EPS}"
echo "kl_loss_coef: ${KL_LOSS_COEF}"
echo "ttt_entropic_target_kl: ${TTT_ENTROPIC_TARGET_KL}"
echo "sglang_context_length: ${SGLANG_CONTEXT_LENGTH}"
echo "rollout_max_response_len: ${ROLLOUT_MAX_RESPONSE_LEN}"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L
else
  echo "nvidia-smi is not available in the container." >&2
  exit 2
fi

python3 - <<'PY'
import os
from pathlib import Path

import torch

expected_gpus = int(os.environ["TOTAL_GPUS"])
actual_gpus = torch.cuda.device_count()
print(f"torch.cuda.device_count={actual_gpus}")
if actual_gpus != expected_gpus:
    raise SystemExit(f"Expected {expected_gpus} CUDA devices, got {actual_gpus}")

required_paths = [
    Path(os.environ["MODEL_RAW"]),
    Path(os.environ["REF_CKPT"]),
    Path(os.environ["REF_CKPT"]) / "latest_checkpointed_iteration.txt",
]
for path in required_paths:
    print(f"check_path: {path}")
    if not path.exists():
        raise SystemExit(f"Required path does not exist: {path}")
PY

cd "${REPO_DIR}"
scripts/run_qwen3_8b_erdos_rl_in_container.sh
