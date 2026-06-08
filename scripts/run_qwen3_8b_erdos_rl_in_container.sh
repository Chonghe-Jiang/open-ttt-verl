#!/bin/bash
set -Eeuo pipefail

REPO_DIR=${REPO_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)}
WORKSPACE=${WORKSPACE:-/root/workspace}
LOG_ROOT=${LOG_ROOT:-${WORKSPACE}/logs}

MODEL_RAW=${MODEL_RAW:-${WORKSPACE}/models/Qwen3-8B}
REF_CKPT=${REF_CKPT:-${WORKSPACE}/ckpt/Qwen3-8B_torch_dist}
SAVE_CKPT=${SAVE_CKPT:-${WORKSPACE}/ckpt/Qwen3-8B_erdos_lora_2gpu}
PROMPT_DATA=${PROMPT_DATA:-${REPO_DIR}/data/erdos_single.jsonl}
ARCHIVE_PATH=${ARCHIVE_PATH:-${WORKSPACE}/data/qwen3_8b_erdos_archive_2gpu.json}
TTT_SANDBOX_WORK_DIR=${TTT_SANDBOX_WORK_DIR:-${WORKSPACE}/tmp/erdos-sandbox}

TOTAL_GPUS=${TOTAL_GPUS:-2}
ACTOR_NUM_GPUS=${ACTOR_NUM_GPUS:-2}
ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-2}
ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-${TOTAL_GPUS}}
COLOCATE=${COLOCATE:-1}
TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-1}
PIPELINE_MODEL_PARALLEL_SIZE=${PIPELINE_MODEL_PARALLEL_SIZE:-2}
CONTEXT_PARALLEL_SIZE=${CONTEXT_PARALLEL_SIZE:-1}
EXPERT_MODEL_PARALLEL_SIZE=${EXPERT_MODEL_PARALLEL_SIZE:-1}
EXPERT_TENSOR_PARALLEL_SIZE=${EXPERT_TENSOR_PARALLEL_SIZE:-1}

NUM_ROLLOUT=${NUM_ROLLOUT:-50}
START_ROLLOUT_ID=${START_ROLLOUT_ID:-0}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-2}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-16}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-30000}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-1.0}
SEQ_LENGTH=${SEQ_LENGTH:-4096}
MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS:-40960}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-8192}

SAVE_INTERVAL=${SAVE_INTERVAL:-10}
LORA_RANK=${LORA_RANK:-64}
LORA_ALPHA=${LORA_ALPHA:-64}
LR=${LR:-5e-7}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.35}
SGLANG_CONTEXT_LENGTH=${SGLANG_CONTEXT_LENGTH:-16384}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
KILL_STALE_PROCESSES=${KILL_STALE_PROCESSES:-1}

mkdir -p "${LOG_ROOT}" "$(dirname "${SAVE_CKPT}")" "$(dirname "${ARCHIVE_PATH}")" "${WORKSPACE}/tmp" "${TTT_SANDBOX_WORK_DIR}"

RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
LOG_FILE=${LOG_FILE:-${LOG_ROOT}/run_qwen3_8b_erdos_rl_${RUN_ID}.log}
TRAIN_LOG_FILE=${TRAIN_LOG_FILE:-${LOG_ROOT}/qwen3_8b_erdos_rl_${RUN_ID}.train.log}

touch "${LOG_FILE}" "${TRAIN_LOG_FILE}"
rm -f "${LOG_ROOT}/run_qwen3_8b_erdos_rl.latest.log" "${LOG_ROOT}/qwen3_8b_erdos_rl.latest.train.log"
ln -s "${LOG_FILE}" "${LOG_ROOT}/run_qwen3_8b_erdos_rl.latest.log"
ln -s "${TRAIN_LOG_FILE}" "${LOG_ROOT}/qwen3_8b_erdos_rl.latest.train.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
trap 'status=$?; ray stop --force >/dev/null 2>&1 || true; echo "Exit status: ${status}"; echo "Ended at: $(date -Is)"' EXIT

MODEL_PARALLEL_WORLD_SIZE=$((TENSOR_MODEL_PARALLEL_SIZE * PIPELINE_MODEL_PARALLEL_SIZE * CONTEXT_PARALLEL_SIZE))
if [ "${MODEL_PARALLEL_WORLD_SIZE}" -le 0 ]; then
  echo "Invalid model parallel world size: ${MODEL_PARALLEL_WORLD_SIZE}" >&2
  exit 1
fi
if [ $((ACTOR_NUM_GPUS % MODEL_PARALLEL_WORLD_SIZE)) -ne 0 ]; then
  echo "Invalid actor config: ACTOR_NUM_GPUS=${ACTOR_NUM_GPUS} must be divisible by TP*PP*CP=${MODEL_PARALLEL_WORLD_SIZE}" >&2
  exit 1
fi
if [ $((ROLLOUT_NUM_GPUS % ROLLOUT_NUM_GPUS_PER_ENGINE)) -ne 0 ]; then
  echo "Invalid rollout config: ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS} must be divisible by ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE}" >&2
  exit 1
fi
if [ $((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT)) -ne "${GLOBAL_BATCH_SIZE}" ]; then
  echo "Invalid rollout batch config: ROLLOUT_BATCH_SIZE*N_SAMPLES_PER_PROMPT must equal GLOBAL_BATCH_SIZE." >&2
  echo "Got ${ROLLOUT_BATCH_SIZE}*${N_SAMPLES_PER_PROMPT} != ${GLOBAL_BATCH_SIZE}" >&2
  exit 1
fi

echo "Logging to ${LOG_FILE}"
echo "Train log: ${TRAIN_LOG_FILE}"
echo "Repo: ${REPO_DIR}"
echo "Workspace: ${WORKSPACE}"
echo "HF model: ${MODEL_RAW}"
echo "Megatron ref checkpoint: ${REF_CKPT}"
echo "Save checkpoint: ${SAVE_CKPT}"
echo "Prompt data: ${PROMPT_DATA}"
echo "Archive: ${ARCHIVE_PATH}"
echo "Host: $(hostname)"
echo "Total GPUs: ${TOTAL_GPUS}"
echo "Actor GPUs: ${ACTOR_NUM_GPUS}"
echo "Rollout GPUs: ${ROLLOUT_NUM_GPUS}"
echo "Rollout GPUs per engine: ${ROLLOUT_NUM_GPUS_PER_ENGINE}"
echo "GPUs per node: ${NUM_GPUS_PER_NODE}"
echo "Colocate: ${COLOCATE}"
echo "TP/PP/CP: ${TENSOR_MODEL_PARALLEL_SIZE}/${PIPELINE_MODEL_PARALLEL_SIZE}/${CONTEXT_PARALLEL_SIZE}"
echo "Rollouts: ${NUM_ROLLOUT} steps, ${ROLLOUT_BATCH_SIZE} prompts/step, ${N_SAMPLES_PER_PROMPT} samples/prompt, ${GLOBAL_BATCH_SIZE} samples/train-step"
echo "LoRA: rank=${LORA_RANK}, alpha=${LORA_ALPHA}"
echo "Reasoning effort: high"

test -d "${REPO_DIR}/slime"
test -f "${REPO_DIR}/slime/scripts/models/qwen3-8B.sh"
test -f "${PROMPT_DATA}"
test -f "${MODEL_RAW}/config.json" || {
  echo "Missing HF model at ${MODEL_RAW}. Run scripts/download_qwen3_8b.sh first." >&2
  exit 1
}
test -f "${REF_CKPT}/latest_checkpointed_iteration.txt" || {
  echo "Missing converted Megatron checkpoint at ${REF_CKPT}. Run scripts/convert_qwen3_8b_in_container.sh first." >&2
  exit 1
}
test -f "${REPO_DIR}/erdos_slime/ttt_discover/__init__.py"

cd "${REPO_DIR}/slime"
export PYTHONUNBUFFERED=1
export PYTHONBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR="${MASTER_ADDR}"
export TOTAL_GPUS="${TOTAL_GPUS}"
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONPATH="${REPO_DIR}/Megatron-LM:${REPO_DIR}:${REPO_DIR}/slime:${PYTHONPATH:-}"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

python3 - <<'PY'
import os
import torch

required = int(os.environ["TOTAL_GPUS"])
count = torch.cuda.device_count()
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", count)
if not torch.cuda.is_available() or count < required:
    raise SystemExit(f"Need at least {required} visible CUDA devices, got {count}.")
PY

if [ "${KILL_STALE_PROCESSES}" = "1" ]; then
  pkill -9 sglang 2>/dev/null || true
  ray stop --force 2>/dev/null || true
  pkill -9 ray 2>/dev/null || true
  sleep 3
fi

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)
if [ "${NVLINK_COUNT}" -gt 0 ]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi
echo "HAS_NVLINK: ${HAS_NVLINK} (detected ${NVLINK_COUNT} NVLink references)"

source scripts/models/qwen3-8B.sh

CKPT_ARGS=(
  --hf-checkpoint "${MODEL_RAW}"
  --ref-load "${REF_CKPT}"
  --save "${SAVE_CKPT}"
  --save-interval "${SAVE_INTERVAL}"
  --lora-rank "${LORA_RANK}"
  --lora-alpha "${LORA_ALPHA}"
  --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2
  --lora-save-only
)

TTT_ARGS=(
  --prompt-data "${PROMPT_DATA}"
  --input-key prompt
  --label-key label
  --apply-chat-template
  --custom-generate-function-path erdos_slime.erdos_generate.generate
  --custom-rm-path erdos_slime.erdos_rm.reward
  --custom-reward-post-process-path erdos_slime.ttt_slime.ttt_reward_post_process
  --custom-advantage-function-path erdos_slime.ttt_slime.ttt_advantages
  --loss-type custom_loss
  --custom-loss-function-path erdos_slime.ttt_slime.ttt_reinforce_loss
  --ttt-archive-path "${ARCHIVE_PATH}"
  --ttt-puct-c 1.0
  --ttt-topk-children 2
  --ttt-sandbox-timeout-s 60
  --ttt-sandbox-cpus 1
  --ttt-sandbox-work-dir "${TTT_SANDBOX_WORK_DIR}"
  --ttt-advantage-clip 20.0
  --reasoning-effort high
)

ROLLOUT_ARGS=(
  --num-rollout "${NUM_ROLLOUT}"
  --start-rollout-id "${START_ROLLOUT_ID}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
  --rollout-temperature "${ROLLOUT_TEMPERATURE}"
  --rollout-top-p "${ROLLOUT_TOP_P}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --micro-batch-size "${MICRO_BATCH_SIZE}"
  --balance-data
  --use-rollout-logprobs
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr "${LR}"
  --lr-decay-style constant
  --weight-decay 0.01
  --adam-beta1 0.9
  --adam-beta2 0.99
  --clip-grad 1.0
)

ALGO_ARGS=(
  --advantage-estimator grpo
  --use-kl-loss
  --kl-loss-coef "${KL_LOSS_COEF}"
  --kl-loss-type low_var_kl
  --eps-clip 0.2
  --entropy-coef 0.0
)

PARALLEL_ARGS=(
  --actor-num-nodes 1
  --actor-num-gpus-per-node "${ACTOR_NUM_GPUS}"
  --num-gpus-per-node "${NUM_GPUS_PER_NODE}"
  --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}"
  --pipeline-model-parallel-size "${PIPELINE_MODEL_PARALLEL_SIZE}"
  --context-parallel-size "${CONTEXT_PARALLEL_SIZE}"
  --expert-model-parallel-size "${EXPERT_MODEL_PARALLEL_SIZE}"
  --expert-tensor-parallel-size "${EXPERT_TENSOR_PARALLEL_SIZE}"
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
  --seq-length "${SEQ_LENGTH}"
  --max-position-embeddings "${MAX_POSITION_EMBEDDINGS}"
)

if [ "${COLOCATE}" = "1" ]; then
  PARALLEL_ARGS+=(--colocate)
else
  PARALLEL_ARGS+=(--rollout-num-gpus "${ROLLOUT_NUM_GPUS}")
fi

if [ "${TENSOR_MODEL_PARALLEL_SIZE}" -gt 1 ]; then
  PARALLEL_ARGS+=(--sequence-parallel)
fi

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
  --sglang-context-length "${SGLANG_CONTEXT_LENGTH}"
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

ray start --head --node-ip-address "${MASTER_ADDR}" \
  --num-gpus "${TOTAL_GPUS}" --disable-usage-stats \
  --dashboard-host=0.0.0.0 --dashboard-port="${RAY_DASHBOARD_PORT}"

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${REPO_DIR}/Megatron-LM:${REPO_DIR}:${REPO_DIR}/slime\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
  "${PARALLEL_ARGS[@]}" \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${TTT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${ALGO_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${MISC_ARGS[@]}" \
  2>&1 | tee -a "${TRAIN_LOG_FILE}"

echo "Qwen3-8B Erdos RL finished. Save checkpoint: ${SAVE_CKPT}"
