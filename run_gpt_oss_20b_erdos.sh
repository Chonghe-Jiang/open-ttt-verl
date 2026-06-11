#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
WORKSPACE=${WORKSPACE:-${HOME}/scratch/open-ttt-workspace}
ERDOS_DIR=${ERDOS_DIR:-${SCRIPT_DIR}}
MODEL_DIR=${MODEL_DIR:-${WORKSPACE}/models}
CKPT_DIR=${CKPT_DIR:-${CKPT_ROOT:-${WORKSPACE}/ckpt}}
LOG_DIR=${LOG_DIR:-${ERDOS_DIR}/logs}
RUN_ID=${RUN_ID:-gpt_oss_20b_onestep_$(date +%Y%m%d_%H%M%S)}
ARCHIVE_PATH=${ARCHIVE_PATH:-${WORKSPACE}/data/gpt_oss_20b_erdos_archive_${RUN_ID}.json}
MODEL_BF16=${MODEL_BF16:-${MODEL_DIR}/gpt-oss-20b-bf16}
REF_CKPT=${REF_CKPT:-${CKPT_DIR}/gpt-oss-20b_torch_dist}
SAVE_CKPT=${SAVE_CKPT:-${CKPT_DIR}/gpt-oss-20b_erdos_lora_${RUN_ID}}
NUM_ROLLOUT=${NUM_ROLLOUT:-1}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-1}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-2}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-2}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-30000}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-1.0}
SAVE_INTERVAL=${SAVE_INTERVAL:-1}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-32}
LR=${LR:-4e-5}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.1}
TTT_ENTROPIC_TARGET_KL=${TTT_ENTROPIC_TARGET_KL:-0.6931471805599453}
TTT_TARGET_C5=${TTT_TARGET_C5:-0.38080}
TTT_SANDBOX_TIMEOUT_S=${TTT_SANDBOX_TIMEOUT_S:-1000}
TTT_SANDBOX_CPUS=${TTT_SANDBOX_CPUS:-2}
TTT_SANDBOX_WORK_DIR=${TTT_SANDBOX_WORK_DIR:-${WORKSPACE}/tmp/erdos-sandbox}

test -f "${MODEL_BF16}/config.json" || {
  echo "Missing BF16 model at ${MODEL_BF16}. Run scripts/prepare_gpt_oss_20b_slime.sh first." >&2
  exit 1
}
test -f "${REF_CKPT}/latest_checkpointed_iteration.txt" || {
  echo "Missing Megatron torch_dist checkpoint at ${REF_CKPT}. Run scripts/prepare_gpt_oss_20b_slime.sh first." >&2
  exit 1
}
test -d "${ERDOS_DIR}/slime" || {
  echo "Missing Slime directory at ${ERDOS_DIR}/slime. Set ERDOS_DIR to the repo root." >&2
  exit 1
}

pkill -9 sglang 2>/dev/null || true
sleep 2
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
pkill -9 python 2>/dev/null || true
sleep 3

set -x

if [ $((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT)) -ne "${GLOBAL_BATCH_SIZE}" ]; then
  echo "Invalid rollout batch config: ROLLOUT_BATCH_SIZE*N_SAMPLES_PER_PROMPT must equal GLOBAL_BATCH_SIZE." >&2
  echo "Got ${ROLLOUT_BATCH_SIZE}*${N_SAMPLES_PER_PROMPT} != ${GLOBAL_BATCH_SIZE}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}" "${CKPT_DIR}" "$(dirname "${ARCHIVE_PATH}")" "${TTT_SANDBOX_WORK_DIR}"

echo "=== gpt-oss-20b erdos one-step/default training ==="
echo "run_id: ${RUN_ID}"
echo "workspace: ${WORKSPACE}"
echo "model_bf16: ${MODEL_BF16}"
echo "ref_ckpt: ${REF_CKPT}"
echo "save_ckpt: ${SAVE_CKPT}"
echo "archive_path: ${ARCHIVE_PATH}"
echo "num_rollout: ${NUM_ROLLOUT}"
echo "rollout_batch_size: ${ROLLOUT_BATCH_SIZE}"
echo "n_samples_per_prompt: ${N_SAMPLES_PER_PROMPT}"
echo "global_batch_size: ${GLOBAL_BATCH_SIZE}"
echo "rollout_max_response_len: ${ROLLOUT_MAX_RESPONSE_LEN}"
echo "lora_rank: ${LORA_RANK}"
echo "lr: ${LR}"
echo "kl_loss_coef: ${KL_LOSS_COEF}"

cd "${ERDOS_DIR}/slime"
source scripts/models/gpt-oss-20B.sh

export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export PYTHONBUFFERED=1

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)
if [ "${NVLINK_COUNT}" -gt 0 ]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi

ray start --head --node-ip-address "${MASTER_ADDR}" \
  --num-gpus 8 --disable-usage-stats \
  --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${ERDOS_DIR}/Megatron-LM:${ERDOS_DIR}:${ERDOS_DIR}/slime\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"ERDOS_ARCHIVE_PATH\": \"${ARCHIVE_PATH}\",
    \"ERDOS_RUN_ID\": \"${RUN_ID}\",
    \"ERDOS_MODEL_PATH\": \"${MODEL_BF16}\",
    \"ERDOS_ROLLOUT_N\": \"${N_SAMPLES_PER_PROMPT}\",
    \"ERDOS_TRAIN_MAX_RESPONSE_TOKENS\": \"${ROLLOUT_MAX_RESPONSE_LEN}\",
    \"ERDOS_ENABLE_THINKING\": \"0\"
  }
}"

CKPT_ARGS=(
  --hf-checkpoint "${MODEL_BF16}"
  --ref-load "${REF_CKPT}"
  --save "${SAVE_CKPT}"
  --save-interval "${SAVE_INTERVAL}"
  --lora-rank "${LORA_RANK}"
  --lora-alpha "${LORA_ALPHA}"
  --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2
  --lora-save-only
)

TTT_ARGS=(
  --prompt-data "${ERDOS_DIR}/data/erdos_single.jsonl"
  --input-key prompt
  --label-key label
  --apply-chat-template
  --custom-generate-function-path erdos_slime.ttt_slime.generate
  --custom-rm-path erdos_slime.ttt_slime.reward
  --custom-reward-post-process-path erdos_slime.ttt_slime.ttt_reward_post_process
  --custom-advantage-function-path erdos_slime.ttt_slime.ttt_advantages
  --loss-type custom_loss
  --custom-loss-function-path erdos_slime.ttt_slime.ttt_reinforce_loss
  --ttt-archive-path "${ARCHIVE_PATH}"
  --ttt-puct-c 1.0
  --ttt-topk-children 2
  --ttt-sandbox-timeout-s "${TTT_SANDBOX_TIMEOUT_S}"
  --ttt-sandbox-cpus "${TTT_SANDBOX_CPUS}"
  --ttt-sandbox-work-dir "${TTT_SANDBOX_WORK_DIR}"
  --ttt-target-c5 "${TTT_TARGET_C5}"
  --ttt-entropic-target-kl "${TTT_ENTROPIC_TARGET_KL}"
  --ttt-advantage-clip 20.0
  --ttt-is-clip 0
  --reasoning-effort high
)

ROLLOUT_ARGS=(
  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
  --rollout-temperature "${ROLLOUT_TEMPERATURE}"
  --rollout-top-p "${ROLLOUT_TOP_P}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --micro-batch-size "${MICRO_BATCH_SIZE}"
  --use-rollout-logprobs
)

OPTIMIZER_ARGS=(
  --lr "${LR}"
  --lr-decay-style cosine
  --min-lr 0
  --lr-warmup-fraction 0.01
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
  --disable-rewards-normalization
)

PARALLEL_ARGS=(
  --actor-num-nodes 1
  --actor-num-gpus-per-node 8
  --rollout-num-gpus 8
  --colocate
  --tensor-model-parallel-size 1
  --pipeline-model-parallel-size 2
  --context-parallel-size 1
  --expert-model-parallel-size 4
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu 1536
  --seq-length 1536
  --train-memory-margin-bytes 268435456
)

SGLANG_ARGS=(
  --sglang-dp-attention
  --sglang-tp 2
  --sglang-mem-fraction-static 0.4
  --sglang-cuda-graph-max-bs 16
  --sglang-max-running-requests 64
  --sglang-context-length 16384
)

MISC_ARGS=(
  --megatron-to-hf-mode bridge
  --moe-token-dispatcher-type alltoall
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
)

ray job submit --address="http://127.0.0.1:8265" \
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
  2>&1 | tee "${LOG_DIR}/gpt_oss_20b.log"
