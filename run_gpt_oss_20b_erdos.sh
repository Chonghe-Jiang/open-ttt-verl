#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
WORKSPACE=${WORKSPACE:-${HOME}/scratch/open-ttt-workspace}
ERDOS_DIR=${ERDOS_DIR:-${SCRIPT_DIR}}
MODEL_DIR=${MODEL_DIR:-${WORKSPACE}/models}
CKPT_DIR=${CKPT_DIR:-${CKPT_ROOT:-${WORKSPACE}/ckpt}}
LOG_DIR=${LOG_DIR:-${ERDOS_DIR}/logs}
ARCHIVE_PATH=${ARCHIVE_PATH:-${ERDOS_DIR}/data/archive.json}
MODEL_BF16=${MODEL_BF16:-${MODEL_DIR}/gpt-oss-20b-bf16}
REF_CKPT=${REF_CKPT:-${CKPT_DIR}/gpt-oss-20b_torch_dist}
SAVE_CKPT=${SAVE_CKPT:-${CKPT_DIR}/gpt-oss-20b_erdos_lora}

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

mkdir -p "${LOG_DIR}" "${CKPT_DIR}" "${ERDOS_DIR}/data"

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
    \"PYTHONPATH\": \"${ERDOS_DIR}/Megatron-LM:${ERDOS_DIR}:${ERDOS_DIR}/open-ttt-verl\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

CKPT_ARGS=(
  --hf-checkpoint "${MODEL_BF16}"
  --ref-load "${REF_CKPT}"
  --save "${SAVE_CKPT}"
  --save-interval 20
  --lora-rank 128
  --lora-alpha 128
  --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2
  --lora-save-only
)

TTT_ARGS=(
  --prompt-data "${ERDOS_DIR}/data/erdos_single.jsonl"
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
  --ttt-advantage-clip 20.0
  --reasoning-effort high
)

ROLLOUT_ARGS=(
  --num-rollout 500
  --rollout-batch-size 16
  --n-samples-per-prompt 16
  --rollout-max-response-len 8192
  --rollout-temperature 0.6
  --rollout-top-p 1.0
  --global-batch-size 16
  --micro-batch-size 1
  --use-rollout-logprobs
)

OPTIMIZER_ARGS=(
  --lr 5e-7
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
  --kl-loss-coef 0.0005
  --kl-loss-type low_var_kl
  --eps-clip 0.2
  --entropy-coef 0.0
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
