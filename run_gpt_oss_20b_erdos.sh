#!/bin/bash
pkill -9 sglang 2>/dev/null; sleep 2
ray stop --force 2>/dev/null; pkill -9 ray 2>/dev/null
pkill -9 python 2>/dev/null; sleep 3
set -ex

cd /root/workspace/erdos/slime
source scripts/models/gpt-oss-20B.sh

export MASTER_ADDR=127.0.0.1
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)

ray start --head --node-ip-address ${MASTER_ADDR} \
  --num-gpus 8 --disable-usage-stats \
  --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/workspace/erdos/Megatron-LM:/root/workspace/erdos\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 8 \
  --rollout-num-gpus 8 \
  --colocate \
  ${MODEL_ARGS[@]} \
  --hf-checkpoint /root/workspace/models/gpt-oss-20b-BF16-processed \
  --ref-load /root/workspace/erdos/ckpt/gpt-oss-20b_torch_dist \
  --save /root/workspace/erdos/ckpt/gpt-oss-20b_erdos \
  --save-interval 20 \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 2 \
  --expert-model-parallel-size 4 \
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
  --use-dynamic-batch-size --max-tokens-per-gpu 1536 --seq-length 1536 \
  --train-memory-margin-bytes 268435456 \
  --megatron-to-hf-mode bridge \
  --moe-token-dispatcher-type alltoall \
  --attention-dropout 0.0 --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 \
  --prompt-data /root/workspace/erdos/data/erdos_single.jsonl \
  --input-key prompt --label-key label \
  --apply-chat-template \
  --custom-generate-function-path erdos_slime.erdos_generate.generate \
  --custom-rm-path erdos_slime.erdos_rm.reward \
  --advantage-estimator grpo \
  --use-kl-loss --kl-loss-coef 0.0005 --kl-loss-type low_var_kl \
  --eps-clip 0.2 \
  --num-rollout 500 \
  --rollout-batch-size 16 \
  --n-samples-per-prompt 16 \
  --rollout-max-response-len 8192 \
  --rollout-temperature 0.6 \
  --global-batch-size 256 \
  --micro-batch-size 1 \
  --lr 5e-7 --lr-decay-style cosine --min-lr 0 \
  --lr-warmup-fraction 0.01 \
  --weight-decay 0.01 --adam-beta1 0.9 --adam-beta2 0.99 --clip-grad 1.0 \
  --sglang-dp-attention \
  --sglang-tp 2 \
  --sglang-mem-fraction-static 0.4 \
  --sglang-cuda-graph-max-bs 16 \
  --sglang-max-running-requests 64 \
  --sglang-context-length 16384 \
  2>&1 | tee /root/workspace/erdos/logs/gpt_oss_20b.log