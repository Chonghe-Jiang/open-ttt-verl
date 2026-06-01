#!/bin/bash
pkill -9 sglang 2>/dev/null; sleep 2
ray stop --force 2>/dev/null; pkill -9 ray 2>/dev/null
pkill -9 python 2>/dev/null; sleep 3
set -ex

cd /root/workspace/erdos/slime

export MASTER_ADDR=127.0.0.1
ray start --head --node-ip-address ${MASTER_ADDR} \
  --num-gpus 8 --disable-usage-stats \
  --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/workspace/erdos/Megatron-LM:/root/workspace/erdos\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 4 \
  --rollout-num-gpus 4 \
  --rollout-num-gpus-per-engine 4 \
  --num-layers 36 --hidden-size 4096 --ffn-hidden-size 12288 \
  --swiglu --padded-vocab-size 151936 --disable-bias-linear \
  --num-attention-heads 32 --group-query-attention --num-query-groups 8 \
  --kv-channels 128 --qk-layernorm --normalization RMSNorm --norm-epsilon 1e-6 \
  --use-rotary-position-embeddings --rotary-base 1000000 \
  --seq-length 4096 --max-position-embeddings 40960 \
  --tokenizer-type HuggingFaceTokenizer --bf16 \
  --untie-embeddings-and-output-weights \
  --use-mcore-models --sequence-parallel \
  --tensor-model-parallel-size 2 \
  --pipeline-model-parallel-size 1 \
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
  --hf-checkpoint /root/workspace/models/Qwen3-8B \
  --ref-load /root/workspace/erdos/ckpt/Qwen3-8B_torch_dist \
  --save /root/workspace/erdos/ckpt/Qwen3-8B_erdos \
  --save-interval 50 \
  --prompt-data /root/workspace/erdos/data/erdos_single.jsonl \
  --input-key prompt --label-key label \
  --apply-chat-template \
  --custom-generate-function-path erdos_slime.erdos_generate.generate \
  --custom-rm-path erdos_slime.erdos_rm.reward \
  --advantage-estimator grpo \
  --use-kl-loss --kl-loss-coef 0.001 --kl-loss-type low_var_kl \
  --eps-clip 0.2 \
  --num-rollout 300 \
  --rollout-batch-size 2 \
  --n-samples-per-prompt 8 \
  --rollout-max-response-len 8192 \
  --rollout-temperature 1.0 \
  --global-batch-size 16 \
  --micro-batch-size 1 \
  --use-distributed-optimizer \
  --lr 5e-7 --lr-decay-style constant \
  --weight-decay 0.01 --adam-beta1 0.9 --adam-beta2 0.99 --clip-grad 1.0 \
  --sglang-mem-fraction-static 0.82 \
  --sglang-context-length 16384 \
  2>&1 | tee /root/workspace/erdos/logs/qwen3_8b.log
