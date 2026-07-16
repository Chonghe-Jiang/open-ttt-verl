#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FRONTIER_DIR="${REPO_ROOT}/reference/Frontier-CS"
ALG_DIR="${FRONTIER_DIR}/algorithmic"
RUNTIME_DIR="${REPO_ROOT}/.runtime"
SIF_PATH="${SIF_PATH:-/work/mit/ppliang_mit/chonghej/open-ttt-verl/containers/open-ttt-verl-ttt-vllm.sif}"
CONFIG="${CONFIG:-guidance_ttt/config/polyomino_b200_3gpu_gpt_oss_120b_batch8_group16_code_delta_8192_smoke.yaml}"
GUIDANCE_MODEL_DIR="${GUIDANCE_MODEL_DIR:-models/Qwen3-8B}"
EXTRA_CONTAINER_PYTHONPATH="${EXTRA_CONTAINER_PYTHONPATH:-}"
EXECUTION_MODEL_DIR="${EXECUTION_MODEL_DIR:-models/gpt-oss-120b}"
EXECUTION_MODEL_NAME="${EXECUTION_MODEL_NAME:-openai/gpt-oss-120b}"
EXECUTION_EXTRA_CONTAINER_PYTHONPATH="${EXECUTION_EXTRA_CONTAINER_PYTHONPATH:-}"
EXECUTION_LOG_NAME="${EXECUTION_LOG_NAME:-gpt-oss-120b}"
EXECUTION_VLLM_DTYPE="${EXECUTION_VLLM_DTYPE:-auto}"
EXECUTION_VLLM_GPU_MEMORY_UTILIZATION="${EXECUTION_VLLM_GPU_MEMORY_UTILIZATION:-0.88}"
EXECUTION_VLLM_MAX_MODEL_LEN="${EXECUTION_VLLM_MAX_MODEL_LEN:-32768}"
EXECUTION_VLLM_MAX_NUM_SEQS="${EXECUTION_VLLM_MAX_NUM_SEQS:-16}"
EXECUTION_VLLM_ENFORCE_EAGER="${EXECUTION_VLLM_ENFORCE_EAGER:-1}"
EXECUTION_VLLM_LANGUAGE_MODEL_ONLY="${EXECUTION_VLLM_LANGUAGE_MODEL_ONLY:-0}"
EXECUTION_VLLM_REASONING_PARSER="${EXECUTION_VLLM_REASONING_PARSER:-}"
START_EXECUTION_SERVER="${START_EXECUTION_SERVER:-1}"
BOOTSTRAP_BEFORE_RUN="${BOOTSTRAP_BEFORE_RUN:-0}"
RUN_TAG="${RUN_TAG:-${SLURM_JOB_ID:-manual}}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/guidance_ttt/polyomino_b200_3gpu_smoke_${RUN_TAG}}"
EXPECTED_GPUS="${EXPECTED_GPUS:-3}"
TRAINING_GPUS="${TRAINING_GPUS:-0,1}"
EXECUTION_GPU="${EXECUTION_GPU:-2}"
EXPECTED_GROUPS="${EXPECTED_GROUPS:-8}"
EXPECTED_GROUP_SIZE="${EXPECTED_GROUP_SIZE:-16}"
LOG_DIR="${REPO_ROOT}/outputs/slurm"
NODE_BIN="${RUNTIME_DIR}/node-v${NODE_VERSION:-20.19.4}-linux-x64/bin"
GOJUDGE="${RUNTIME_DIR}/go-judge/bin/go-judge"
MODE="${1:-run}"

cd "${REPO_ROOT}"
mkdir -p "${LOG_DIR}" .hf_cache .tmp .triton_cache .ray_tmp .apptainer_home/.cache

required_artifacts=(
  "${SIF_PATH}"
  "${GUIDANCE_MODEL_DIR}/config.json"
  "${ALG_DIR}/problems/0/config.yaml"
  "${NODE_BIN}/node"
  "${GOJUDGE}"
  "${RUNTIME_DIR}/go-judge/mount.yaml"
)
if [[ "${START_EXECUTION_SERVER}" == "1" ]]; then
  required_artifacts+=("${EXECUTION_MODEL_DIR}/config.json")
fi
for required in "${required_artifacts[@]}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required runtime artifact: ${required}" >&2
    echo "Run scripts/setup_polyomino_b200.sh first." >&2
    exit 1
  fi
done

export APPTAINER_CACHEDIR="${REPO_ROOT}/.apptainer_cache"
export APPTAINER_TMPDIR="${REPO_ROOT}/.apptainer_tmp"
export HF_HOME="${REPO_ROOT}/.hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_XET_CACHE="${HF_HOME}/xet"
export TMPDIR="${REPO_ROOT}/.tmp"
export TRITON_CACHE_DIR="${REPO_ROOT}/.triton_cache"
export RAY_TMPDIR="${REPO_ROOT}/.ray_tmp"
export PATH="${NODE_BIN}:$(dirname "${GOJUDGE}"):${PATH}"
mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}" "${HF_DATASETS_CACHE}" "${HUGGINGFACE_HUB_CACHE}"

CONTAINER_PYTHONPATH="/workspace/guidance:/workspace/guidance/reference/Frontier-CS/src"
if [[ -n "${EXTRA_CONTAINER_PYTHONPATH}" ]]; then
  CONTAINER_PYTHONPATH="${EXTRA_CONTAINER_PYTHONPATH}:${CONTAINER_PYTHONPATH}"
fi
EXECUTION_CONTAINER_PYTHONPATH="/workspace/guidance:/workspace/guidance/reference/Frontier-CS/src"
if [[ -n "${EXECUTION_EXTRA_CONTAINER_PYTHONPATH}" ]]; then
  EXECUTION_CONTAINER_PYTHONPATH="${EXECUTION_EXTRA_CONTAINER_PYTHONPATH}:${EXECUTION_CONTAINER_PYTHONPATH}"
fi

APPTAINER_COMMON=(
  apptainer exec --nv --cleanenv
  --home "${REPO_ROOT}/.apptainer_home:/container_home"
  --bind "${REPO_ROOT}:/workspace/guidance"
  --pwd /workspace/guidance
  --env HF_HOME=/workspace/guidance/.hf_cache
  --env HF_DATASETS_CACHE=/workspace/guidance/.hf_cache/datasets
  --env HUGGINGFACE_HUB_CACHE=/workspace/guidance/.hf_cache/hub
  --env TRANSFORMERS_CACHE=/workspace/guidance/.hf_cache/hub
  --env HF_XET_CACHE=/workspace/guidance/.hf_cache/xet
  --env TMPDIR=/workspace/guidance/.tmp
  --env TRITON_CACHE_DIR=/workspace/guidance/.triton_cache
  --env RAY_TMPDIR=/workspace/guidance/.ray_tmp
  --env XDG_CACHE_HOME=/container_home/.cache
  --env HYDRA_FULL_ERROR=1
  --env RAY_DEDUP_LOGS=0
  --env RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
  --env EXPECTED_GPUS="${EXPECTED_GPUS}"
  --env VLLM_NO_USAGE_STATS=1
  --env VLLM_WORKER_MULTIPROC_METHOD=spawn
  --env NCCL_IB_DISABLE=1
  --env TOKENIZERS_PARALLELISM=false
)
APPTAINER_BASE=("${APPTAINER_COMMON[@]}" --env PYTHONPATH="${CONTAINER_PYTHONPATH}" "${SIF_PATH}")
APPTAINER_EXECUTION_BASE=(
  "${APPTAINER_COMMON[@]}"
  --env PYTHONPATH="${EXECUTION_CONTAINER_PYTHONPATH}"
  "${SIF_PATH}"
)

if [[ "${MODE}" == "preflight" ]]; then
  "${APPTAINER_BASE[@]}" python - <<'PY'
import os
import torch
import vllm
from frontier_cs import SingleEvaluator
import guidance_ttt
import verl

print(f"torch={torch.__version__} cuda={torch.version.cuda}")
print(f"vllm={vllm.__version__}")
print(f"cuda_available={torch.cuda.is_available()} devices={torch.cuda.device_count()}")
print(f"HF_HOME={os.environ['HF_HOME']}")
assert torch.cuda.is_available()
assert torch.cuda.device_count() == int(os.environ["EXPECTED_GPUS"])
PY
  exit 0
fi

if [[ "${MODE}" == "prepare" ]]; then
  "${APPTAINER_BASE[@]}" python -m guidance_ttt.main_erdos \
    --config "${CONFIG}" --prepare-only run.output_dir="${OUTPUT_DIR}"
  exit 0
fi

if [[ "${MODE}" != "run" ]]; then
  echo "Usage: $0 [preflight|prepare|run]" >&2
  exit 2
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  gpu_count="${SLURM_GPUS_ON_NODE:-0}"
  if [[ "${gpu_count}" != "${EXPECTED_GPUS}" ]]; then
    echo "Expected exactly ${EXPECTED_GPUS} allocated GPUs, got SLURM_GPUS_ON_NODE=${gpu_count}" >&2
    exit 1
  fi
fi

rm -f "${ALG_DIR}/problems/0/chk.cc.bin"
"${GOJUDGE}" -mount-conf "${RUNTIME_DIR}/go-judge/mount.yaml" -parallelism 16 -pre-fork 0 \
  > "${LOG_DIR}/gojudge-${RUN_TAG}.log" 2>&1 &
GOJUDGE_PID=$!
PORT=8081 GJ_ADDR=http://127.0.0.1:5050 JUDGE_WORKERS=16 GJ_PARALLELISM=16 \
  SAVE_OUTPUTS=false TESTLIB_INSIDE=/testlib \
  "${NODE_BIN}/node" "${ALG_DIR}/server.js" > "${LOG_DIR}/frontiercs-${RUN_TAG}.log" 2>&1 &
JUDGE_PID=$!
EXECUTION_PID=""
cleanup() {
  if [[ -n "${EXECUTION_PID}" ]]; then kill "${EXECUTION_PID}" 2>/dev/null || true; fi
  kill "${JUDGE_PID}" "${GOJUDGE_PID}" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:8081/health >/dev/null; then break; fi
  if ! kill -0 "${JUDGE_PID}" 2>/dev/null || ! kill -0 "${GOJUDGE_PID}" 2>/dev/null; then
    echo "FrontierCS judge exited during startup" >&2
    exit 1
  fi
  sleep 2
done
curl -fsS http://127.0.0.1:8081/health

if [[ "${START_EXECUTION_SERVER}" == "1" ]]; then
  EXECUTION_VLLM_ARGS=(
    serve "/workspace/guidance/${EXECUTION_MODEL_DIR}"
    --served-model-name "${EXECUTION_MODEL_NAME}"
    --host 127.0.0.1 --port 8000 --dtype "${EXECUTION_VLLM_DTYPE}" --trust-remote-code
    --tensor-parallel-size 1
    --gpu-memory-utilization "${EXECUTION_VLLM_GPU_MEMORY_UTILIZATION}"
    --max-model-len "${EXECUTION_VLLM_MAX_MODEL_LEN}"
    --max-num-seqs "${EXECUTION_VLLM_MAX_NUM_SEQS}"
    --download-dir /workspace/guidance/.hf_cache/hub
  )
  if [[ "${EXECUTION_VLLM_ENFORCE_EAGER}" == "1" ]]; then
    EXECUTION_VLLM_ARGS+=(--enforce-eager)
  fi
  if [[ "${EXECUTION_VLLM_LANGUAGE_MODEL_ONLY}" == "1" ]]; then
    EXECUTION_VLLM_ARGS+=(--language-model-only)
  fi
  if [[ -n "${EXECUTION_VLLM_REASONING_PARSER}" ]]; then
    EXECUTION_VLLM_ARGS+=(--reasoning-parser "${EXECUTION_VLLM_REASONING_PARSER}")
  fi

  APPTAINERENV_CUDA_VISIBLE_DEVICES="${EXECUTION_GPU}" CUDA_VISIBLE_DEVICES="${EXECUTION_GPU}" \
    "${APPTAINER_EXECUTION_BASE[@]}" vllm "${EXECUTION_VLLM_ARGS[@]}" \
    > "${LOG_DIR}/${EXECUTION_LOG_NAME}-vllm-${RUN_TAG}.log" 2>&1 &
  EXECUTION_PID=$!

  for _ in $(seq 1 180); do
    if curl -fsS http://127.0.0.1:8000/v1/models >/dev/null; then break; fi
    if ! kill -0 "${EXECUTION_PID}" 2>/dev/null; then
      echo "Execution vLLM server (${EXECUTION_MODEL_NAME}) exited during startup" >&2
      tail -200 "${LOG_DIR}/${EXECUTION_LOG_NAME}-vllm-${RUN_TAG}.log" >&2
      exit 1
    fi
    sleep 10
  done
  curl -fsS http://127.0.0.1:8000/v1/models
else
  echo "Using external execution API; local execution vLLM startup is disabled."
fi

APPTAINERENV_CUDA_VISIBLE_DEVICES="${TRAINING_GPUS}" CUDA_VISIBLE_DEVICES="${TRAINING_GPUS}" \
  "${APPTAINER_BASE[@]}" python scripts/smoke_polyomino_frontiercs.py
RECIPE_OVERRIDES=()
if [[ -n "${NUM_STEPS_OVERRIDE:-}" ]]; then
  RECIPE_OVERRIDES+=("run.num_steps=${NUM_STEPS_OVERRIDE}")
fi
if [[ -n "${TOTAL_EPOCHS_OVERRIDE:-}" ]]; then
  RECIPE_OVERRIDES+=("run.total_epochs=${TOTAL_EPOCHS_OVERRIDE}")
fi
if [[ -n "${SAVE_FREQ_OVERRIDE:-}" ]]; then
  RECIPE_OVERRIDES+=("run.save_freq=${SAVE_FREQ_OVERRIDE}")
fi
if [[ -n "${GROUPS_PER_BATCH_OVERRIDE:-}" ]]; then
  RECIPE_OVERRIDES+=("ttt.groups_per_batch=${GROUPS_PER_BATCH_OVERRIDE}")
fi
if [[ -n "${GROUP_SIZE_OVERRIDE:-}" ]]; then
  RECIPE_OVERRIDES+=("ttt.group_size=${GROUP_SIZE_OVERRIDE}")
fi
if [[ -n "${EXECUTION_CONCURRENCY_OVERRIDE:-}" ]]; then
  RECIPE_OVERRIDES+=("llm.execution.concurrency=${EXECUTION_CONCURRENCY_OVERRIDE}")
fi
if [[ "${BOOTSTRAP_BEFORE_RUN}" == "1" ]]; then
  APPTAINERENV_CUDA_VISIBLE_DEVICES="${TRAINING_GPUS}" CUDA_VISIBLE_DEVICES="${TRAINING_GPUS}" \
    "${APPTAINER_BASE[@]}" python -m guidance_ttt.main_erdos \
    --config "${CONFIG}" --bootstrap-only run.output_dir="${OUTPUT_DIR}" "${RECIPE_OVERRIDES[@]}"
fi
APPTAINERENV_CUDA_VISIBLE_DEVICES="${TRAINING_GPUS}" CUDA_VISIBLE_DEVICES="${TRAINING_GPUS}" \
  "${APPTAINER_BASE[@]}" python -m guidance_ttt.main_erdos \
  --config "${CONFIG}" run.output_dir="${OUTPUT_DIR}" "${RECIPE_OVERRIDES[@]}"
"${APPTAINER_BASE[@]}" python scripts/validate_polyomino_b200_smoke.py "${OUTPUT_DIR}" \
  --expected-groups "${EXPECTED_GROUPS}" --expected-group-size "${EXPECTED_GROUP_SIZE}" \
  --expected-prompt-mode "${EXPECTED_PROMPT_MODE:-code_delta}"
"${APPTAINER_BASE[@]}" python -m guidance_ttt.run_summary "${OUTPUT_DIR}" --config "${CONFIG}"
