#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SIF_PATH="${SIF_PATH:-/work/mit/ppliang_mit/chonghej/open-ttt-verl/containers/open-ttt-verl-ttt-vllm.sif}"
CONFIG="${CONFIG:-guidance_ttt/config/trimul_b200_1gpu_qwen3_8b_evolvent_glm52_thinking_cache_group4_1step.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/guidance_ttt/trimul_b200_glm52_thinking_cache_${SLURM_JOB_ID:-manual}}"
EXPECTED_GPUS="${EXPECTED_GPUS:-1}"
EXPECTED_CHILDREN="${EXPECTED_CHILDREN:-4}"
EXPECTED_GROUPS="${EXPECTED_GROUPS:-1}"
EXPECTED_GROUP_SIZE="${EXPECTED_GROUP_SIZE:-4}"
MINIMUM_CACHE_HITS="${MINIMUM_CACHE_HITS:-1}"
REQUIRE_TRAINING_UPDATE="${REQUIRE_TRAINING_UPDATE:-0}"
MODE="${1:-run}"

cd "${REPO_ROOT}"
mkdir -p outputs/slurm .hf_cache .tmp .triton_cache .ray_tmp .apptainer_home/.cache

required_artifacts=(
  "${SIF_PATH}"
  "models/Qwen3-8B/config.json"
  "guidance_ttt/seeds/trimul/glm52_scratch_bootstrap_library.json"
  ".secrets/evolvent_api_key"
  ".secrets/trimul_judge_token"
  ".runtime/trimul_judge_url"
)
for required in "${required_artifacts[@]}"; do
  if [[ ! -s "${required}" ]]; then
    echo "Missing required runtime artifact: ${required}" >&2
    exit 1
  fi
done
for secret in .secrets/evolvent_api_key .secrets/trimul_judge_token; do
  if [[ "$(stat -c '%a' "${secret}")" != "600" ]]; then
    echo "Secret must have mode 600: ${secret}" >&2
    exit 1
  fi
done

export APPTAINER_CACHEDIR="${REPO_ROOT}/.apptainer_cache"
export APPTAINER_TMPDIR="${REPO_ROOT}/.apptainer_tmp"
export HF_HOME="${REPO_ROOT}/.hf_cache"
export TMPDIR="${REPO_ROOT}/.tmp"
export TRITON_CACHE_DIR="${REPO_ROOT}/.triton_cache"
export RAY_TMPDIR="${REPO_ROOT}/.ray_tmp"
export APPTAINERENV_TRIMUL_JUDGE_URL
export APPTAINERENV_TRIMUL_JUDGE_TOKEN
APPTAINERENV_TRIMUL_JUDGE_URL="$(<.runtime/trimul_judge_url)"
APPTAINERENV_TRIMUL_JUDGE_TOKEN="$(<.secrets/trimul_judge_token)"

APPTAINER_BASE=(
  apptainer exec --nv --cleanenv
  --home "${REPO_ROOT}/.apptainer_home:/container_home"
  --bind "${REPO_ROOT}:/workspace/guidance"
  --pwd /workspace/guidance
  --env PYTHONPATH=/workspace/guidance
  --env HF_HOME=/workspace/guidance/.hf_cache
  --env HF_DATASETS_CACHE=/workspace/guidance/.hf_cache/datasets
  --env HUGGINGFACE_HUB_CACHE=/workspace/guidance/.hf_cache/hub
  --env TRANSFORMERS_CACHE=/workspace/guidance/.hf_cache/hub
  --env TMPDIR=/workspace/guidance/.tmp
  --env TRITON_CACHE_DIR=/workspace/guidance/.triton_cache
  --env RAY_TMPDIR=/workspace/guidance/.ray_tmp
  --env XDG_CACHE_HOME=/container_home/.cache
  --env HYDRA_FULL_ERROR=1
  --env RAY_DEDUP_LOGS=0
  --env RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
  --env VLLM_NO_USAGE_STATS=1
  --env VLLM_WORKER_MULTIPROC_METHOD=spawn
  --env NCCL_IB_DISABLE=1
  --env TOKENIZERS_PARALLELISM=false
  "${SIF_PATH}"
)

if [[ "${MODE}" == "preflight" ]]; then
  "${APPTAINER_BASE[@]}" python - <<'PY'
import os
import torch
import vllm
import guidance_ttt
import verl

print(f"torch={torch.__version__} cuda={torch.version.cuda}")
print(f"vllm={vllm.__version__}")
print(f"cuda_available={torch.cuda.is_available()} devices={torch.cuda.device_count()}")
print(f"judge_url_configured={bool(os.environ.get('TRIMUL_JUDGE_URL'))}")
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
assert os.environ.get("TRIMUL_JUDGE_URL")
assert os.environ.get("TRIMUL_JUDGE_TOKEN")
PY
  exit 0
fi

if [[ "${MODE}" != "run" ]]; then
  echo "Usage: $0 [preflight|run]" >&2
  exit 2
fi
if [[ -n "${SLURM_JOB_ID:-}" && "${SLURM_GPUS_ON_NODE:-0}" != "${EXPECTED_GPUS}" ]]; then
  echo "Expected ${EXPECTED_GPUS} GPU, got SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-0}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
TRAINER_LOG="${OUTPUT_DIR}/trainer.log"
APPTAINERENV_CUDA_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
  "${APPTAINER_BASE[@]}" python -m guidance_ttt.main_erdos \
  --config "${CONFIG}" run.output_dir="${OUTPUT_DIR}" 2>&1 | tee "${TRAINER_LOG}"

"${APPTAINER_BASE[@]}" python scripts/validate_trimul_evolvent_cache_smoke.py \
  "${OUTPUT_DIR}" \
  --expected-children "${EXPECTED_CHILDREN}" \
  --expected-groups "${EXPECTED_GROUPS}" \
  --expected-group-size "${EXPECTED_GROUP_SIZE}" \
  --minimum-cache-hits "${MINIMUM_CACHE_HITS}"

if [[ "${REQUIRE_TRAINING_UPDATE}" == "1" ]]; then
  "${APPTAINER_BASE[@]}" python scripts/validate_trimul_training_update.py "${TRAINER_LOG}"
fi

if find "${OUTPUT_DIR}" -type d \( -name 'global_step_*' -o -name 'checkpoint-*' \) -print -quit | grep -q .; then
  echo "Unexpected checkpoint directory found despite save_freq=-1" >&2
  exit 1
fi
echo "TRIMUL_EVOLVENT_CACHE_SMOKE_COMPLETE"
