#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FRONTIER_DIR="${REPO_ROOT}/reference/Frontier-CS"
RUNTIME_DIR="${REPO_ROOT}/.runtime"
NODE_VERSION="${NODE_VERSION:-20.19.4}"
NODE_DIR="${RUNTIME_DIR}/node-v${NODE_VERSION}-linux-x64"
GOJUDGE="${RUNTIME_DIR}/go-judge/bin/go-judge"
SIF_PATH="${SIF_PATH:-/work/mit/ppliang_mit/chonghej/open-ttt-verl/containers/open-ttt-verl-ttt-vllm.sif}"
FRONTIER_COMMIT="${FRONTIER_COMMIT:-6d597dfb60be9e592881aef051b94e30d197c436}"

mkdir -p "${RUNTIME_DIR}" "${REPO_ROOT}/models" "${REPO_ROOT}/.hf_cache"

if [[ ! -d "${FRONTIER_DIR}/.git" ]]; then
  git clone --filter=blob:none --no-checkout https://github.com/FrontierCS/Frontier-CS.git "${FRONTIER_DIR}"
fi
git -C "${FRONTIER_DIR}" sparse-checkout init --cone
git -C "${FRONTIER_DIR}" sparse-checkout set --skip-checks \
  pyproject.toml README.md src \
  algorithmic/package.json algorithmic/package-lock.json algorithmic/server.js \
  algorithmic/entrypoint.sh algorithmic/judge algorithmic/problems/0
git -C "${FRONTIER_DIR}" fetch --depth 1 origin "${FRONTIER_COMMIT}"
git -C "${FRONTIER_DIR}" checkout --detach "${FRONTIER_COMMIT}"
FRONTIER_PATCH="${SCRIPT_DIR}/frontiercs_optional_generation_sdks.patch"
if ! git -C "${FRONTIER_DIR}" apply --reverse --check "${FRONTIER_PATCH}" >/dev/null 2>&1; then
  git -C "${FRONTIER_DIR}" apply "${FRONTIER_PATCH}"
fi

if [[ ! -x "${NODE_DIR}/bin/node" ]]; then
  archive="${RUNTIME_DIR}/node-v${NODE_VERSION}-linux-x64.tar.xz"
  curl -fL "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz" -o "${archive}"
  tar -xJf "${archive}" -C "${RUNTIME_DIR}"
  rm -f "${archive}"
fi

export PATH="${NODE_DIR}/bin:${PATH}"
if [[ ! -d "${FRONTIER_DIR}/algorithmic/node_modules" ]]; then
  npm --prefix "${FRONTIER_DIR}/algorithmic" install --omit=dev --ignore-scripts
fi
ln -sfn judge/src "${FRONTIER_DIR}/algorithmic/src"
ln -sfn judge/include "${FRONTIER_DIR}/algorithmic/include"
ln -sfn judge/config "${FRONTIER_DIR}/algorithmic/config"
mkdir -p "${FRONTIER_DIR}/algorithmic/data" "${FRONTIER_DIR}/algorithmic/submissions"

if [[ ! -x "${GOJUDGE}" ]]; then
  mkdir -p "$(dirname "${GOJUDGE}")"
  curl -fL \
    https://github.com/criyle/go-judge/releases/download/v1.11.1/go-judge_1.11.1_linux_amd64v2 \
    -o "${GOJUDGE}"
  chmod +x "${GOJUDGE}"
fi
sed "s|__TESTLIB_PATH__|${FRONTIER_DIR}/algorithmic/judge/include|g" \
  "${SCRIPT_DIR}/frontiercs_gojudge_mount.yaml.in" > "${RUNTIME_DIR}/go-judge/mount.yaml"

if [[ "${SKIP_MODELS:-0}" != "1" ]]; then
  if [[ ! -f "${SIF_PATH}" ]]; then
    echo "Apptainer image not found: ${SIF_PATH}" >&2
    exit 1
  fi
  export APPTAINER_CACHEDIR="${REPO_ROOT}/.apptainer_cache"
  export APPTAINER_TMPDIR="${REPO_ROOT}/.apptainer_tmp"
  mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"
  for spec in "Qwen/Qwen3-8B:Qwen3-8B" "openai/gpt-oss-120b:gpt-oss-120b"; do
    model_id="${spec%%:*}"
    local_name="${spec##*:}"
    if [[ ! -f "${REPO_ROOT}/models/${local_name}/.download-complete" ]]; then
      apptainer exec --cleanenv \
        --bind "${REPO_ROOT}:/workspace/guidance" \
        --pwd /workspace/guidance \
        --env HF_HOME=/workspace/guidance/.hf_cache \
        --env HUGGINGFACE_HUB_CACHE=/workspace/guidance/.hf_cache/hub \
        --env HF_XET_CACHE=/workspace/guidance/.hf_cache/xet \
        --env HF_HUB_ENABLE_HF_TRANSFER=1 \
        --env HF_XET_HIGH_PERFORMANCE=0 \
        --env HF_XET_NUM_CONCURRENT_RANGE_GETS=2 \
        "${SIF_PATH}" hf download "${model_id}" \
        --local-dir "models/${local_name}" --max-workers 1 \
        --exclude 'metal/*' 'original/*'
      touch "${REPO_ROOT}/models/${local_name}/.download-complete"
    fi
  done
fi

echo "B200 runtime ready under ${REPO_ROOT}"
