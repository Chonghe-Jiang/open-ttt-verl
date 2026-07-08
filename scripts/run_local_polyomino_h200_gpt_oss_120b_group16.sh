#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
APPTAINER_IMAGE="${APPTAINER_IMAGE:-/orcd/scratch/orcd/010/dwai/vllm.sif}"
CONFIG="${CONFIG:-guidance_ttt/config/polyomino_modal_h200_3gpu_gpt_oss_120b_group16_h200_tuned.yaml}"
HOST_RUNS_DIR="${HOST_RUNS_DIR:-$PWD/.modal_local_runs}"
HOST_CACHE_DIR="${HOST_CACHE_DIR:-$PWD/.modal_local_cache}"
HOST_TMP_DIR="${HOST_TMP_DIR:-$PWD/.modal_local_tmp}"
HOST_TRITON_DIR="${HOST_TRITON_DIR:-$PWD/.modal_local_triton}"
HOST_FRONTIERCS_DIR="${HOST_FRONTIERCS_DIR:-/home/qua/code/reference/Frontier-CS}"
REMOTE_OUTPUT_DIR="${REMOTE_OUTPUT_DIR:-/runs/guidance_ttt/polyomino_modal_h200_3gpu_gpt_oss_120b_group16_h200_tuned_50step}"
JUDGE_URL="${JUDGE_URL:-http://127.0.0.1:8081}"
JUDGE_APP_DIR="${JUDGE_APP_DIR:-$HOST_TMP_DIR/frontiercs-local-judge}"
RESET_OUTPUT_DIR="${RESET_OUTPUT_DIR:-1}"
TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1}"
EXECUTION_CUDA_VISIBLE_DEVICES="${EXECUTION_CUDA_VISIBLE_DEVICES:-2}"
FLASH_ATTN_WHEEL_URL="${FLASH_ATTN_WHEEL_URL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl}"
FLASH_ATTN_WHEEL="${FLASH_ATTN_WHEEL:-$HOST_CACHE_DIR/wheels/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-1}"
START_FRONTIER_JUDGE="${START_FRONTIER_JUDGE:-1}"
START_EXECUTION_SERVER="${START_EXECUTION_SERVER:-1}"
RUN_XML_PROBE="${RUN_XML_PROBE:-1}"

export HF_HOME="/cache/huggingface"
export HF_DATASETS_CACHE="/cache/huggingface/datasets"
export HUGGINGFACE_HUB_CACHE="/cache/huggingface/hub"
export TRANSFORMERS_CACHE="/cache/huggingface/hub"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0 9.0a}"
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export RAY_raylet_start_wait_time_s="${RAY_raylet_start_wait_time_s:-300}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export NCCL_NET="${NCCL_NET:-Socket}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

mkdir -p \
  "$HOST_RUNS_DIR" \
  "$HOST_CACHE_DIR/huggingface/datasets" \
  "$HOST_CACHE_DIR/huggingface/hub" \
  "$HOST_CACHE_DIR/wheels" \
  "$HOST_TMP_DIR" \
  "$HOST_TRITON_DIR"

if [[ ! -f "$APPTAINER_IMAGE" ]]; then
  echo "Apptainer image not found: $APPTAINER_IMAGE" >&2
  exit 1
fi
if [[ ! -d "$HOST_FRONTIERCS_DIR/algorithmic" ]]; then
  echo "FrontierCS algorithmic dir not found: $HOST_FRONTIERCS_DIR/algorithmic" >&2
  exit 1
fi

if ! command -v apptainer >/dev/null 2>&1 && ! [[ -x /usr/bin/apptainer ]]; then
  for module_init in /etc/profile.d/modules.sh /usr/share/Modules/init/bash /usr/share/lmod/lmod/init/bash; do
    if [[ -f "$module_init" ]]; then
      # shellcheck disable=SC1090
      source "$module_init"
      break
    fi
  done
  if command -v module >/dev/null 2>&1; then
    module load apptainer/1.4.2 >/dev/null 2>&1 || true
  fi
fi

if command -v apptainer >/dev/null 2>&1; then
  CONTAINER_RUNTIME="$(command -v apptainer)"
elif [[ -x /usr/bin/apptainer ]]; then
  CONTAINER_RUNTIME="/usr/bin/apptainer"
else
  echo "No apptainer runtime found" >&2
  exit 127
fi

APPTAINER_EXEC=(
  "$CONTAINER_RUNTIME" exec --nv
  --bind "$PWD:$PWD"
  --bind "$HOST_RUNS_DIR:/runs"
  --bind "$HOST_CACHE_DIR:/cache"
  --bind "$HOST_TMP_DIR:/tmp/guidance-ttt-local"
  --bind "$HOST_TRITON_DIR:/cache/triton"
  --bind "$HOST_FRONTIERCS_DIR:/opt/Frontier-CS"
  "$APPTAINER_IMAGE"
)

judge_pid=""
server_pid=""
cleanup() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  if [[ -n "$judge_pid" ]] && kill -0 "$judge_pid" 2>/dev/null; then
    kill "$judge_pid" 2>/dev/null || true
    wait "$judge_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

wait_for_url() {
  local url="$1"
  local timeout_s="${2:-120}"
  "$PYTHON_BIN" - "$url" "$timeout_s" <<'PY'
import sys
import time
import urllib.request

url = sys.argv[1]
deadline = time.time() + int(sys.argv[2])
last = None
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            print(f"ready: {url} -> {response.status}", flush=True)
            raise SystemExit(0)
    except Exception as exc:
        last = exc
        time.sleep(2)
raise SystemExit(f"timed out waiting for {url}: {last}")
PY
}

wait_for_tcp() {
  local host="$1"
  local port="$2"
  local timeout_s="${3:-1800}"
  "$PYTHON_BIN" - "$host" "$port" "$timeout_s" <<'PY'
import socket
import sys
import time

host = sys.argv[1]
port = int(sys.argv[2])
deadline = time.time() + int(sys.argv[3])
last = None
while time.time() < deadline:
    try:
        with socket.create_connection((host, port), timeout=5):
            print(f"tcp ready: {host}:{port}", flush=True)
            raise SystemExit(0)
    except Exception as exc:
        last = exc
        time.sleep(2)
raise SystemExit(f"timed out waiting for {host}:{port}: {last}")
PY
}

download_flash_attn_wheel() {
  if [[ -f "$FLASH_ATTN_WHEEL" ]]; then
    return
  fi
  echo "Downloading flash-attn wheel to $FLASH_ATTN_WHEEL"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 -o "$FLASH_ATTN_WHEEL" "$FLASH_ATTN_WHEEL_URL"
  else
    "$PYTHON_BIN" - "$FLASH_ATTN_WHEEL_URL" "$FLASH_ATTN_WHEEL" <<'PY'
import sys
import urllib.request

urllib.request.urlretrieve(sys.argv[1], sys.argv[2])
PY
  fi
}

ensure_training_packages() {
  if ! "${APPTAINER_EXEC[@]}" "$PYTHON_BIN" - <<'PY'
checks = [
    ("torch", "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"),
    ("vllm", "import vllm; print('vllm', vllm.__version__)"),
    ("flash_attn", "import flash_attn; print('flash_attn', getattr(flash_attn, '__version__', 'unknown'))"),
    ("frontier_cs", "import frontier_cs; print('frontier_cs', getattr(frontier_cs, '__version__', 'unknown'))"),
]
missing = []
for name, code in checks:
    try:
        exec(code)
    except Exception as exc:
        print(f"{name} ERROR {type(exc).__name__}: {exc}", flush=True)
        missing.append(name)
if missing:
    raise SystemExit(1)
PY
  then
    if [[ "$INSTALL_FLASH_ATTN" != "1" ]]; then
      echo "Required package check failed and INSTALL_FLASH_ATTN is not 1" >&2
      exit 1
    fi
    download_flash_attn_wheel
    if ! "${APPTAINER_EXEC[@]}" "$PYTHON_BIN" -m pip install --user --no-build-isolation "$FLASH_ATTN_WHEEL"; then
      echo "Prebuilt flash-attn wheel install failed; compiling flash-attn==2.8.3 locally." >&2
      "${APPTAINER_EXEC[@]}" "$PYTHON_BIN" -m pip install --user --no-build-isolation "flash-attn==2.8.3"
    fi
    "${APPTAINER_EXEC[@]}" "$PYTHON_BIN" - <<'PY'
import flash_attn
import torch
import vllm
print("flash_attn", getattr(flash_attn, "__version__", "unknown"))
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("vllm", vllm.__version__)
PY
  fi
}

start_frontier_judge() {
  rm -rf "$JUDGE_APP_DIR"
  mkdir -p "$JUDGE_APP_DIR/problems" "$JUDGE_APP_DIR/data" "$JUDGE_APP_DIR/submissions"
  cp "$HOST_FRONTIERCS_DIR/algorithmic/package.json" "$JUDGE_APP_DIR/"
  if [[ -f "$HOST_FRONTIERCS_DIR/algorithmic/package-lock.json" ]]; then
    cp "$HOST_FRONTIERCS_DIR/algorithmic/package-lock.json" "$JUDGE_APP_DIR/"
  fi
  cp "$HOST_FRONTIERCS_DIR/algorithmic/server.js" "$JUDGE_APP_DIR/"
  cp -a "$HOST_FRONTIERCS_DIR/algorithmic/judge/src" "$JUDGE_APP_DIR/src"
  cp -a "$HOST_FRONTIERCS_DIR/algorithmic/judge/include" "$JUDGE_APP_DIR/include"
  cp -a "$HOST_FRONTIERCS_DIR/algorithmic/judge/config" "$JUDGE_APP_DIR/config"
  cp -a "$HOST_FRONTIERCS_DIR/algorithmic/problems/0" "$JUDGE_APP_DIR/problems/0"
  cp scripts/modal_local_gojudge.js "$JUDGE_APP_DIR/src/gojudge.js"
  (cd "$JUDGE_APP_DIR" && npm install --omit=dev --ignore-scripts)

  PORT="${JUDGE_PORT:-8081}" \
  JUDGE_WORKERS="${JUDGE_WORKERS:-8}" \
  TESTLIB_INSIDE="$JUDGE_APP_DIR/include" \
  SAVE_OUTPUTS="${SAVE_OUTPUTS:-false}" \
  node "$JUDGE_APP_DIR/server.js" >"$HOST_RUNS_DIR/frontier_judge.log" 2>&1 &
  judge_pid="$!"
  wait_for_url "$JUDGE_URL/problems" "${JUDGE_START_TIMEOUT_S:-120}"
}

run_polyomino_verifier_smoke() {
  "${APPTAINER_EXEC[@]}" "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

from guidance_ttt.verifier.polyomino import verify_polyomino_solution_text

seed_path = Path("guidance_ttt/seeds/polyomino_packing/openrouter_gpt55_bootstrap_library.json")
data = json.loads(seed_path.read_text())
entries = [entry for entry in (data.get("entries") or {}).values() if isinstance(entry, dict)]
if not entries:
    raise SystemExit(f"no seed entries found in {seed_path}")
solution = str(entries[0].get("solution") or "")
result = verify_polyomino_solution_text(
    f"<solution>\n```cpp\n{solution}\n```\n</solution>",
    problem_id="0",
    config={
        "base_dir": "/opt/Frontier-CS",
        "judge_url": "http://127.0.0.1:8081",
        "n_cases": 1,
        "total_timeout_s": 120,
    },
)
summary = {
    "status": result.status,
    "valid": result.valid,
    "raw_score": result.raw_score,
    "message": result.message,
}
print(json.dumps(summary, indent=2), flush=True)
if not result.valid:
    raise SystemExit(f"polyomino verifier smoke failed: {summary}")
PY
}

start_execution_server() {
  CUDA_VISIBLE_DEVICES="$EXECUTION_CUDA_VISIBLE_DEVICES" \
  VLLM_NO_USAGE_STATS=1 \
  HF_HOME="$HF_HOME" \
  HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
  HUGGINGFACE_HUB_CACHE="$HUGGINGFACE_HUB_CACHE" \
  TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
  "${APPTAINER_EXEC[@]}" vllm serve openai/gpt-oss-120b \
    --host 127.0.0.1 \
    --port 8000 \
    --dtype auto \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.88 \
    --max-model-len 32768 \
    --max-num-seqs 8 \
    --enforce-eager \
    --download-dir /cache/huggingface/hub \
    >"$HOST_RUNS_DIR/gpt_oss_120b_vllm.log" 2>&1 &
  server_pid="$!"
  wait_for_tcp 127.0.0.1 8000 "${VLLM_START_TIMEOUT_S:-1800}"
}

run_xml_probe() {
  "${APPTAINER_EXEC[@]}" "$PYTHON_BIN" - <<'PY'
import json
import urllib.request

payload = {
    "model": "openai/gpt-oss-120b",
    "messages": [
        {"role": "system", "content": "You are an execution model. Output exactly the requested XML blocks and no extra text."},
        {"role": "user", "content": "Return exactly three XML blocks: <execution_thinking>one sentence</execution_thinking><solution>```cpp\\nint main(){return 0;}\\n```</solution><summary>one sentence</summary>"},
    ],
    "temperature": 0.0,
    "max_tokens": 512,
}
request = urllib.request.Request(
    "http://127.0.0.1:8000/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Authorization": "Bearer local-vllm", "Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(request, timeout=600) as response:
    data = json.loads(response.read().decode())
text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
print(json.dumps({"text_len": len(text), "has_summary_close": "</summary>" in text, "tail": text[-300:]}, indent=2))
if "</summary>" not in text:
    raise SystemExit("execution XML probe did not close summary")
PY
}

run_prepare_smoke() {
  local smoke_output="/runs/guidance_ttt/polyomino_modal_h200_3gpu_gpt_oss_120b_group16_h200_tuned_cpu_smoke"
  rm -rf "$HOST_RUNS_DIR/guidance_ttt/polyomino_modal_h200_3gpu_gpt_oss_120b_group16_h200_tuned_cpu_smoke"
  "${APPTAINER_EXEC[@]}" "$PYTHON_BIN" -m guidance_ttt.main_erdos \
    --config "$CONFIG" \
    --prepare-only \
    "run.output_dir=$smoke_output"
}

if [[ "${SMOKE_ONLY:-0}" == "1" ]]; then
  ensure_training_packages
  start_frontier_judge
  run_polyomino_verifier_smoke
  run_prepare_smoke
  echo "Local Modal-equivalent smoke passed."
  exit 0
fi

ensure_training_packages
run_prepare_smoke

if [[ "$RESET_OUTPUT_DIR" == "1" ]]; then
  rm -rf "$HOST_RUNS_DIR/guidance_ttt/polyomino_modal_h200_3gpu_gpt_oss_120b_group16_h200_tuned_50step"
fi

if [[ "$START_EXECUTION_SERVER" == "1" ]]; then
  start_execution_server
else
  wait_for_tcp 127.0.0.1 8000 "${VLLM_START_TIMEOUT_S:-30}"
fi
if [[ "$RUN_XML_PROBE" == "1" ]]; then
  run_xml_probe
fi
if [[ "$START_FRONTIER_JUDGE" == "1" ]]; then
  start_frontier_judge
else
  wait_for_url "$JUDGE_URL/problems" "${JUDGE_START_TIMEOUT_S:-10}"
fi

CUDA_VISIBLE_DEVICES="$TRAIN_CUDA_VISIBLE_DEVICES" \
"${APPTAINER_EXEC[@]}" "$PYTHON_BIN" -m guidance_ttt.main_erdos --config "$CONFIG" "$@"

"${APPTAINER_EXEC[@]}" "$PYTHON_BIN" -m guidance_ttt.run_summary "$REMOTE_OUTPUT_DIR" --config "$CONFIG" || true
