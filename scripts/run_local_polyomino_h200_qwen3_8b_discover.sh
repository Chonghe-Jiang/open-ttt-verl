#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python3.12 >/dev/null 2>&1; then
    PYTHON_BIN="python3.12"
  else
    PYTHON_BIN="python3"
  fi
fi
CONTAINER_PYTHON_BIN="${CONTAINER_PYTHON_BIN:-python3}"
APPTAINER_IMAGE="${APPTAINER_IMAGE:-/orcd/scratch/orcd/010/dwai/vllm.sif}"
CONFIG="${CONFIG:-guidance_ttt/config/polyomino_h200_4gpu_qwen3_8b_discover.yaml}"
FRONTIERCS_DIR="${FRONTIERCS_DIR:-/home/qua/code/reference/Frontier-CS}"
JUDGE_URL="${JUDGE_URL:-http://127.0.0.1:8081}"
JUDGE_PORT="${JUDGE_PORT:-8081}"
JUDGE_WORKERS="${JUDGE_WORKERS:-8}"
JUDGE_APP_DIR="${JUDGE_APP_DIR:-$PWD/.tmp/frontiercs-polyomino-discover-judge}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/guidance_ttt/polyomino_h200_4gpu_qwen3_8b_discover}"
RESET_OUTPUT_DIR="${RESET_OUTPUT_DIR:-0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export TMPDIR="${TMPDIR:-$PWD/.tmp}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PWD/.triton_cache}"

mkdir -p "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TMPDIR" "$TRITON_CACHE_DIR" "$(dirname "$JUDGE_APP_DIR")"

if [[ ! -f "$APPTAINER_IMAGE" ]]; then
  echo "Apptainer image not found: $APPTAINER_IMAGE" >&2
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
  --bind "$HF_HOME:$HF_HOME"
  --bind "$HF_DATASETS_CACHE:$HF_DATASETS_CACHE"
  --bind "$HUGGINGFACE_HUB_CACHE:$HUGGINGFACE_HUB_CACHE"
  --bind "$TMPDIR:$TMPDIR"
  --bind "$TRITON_CACHE_DIR:$TRITON_CACHE_DIR"
  --bind "$FRONTIERCS_DIR:$FRONTIERCS_DIR"
  "$APPTAINER_IMAGE"
)

judge_pid=""
cleanup() {
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

preflight() {
  if [[ "${SKIP_GPU_CHECK:-0}" != "1" ]]; then
    command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi not found" >&2; exit 1; }
    "$PYTHON_BIN" - <<'PY'
import subprocess
import os

names = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
    text=True,
).strip().splitlines()
cuda_visible_devices = [part for part in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if part.strip()]
default_expected = len(cuda_visible_devices) if cuda_visible_devices else 4
expected = int(os.environ.get("EXPECTED_H200_COUNT", default_expected))
h200_names = [name for name in names if "H200" in name]
if len(h200_names) < expected:
    raise SystemExit(f"expected at least {expected} H200 GPU(s), saw: {names}")
print(f"GPU preflight: expected {expected} H200 GPU(s), found {len(h200_names)}")
PY
  fi

  if [[ "${REQUIRE_DOCKER:-0}" == "1" ]]; then
    command -v docker >/dev/null 2>&1 || { echo "docker not found" >&2; exit 1; }
    docker info >/dev/null 2>&1 || { echo "docker daemon is not reachable" >&2; exit 1; }
  fi
  command -v node >/dev/null 2>&1 || { echo "node not found" >&2; exit 1; }
  command -v npm >/dev/null 2>&1 || { echo "npm not found" >&2; exit 1; }
  [[ -d "$FRONTIERCS_DIR/algorithmic/problems/0" ]] || {
    echo "FrontierCS problem 0 not found under $FRONTIERCS_DIR" >&2
    exit 1
  }

  "${APPTAINER_EXEC[@]}" "$CONTAINER_PYTHON_BIN" - <<'PY'
missing = []
for name in ("torch", "vllm", "pandas", "yaml", "omegaconf", "frontier_cs"):
    try:
        __import__(name)
    except Exception as exc:
        missing.append(f"{name}: {type(exc).__name__}: {exc}")
if missing:
    raise SystemExit("Python dependency preflight failed:\n" + "\n".join(missing))
print("Python dependency preflight passed.")
PY
}

start_go_judge() {
  rm -rf "$JUDGE_APP_DIR"
  mkdir -p "$JUDGE_APP_DIR/problems" "$JUDGE_APP_DIR/data" "$JUDGE_APP_DIR/submissions"
  cp "$FRONTIERCS_DIR/algorithmic/package.json" "$JUDGE_APP_DIR/"
  if [[ -f "$FRONTIERCS_DIR/algorithmic/package-lock.json" ]]; then
    cp "$FRONTIERCS_DIR/algorithmic/package-lock.json" "$JUDGE_APP_DIR/"
  fi
  cp "$FRONTIERCS_DIR/algorithmic/server.js" "$JUDGE_APP_DIR/"
  cp -a "$FRONTIERCS_DIR/algorithmic/judge/src" "$JUDGE_APP_DIR/src"
  cp -a "$FRONTIERCS_DIR/algorithmic/judge/include" "$JUDGE_APP_DIR/include"
  cp -a "$FRONTIERCS_DIR/algorithmic/judge/config" "$JUDGE_APP_DIR/config"
  cp -a "$FRONTIERCS_DIR/algorithmic/problems/0" "$JUDGE_APP_DIR/problems/0"
  cp scripts/*_local_gojudge.js "$JUDGE_APP_DIR/src/gojudge.js"
  (cd "$JUDGE_APP_DIR" && npm install --omit=dev --ignore-scripts)

  PORT="$JUDGE_PORT" \
  JUDGE_WORKERS="$JUDGE_WORKERS" \
  TESTLIB_INSIDE="$JUDGE_APP_DIR/include" \
  SAVE_OUTPUTS="${SAVE_OUTPUTS:-false}" \
  node "$JUDGE_APP_DIR/server.js" >"$TMPDIR/frontier_polyomino_discover_judge.log" 2>&1 &
  judge_pid="$!"
  wait_for_url "$JUDGE_URL/problems" "${JUDGE_START_TIMEOUT_S:-120}"
}

run_verifier_smoke() {
  "${APPTAINER_EXEC[@]}" "$CONTAINER_PYTHON_BIN" - "$FRONTIERCS_DIR" "$JUDGE_URL" <<'PY'
import json
import sys

from guidance_ttt.verifier.polyomino import verify_polyomino_solution_text

frontiercs_dir = sys.argv[1]
judge_url = sys.argv[2]
solution = r'''
#include <bits/stdc++.h>
using namespace std;
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    if (!(cin >> n)) return 0;
    vector<vector<pair<int,int>>> pieces(n);
    vector<int> minx(n), miny(n), maxx(n), maxy(n);
    long long W = 0;
    int H = 0;
    for (int i = 0; i < n; ++i) {
        int k;
        cin >> k;
        pieces[i].resize(k);
        minx[i] = miny[i] = INT_MAX;
        maxx[i] = maxy[i] = INT_MIN;
        for (int j = 0; j < k; ++j) {
            cin >> pieces[i][j].first >> pieces[i][j].second;
            minx[i] = min(minx[i], pieces[i][j].first);
            miny[i] = min(miny[i], pieces[i][j].second);
            maxx[i] = max(maxx[i], pieces[i][j].first);
            maxy[i] = max(maxy[i], pieces[i][j].second);
        }
        W += maxx[i] - minx[i] + 1;
        H = max(H, maxy[i] - miny[i] + 1);
    }
    cout << W << ' ' << H << '\n';
    long long x = 0;
    for (int i = 0; i < n; ++i) {
        cout << x - minx[i] << ' ' << -miny[i] << " 0 0\n";
        x += maxx[i] - minx[i] + 1;
    }
    return 0;
}
'''
result = verify_polyomino_solution_text(
    f"<solution>\\n```cpp\\n{solution}\\n```\\n</solution>",
    problem_id="0",
    config={
        "base_dir": frontiercs_dir,
        "judge_url": judge_url,
        "n_cases": 1,
        "total_timeout_s": 120,
    },
)
print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
if not result.valid:
    raise SystemExit("polyomino verifier smoke failed")
PY
}

run_config_compose_smoke() {
  "${APPTAINER_EXEC[@]}" "$CONTAINER_PYTHON_BIN" - \
    "$CONFIG" \
    "$OUTPUT_DIR-smoke" \
    "${EXPECTED_ROLLOUT_N:-32}" \
    "${EXPECTED_H200_COUNT:-4}" \
    "${EXPECTED_TENSOR_MODEL_PARALLEL_SIZE:-4}" \
    "$@" <<'PY'
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from guidance_ttt.main_erdos import (
    _default_verl_config_dir,
    apply_recipe_overrides,
    build_verl_overrides,
    load_recipe_config,
    prepare_run,
    split_overrides,
)

config_path = sys.argv[1]
smoke_output_dir = sys.argv[2]
expected_rollout_n = int(sys.argv[3])
expected_gpus = int(sys.argv[4])
expected_tp = int(sys.argv[5])
recipe_overrides, verl_overrides = split_overrides(sys.argv[6:])
config = apply_recipe_overrides(load_recipe_config(config_path), recipe_overrides)
config["run"]["output_dir"] = smoke_output_dir
prepared = prepare_run(config)
overrides = build_verl_overrides(config, prepared, verl_overrides)
with initialize_config_dir(config_dir=str(_default_verl_config_dir().resolve()), version_base=None):
    verl_config = compose(config_name="ppo_trainer", overrides=overrides)
OmegaConf.resolve(verl_config)
assert verl_config.algorithm.adv_estimator == "entropic_adaptive_beta"
assert verl_config.actor_rollout_ref.rollout.agent.default_agent_loop == "polyomino_discover_task"
assert int(verl_config.actor_rollout_ref.rollout.n) == expected_rollout_n
assert int(verl_config.trainer.n_gpus_per_node) == expected_gpus
assert int(verl_config.actor_rollout_ref.rollout.tensor_model_parallel_size) == expected_tp
print("Config compose smoke passed:", Path(prepared["agent_loop_config"]))
PY
}

preflight
start_go_judge
run_verifier_smoke
run_config_compose_smoke "$@"

if [[ "${SMOKE_ONLY:-0}" == "1" ]]; then
  echo "Local Polyomino direct-discover smoke passed."
  exit 0
fi

if [[ "$RESET_OUTPUT_DIR" == "1" ]]; then
  rm -rf "$OUTPUT_DIR"
fi

"${APPTAINER_EXEC[@]}" "$CONTAINER_PYTHON_BIN" -m guidance_ttt.main_erdos --config "$CONFIG" "run.output_dir=$OUTPUT_DIR" "$@"
"${APPTAINER_EXEC[@]}" "$CONTAINER_PYTHON_BIN" -m guidance_ttt.run_summary "$OUTPUT_DIR" --config "$CONFIG" || true
