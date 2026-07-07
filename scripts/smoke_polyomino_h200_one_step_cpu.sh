#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
  eval "$(
    python3 - <<'PY'
from pathlib import Path
import os
import shlex

keys = {"ENDPOINT", "API_KEY"}
values = {}
for raw_line in Path(".env").read_text().splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    key, value = line.split("=", 1)
    key = key.strip()
    if key.startswith("export "):
        key = key.removeprefix("export ").strip()
    if key in keys and key not in os.environ:
        values[key] = value.strip().strip("'\"")
for key in sorted(values):
    if values[key]:
        print(f"export {key}={shlex.quote(values[key])}")
PY
  )"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
APPTAINER_IMAGE="${APPTAINER_IMAGE:-/orcd/scratch/orcd/010/dwai/vllm.sif}"
CONFIG="${CONFIG:-guidance_ttt/config/polyomino_h200_4gpu_single_summary.yaml}"
OUTPUT_DIR="${CPU_SMOKE_OUTPUT_DIR:-outputs/guidance_ttt/polyomino_h200_4gpu_one_step_30k_cpu_smoke}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
EXECUTION_MODEL="${EXECUTION_MODEL:-claude-sub2api-opus-4-8}"
FRONTIERCS_DIR="${FRONTIERCS_DIR:-/home/qua/code/reference/Frontier-CS}"
JUDGE_URL="${JUDGE_URL:-http://127.0.0.1:8081}"
BOOTSTRAP_LOG="${BOOTSTRAP_LOG:-$OUTPUT_DIR/bootstrap.log}"

export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export TMPDIR="${TMPDIR:-$PWD/.tmp}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PWD/.triton_cache}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

if [[ -z "${ENDPOINT:-}" || -z "${API_KEY:-}" ]]; then
  echo "CPU smoke requires ENDPOINT and API_KEY in the environment or .env." >&2
  exit 1
fi

mkdir -p "$HF_HOME/datasets" "$HF_HOME/hub" "$TMPDIR" "$TRITON_CACHE_DIR" "$(dirname "$OUTPUT_DIR")"

PYTHON_RUN=("$PYTHON_BIN")
if [[ -n "$APPTAINER_IMAGE" ]]; then
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
  if [[ -n "${CONTAINER_RUNTIME_BIN:-}" ]]; then
    CONTAINER_RUNTIME="$CONTAINER_RUNTIME_BIN"
  elif command -v apptainer >/dev/null 2>&1; then
    CONTAINER_RUNTIME="$(command -v apptainer)"
  elif [[ -x /usr/bin/apptainer ]]; then
    CONTAINER_RUNTIME="/usr/bin/apptainer"
  elif command -v singularity >/dev/null 2>&1; then
    CONTAINER_RUNTIME="$(command -v singularity)"
  elif [[ -x /usr/bin/singularity ]]; then
    CONTAINER_RUNTIME="/usr/bin/singularity"
  else
    echo "No apptainer or singularity runtime found; set CONTAINER_RUNTIME_BIN" >&2
    exit 127
  fi
  export CONTAINER_RUNTIME_BIN="$CONTAINER_RUNTIME"
  PYTHON_RUN=("$CONTAINER_RUNTIME_BIN" exec --bind /home/qua/code:/home/qua/code "$APPTAINER_IMAGE" "$PYTHON_BIN")
fi

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

start_frontier_judge() {
  if [[ ! -d "$FRONTIERCS_DIR/algorithmic" ]]; then
    echo "FrontierCS algorithmic dir not found: $FRONTIERCS_DIR/algorithmic" >&2
    exit 1
  fi
  if ! command -v node >/dev/null 2>&1; then
    echo "node is required for CPU smoke" >&2
    exit 1
  fi

  local app_dir="${JUDGE_APP_DIR:-$TMPDIR/frontiercs-local-judge-cpu-smoke}"
  rm -rf "$app_dir"
  mkdir -p "$app_dir/problems" "$app_dir/data" "$app_dir/submissions" "$OUTPUT_DIR"
  cp "$FRONTIERCS_DIR/algorithmic/package.json" "$app_dir/"
  if [[ -f "$FRONTIERCS_DIR/algorithmic/package-lock.json" ]]; then
    cp "$FRONTIERCS_DIR/algorithmic/package-lock.json" "$app_dir/"
  fi
  cp "$FRONTIERCS_DIR/algorithmic/server.js" "$app_dir/"
  cp -a "$FRONTIERCS_DIR/algorithmic/judge/src" "$app_dir/src"
  cp -a "$FRONTIERCS_DIR/algorithmic/judge/include" "$app_dir/include"
  cp -a "$FRONTIERCS_DIR/algorithmic/judge/config" "$app_dir/config"
  cp -a "$FRONTIERCS_DIR/algorithmic/problems/0" "$app_dir/problems/0"
  cp scripts/modal_local_gojudge.js "$app_dir/src/gojudge.js"

  if [[ ! -d "$app_dir/node_modules" ]]; then
    (cd "$app_dir" && npm install --omit=dev --ignore-scripts)
  fi

  PORT="${JUDGE_PORT:-8081}" \
  JUDGE_WORKERS="${JUDGE_WORKERS:-2}" \
  TESTLIB_INSIDE="$app_dir/include" \
  SAVE_OUTPUTS="${SAVE_OUTPUTS:-false}" \
  node "$app_dir/server.js" >"$OUTPUT_DIR.frontier_judge.log" 2>&1 &
  judge_pid="$!"
  wait_for_url "$JUDGE_URL/problems" "${JUDGE_START_TIMEOUT_S:-120}"
}

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"
start_frontier_judge

"${PYTHON_RUN[@]}" - "$CONFIG" "$OUTPUT_DIR" "$MODEL_PATH" "$EXECUTION_MODEL" "$FRONTIERCS_DIR" "$JUDGE_URL" <<'PY'
import json
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
)

config_path, output_dir, model_path, execution_model, frontiercs_dir, judge_url = sys.argv[1:]
config = load_recipe_config(config_path)
config = apply_recipe_overrides(
    config,
    [
        f"run.output_dir={output_dir}",
        f"run.model_path={model_path}",
        f"llm.execution.model={execution_model}",
        f"task.frontiercs.base_dir={frontiercs_dir}",
        f"task.frontiercs.judge_url={judge_url}",
    ],
)
prepared = prepare_run(config)
overrides = build_verl_overrides(config, prepared, [])
config_dir = str(Path(config["run"].get("verl_config_dir", _default_verl_config_dir())).resolve())
with initialize_config_dir(config_dir=config_dir, version_base=None):
    verl_config = compose(config_name="ppo_trainer", overrides=overrides)
OmegaConf.resolve(verl_config)
print(
    json.dumps(
        {
            "compose": "ok",
            "output_dir": str(prepared["output_dir"]),
            "max_response_length": int(config["run"]["max_response_length"]),
            "groups_per_batch": int(config["ttt"]["groups_per_batch"]),
            "group_size": int(config["ttt"]["group_size"]),
            "execution_model": config["llm"]["execution"]["model"],
        },
        indent=2,
    )
)
PY

"${PYTHON_RUN[@]}" -m guidance_ttt.main_erdos \
  --config "$CONFIG" \
  --bootstrap-only \
  "run.output_dir=$OUTPUT_DIR" \
  "run.model_path=$MODEL_PATH" \
  "llm.execution.model=$EXECUTION_MODEL" \
  "task.frontiercs.base_dir=$FRONTIERCS_DIR" \
  "task.frontiercs.judge_url=$JUDGE_URL" 2>&1 | tee "$BOOTSTRAP_LOG"

"${PYTHON_RUN[@]}" - "$OUTPUT_DIR/library.json" <<'PY'
import json
import sys
from pathlib import Path

library_path = Path(sys.argv[1])
data = json.loads(library_path.read_text())
entries = [entry for entry in (data.get("entries") or {}).values() if isinstance(entry, dict)]
bootstrap_entries = [entry for entry in entries if (entry.get("metadata") or {}).get("bootstrap")]
if len(bootstrap_entries) != 1:
    raise SystemExit(f"expected exactly one bootstrap entry, found {len(bootstrap_entries)}")
entry = bootstrap_entries[0]
metadata = entry.get("metadata") or {}
checks = {
    "status": entry.get("verifier_status"),
    "has_summary": bool(str(entry.get("summary") or "").strip()),
    "has_execution_text": bool(str(metadata.get("execution_text") or "").strip()),
    "fallback": bool(metadata.get("execution_fallback_used")),
    "execution_model": metadata.get("execution_model"),
}
print(json.dumps(checks, indent=2, sort_keys=True))
if checks["status"] != "valid":
    raise SystemExit(f"bootstrap entry is not valid: {checks}")
if not checks["has_summary"] or not checks["has_execution_text"]:
    raise SystemExit(f"bootstrap entry is missing required text: {checks}")
if checks["fallback"]:
    raise SystemExit(f"bootstrap entry used execution fallback: {checks}")
PY

echo "CPU smoke passed: config compose, endpoint bootstrap, verifier, and library writeback are OK."
