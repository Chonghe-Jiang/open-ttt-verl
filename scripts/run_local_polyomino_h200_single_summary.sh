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
APPTAINER_IMAGE="${APPTAINER_IMAGE:-}"
CONFIG="${CONFIG:-guidance_ttt/config/polyomino_h200_4gpu_single_summary.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/guidance_ttt/polyomino_h200_4gpu_one_step_30k_summary}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
EXECUTION_MODEL="${EXECUTION_MODEL:-openai/gpt-oss-20b}"
FRONTIERCS_DIR="${FRONTIERCS_DIR:-/home/qua/code/reference/Frontier-CS}"
JUDGE_URL="${JUDGE_URL:-http://127.0.0.1:8081}"
START_FRONTIER_JUDGE="${START_FRONTIER_JUDGE:-1}"
RESET_OUTPUT_DIR="${RESET_OUTPUT_DIR:-1}"
BOOTSTRAP_LOG="${BOOTSTRAP_LOG:-$OUTPUT_DIR/bootstrap.log}"
RUN_LOG="${RUN_LOG:-$OUTPUT_DIR/train.log}"

export NCCL_NET="${NCCL_NET:-Socket}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export TMPDIR="${TMPDIR:-$PWD/.tmp}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PWD/.triton_cache}"
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO="${RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO:-0}"
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

mkdir -p "$HF_HOME/datasets" "$HF_HOME/hub" "$TMPDIR" "$TRITON_CACHE_DIR" "$(dirname "$OUTPUT_DIR")"

PYTHON_RUN=("$PYTHON_BIN")
if [[ -n "$APPTAINER_IMAGE" ]]; then
  export CC="${CONTAINER_CC:-/usr/bin/gcc}"
  export CXX="${CONTAINER_CXX:-/usr/bin/g++}"
  export CUDAHOSTCXX="${CONTAINER_CUDAHOSTCXX:-$CXX}"
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
  echo "CONTAINER_RUNTIME_BIN=$CONTAINER_RUNTIME_BIN"
  PYTHON_RUN=("$CONTAINER_RUNTIME_BIN" exec --nv --bind /home/qua/code:/home/qua/code "$APPTAINER_IMAGE" "$PYTHON_BIN")
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
    echo "node is required for START_FRONTIER_JUDGE=1" >&2
    exit 1
  fi

  local app_dir="${JUDGE_APP_DIR:-$TMPDIR/frontiercs-local-judge}"
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
  JUDGE_WORKERS="${JUDGE_WORKERS:-8}" \
  TESTLIB_INSIDE="$app_dir/include" \
  SAVE_OUTPUTS="${SAVE_OUTPUTS:-false}" \
  node "$app_dir/server.js" >"$OUTPUT_DIR.frontier_judge.log" 2>&1 &
  judge_pid="$!"
  wait_for_url "$JUDGE_URL/problems" "${JUDGE_START_TIMEOUT_S:-120}"
}

write_run_summary_if_possible() {
  "${PYTHON_RUN[@]}" -m guidance_ttt.run_summary "$OUTPUT_DIR" --config "$CONFIG" || true
}

prune_to_single_summary() {
  "${PYTHON_RUN[@]}" - "$OUTPUT_DIR/library.json" <<'PY'
import json
import sys
from pathlib import Path

library_path = Path(sys.argv[1])
data = json.loads(library_path.read_text())
entries = data.get("entries") or {}
if not isinstance(entries, dict) or not entries:
    raise SystemExit("library has no entries to prune")
candidate_items = [
    (entry_id, entry)
    for entry_id, entry in entries.items()
    if isinstance(entry, dict) and int(entry.get("timestep") or -1) == 1
]
if not candidate_items:
    raise SystemExit("library has no timestep=1 guided entries to prune")

def rank(entry):
    metadata = entry.get("metadata") or {}
    raw_score = entry.get("verifier_raw_score")
    reward = entry.get("verifier_reward")
    numeric_score = raw_score if raw_score is not None else reward
    try:
        score = float(numeric_score)
    except (TypeError, ValueError):
        score = float("-inf")
    return (
        1 if str(entry.get("verifier_status")) == "valid" else 0,
        1 if bool(metadata.get("guidance_format_ok")) and str(entry.get("guidance") or "").strip() else 0,
        1 if str(entry.get("summary") or "").strip() else 0,
        1 if str(metadata.get("execution_text") or "").strip() else 0,
        score,
        str(entry.get("id") or ""),
    )

chosen_id, chosen_entry = max(candidate_items, key=lambda item: rank(item[1]))
nodes = data.get("nodes") or {}
chosen_node_id = None
for node_id, node in nodes.items():
    if isinstance(node, dict) and node.get("entry_id") == chosen_id:
        chosen_node_id = node_id
        break

keep_node_ids = set()
current = chosen_node_id
while current and current in nodes:
    keep_node_ids.add(current)
    node = nodes[current] if isinstance(nodes[current], dict) else {}
    parent_id = node.get("parent_id")
    current = str(parent_id) if parent_id else None
for node_id, node in nodes.items():
    if isinstance(node, dict) and node.get("entry_id") is None:
        keep_node_ids.add(node_id)

pruned_nodes = {
    node_id: node
    for node_id, node in nodes.items()
    if node_id in keep_node_ids and isinstance(node, dict)
}
for node in pruned_nodes.values():
    children = node.get("children") or []
    node["children"] = [child for child in children if child in pruned_nodes]
    if node.get("entry_id") not in {None, chosen_id}:
        node["entry_id"] = None
if chosen_node_id:
    data["best_node_id"] = chosen_node_id

groups = data.get("groups") or {}
if isinstance(groups, dict):
    for group in groups.values():
        if not isinstance(group, dict):
            continue
        group["children"] = [child for child in (group.get("children") or []) if child == chosen_node_id]
        group["submitted"] = 1 if group["children"] else 0
        group["finalized"] = bool(group["children"])

if isinstance(data.get("config"), dict):
    data["config"]["rollout_n"] = 1
data["rollout_n"] = 1
data["entries"] = {chosen_id: chosen_entry}
data["nodes"] = pruned_nodes
data.setdefault("metadata", {})
if isinstance(data["metadata"], dict):
    data["metadata"]["single_summary_pruned_from_entries"] = len(entries)
    data["metadata"]["single_summary_chosen_entry_id"] = chosen_id
library_path.write_text(json.dumps(data, indent=2, sort_keys=True))
print(json.dumps({"entry_count": 1, "chosen_entry_id": chosen_id, "chosen_node_id": chosen_node_id, "pruned_from_entry_count": len(entries)}, indent=2))
PY
}

validate_single_summary() {
  "${PYTHON_RUN[@]}" - "$OUTPUT_DIR/library.json" <<'PY'
import json
import sys
from pathlib import Path

library_path = Path(sys.argv[1])
data = json.loads(library_path.read_text())
entries = [entry for entry in (data.get("entries") or {}).values() if isinstance(entry, dict)]
if len(entries) != 1:
    raise SystemExit(f"expected exactly one library entry after pruning, found {len(entries)}")
entry = entries[0]
metadata = entry.get("metadata") or {}
checks = {
    "status": entry.get("verifier_status"),
    "has_summary": bool(str(entry.get("summary") or "").strip()),
    "has_guidance": bool(str(entry.get("guidance") or "").strip()),
    "has_execution_text": bool(str(metadata.get("execution_text") or "").strip()),
    "fallback": bool(metadata.get("execution_fallback_used")),
}
print(json.dumps(checks, indent=2, sort_keys=True))
if int(entry.get("timestep") or -1) != 1:
    raise SystemExit(f"single summary entry is not a timestep=1 guided entry: {entry.get('timestep')}")
if checks["status"] != "valid":
    raise SystemExit(f"single summary entry is not valid: {checks}")
if not checks["has_summary"] or not checks["has_guidance"] or not checks["has_execution_text"]:
    raise SystemExit(f"single summary entry is missing required text: {checks}")
if checks["fallback"]:
    raise SystemExit(f"single summary entry used execution fallback: {checks}")
PY
}

write_step_io_log() {
  "${PYTHON_RUN[@]}" - "$OUTPUT_DIR/library.json" "$OUTPUT_DIR/step1_guidance_execution_log.md" "$OUTPUT_DIR/step1_guidance_execution_log.json" <<'PY'
import json
import sys
from pathlib import Path

library_path = Path(sys.argv[1])
markdown_path = Path(sys.argv[2])
json_path = Path(sys.argv[3])
data = json.loads(library_path.read_text())
entries = [
    entry
    for entry in (data.get("entries") or {}).values()
    if isinstance(entry, dict)
]
entries.sort(key=lambda entry: (int(entry.get("timestep") or -1), str(entry.get("id") or "")))
exported = []
for entry in entries:
    metadata = entry.get("metadata") or {}
    exported.append(
        {
            "entry_id": entry.get("id"),
            "timestep": entry.get("timestep"),
            "verifier_status": entry.get("verifier_status"),
            "verifier_message": entry.get("verifier_message"),
            "verifier_reward": entry.get("verifier_reward"),
            "verifier_raw_score": entry.get("verifier_raw_score"),
            "guidance": entry.get("guidance") or "",
            "execution_output": metadata.get("execution_text") or "",
            "canonical_summary": entry.get("summary") or "",
            "execution_model": metadata.get("execution_model"),
            "execution_fallback_used": metadata.get("execution_fallback_used"),
            "execution_fallback_reason": metadata.get("execution_fallback_reason"),
        }
    )
json_path.write_text(json.dumps({"library": str(library_path), "entries": exported}, indent=2, sort_keys=True))
lines = [
    "# One-Step Guidance And Execution Log",
    "",
    f"- Library: `{library_path}`",
    f"- Entries: {len(exported)}",
    "",
]
for item in exported:
    lines.extend(
        [
            f"## Entry `{item['entry_id']}`",
            "",
            f"- Timestep: `{item['timestep']}`",
            f"- Status: `{item['verifier_status']}`",
            f"- Message: {item['verifier_message']}",
            f"- Reward: `{item['verifier_reward']}`",
            f"- Raw score: `{item['verifier_raw_score']}`",
            f"- Execution model: `{item['execution_model']}`",
            f"- Fallback used: `{item['execution_fallback_used']}`",
            "",
            "### Guidance",
            "",
            "```text",
            str(item["guidance"]),
            "```",
            "",
            "### Execution Output",
            "",
            "```text",
            str(item["execution_output"]),
            "```",
            "",
            "### Canonical Summary",
            "",
            "```text",
            str(item["canonical_summary"]),
            "```",
            "",
        ]
    )
markdown_path.write_text("\n".join(lines))
print(json.dumps({"markdown": str(markdown_path), "json": str(json_path), "entry_count": len(exported)}, indent=2))
PY
}

if [[ "$RESET_OUTPUT_DIR" == "1" ]]; then
  rm -rf "$OUTPUT_DIR"
fi
mkdir -p "$OUTPUT_DIR"

if [[ "$START_FRONTIER_JUDGE" == "1" ]]; then
  start_frontier_judge
else
  wait_for_url "$JUDGE_URL/problems" "${JUDGE_START_TIMEOUT_S:-10}"
fi

"${PYTHON_RUN[@]}" -m guidance_ttt.main_erdos \
  --config "$CONFIG" \
  --bootstrap-only \
  "run.output_dir=$OUTPUT_DIR" \
  "run.model_path=$MODEL_PATH" \
  "llm.execution.model=$EXECUTION_MODEL" \
  "task.frontiercs.base_dir=$FRONTIERCS_DIR" \
  "task.frontiercs.judge_url=$JUDGE_URL" 2>&1 | tee "$BOOTSTRAP_LOG"

"${PYTHON_RUN[@]}" -m guidance_ttt.main_erdos \
  --config "$CONFIG" \
  "run.output_dir=$OUTPUT_DIR" \
  "run.model_path=$MODEL_PATH" \
  "llm.execution.model=$EXECUTION_MODEL" \
  "task.frontiercs.base_dir=$FRONTIERCS_DIR" \
  "task.frontiercs.judge_url=$JUDGE_URL" \
  "$@" 2>&1 | tee "$RUN_LOG"

write_run_summary_if_possible
prune_to_single_summary
validate_single_summary
write_run_summary_if_possible
write_step_io_log
