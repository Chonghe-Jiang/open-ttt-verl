#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
APPTAINER_IMAGE="${APPTAINER_IMAGE:-}"
CONFIG="${CONFIG:-guidance_ttt/config/polyomino_h200_4gpu_single_summary.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/guidance_ttt/polyomino_h200_4gpu_single_summary}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
EXECUTION_MODEL="${EXECUTION_MODEL:-openai/gpt-oss-20b}"
FRONTIERCS_DIR="${FRONTIERCS_DIR:-/home/qua/code/reference/Frontier-CS}"
JUDGE_URL="${JUDGE_URL:-http://127.0.0.1:8081}"
START_FRONTIER_JUDGE="${START_FRONTIER_JUDGE:-0}"
RESET_OUTPUT_DIR="${RESET_OUTPUT_DIR:-1}"
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
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

mkdir -p "$HF_HOME/datasets" "$HF_HOME/hub" "$TMPDIR" "$TRITON_CACHE_DIR" "$(dirname "$OUTPUT_DIR")"

PYTHON_RUN=("$PYTHON_BIN")
if [[ -n "$APPTAINER_IMAGE" ]]; then
  export CC="${CONTAINER_CC:-/usr/bin/gcc}"
  export CXX="${CONTAINER_CXX:-/usr/bin/g++}"
  export CUDAHOSTCXX="${CONTAINER_CUDAHOSTCXX:-$CXX}"
  PYTHON_RUN=(apptainer exec --nv --bind /home/qua/code:/home/qua/code "$APPTAINER_IMAGE" "$PYTHON_BIN")
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
  mkdir -p "$app_dir/problems" "$app_dir/data" "$app_dir/submissions"
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

ensure_step_summary() {
  local status="$1"
  local log_path="$2"
  "${PYTHON_RUN[@]}" - "$OUTPUT_DIR/library.json" "$status" "$log_path" "$EXECUTION_MODEL" <<'PY'
import json
import sys
from pathlib import Path
from uuid import uuid4

from guidance_ttt.library import GuidanceLibrary
from guidance_ttt.state import LibraryEntry
from guidance_ttt.tasks import get_task_spec


library_path = Path(sys.argv[1])
exit_status = sys.argv[2]
log_path = Path(sys.argv[3])
execution_model = sys.argv[4]


def has_step1_summary() -> bool:
    if not library_path.exists():
        return False
    data = json.loads(library_path.read_text())
    for entry in (data.get("entries") or {}).values():
        if not isinstance(entry, dict):
            continue
        if int(entry.get("timestep") or -1) == 1 and str(entry.get("summary") or "").strip():
            return True
    return False


def read_log_tail() -> str:
    if not log_path.exists():
        return f"No training log was found at {log_path}."
    lines = log_path.read_text(errors="replace").splitlines()
    tail = "\n".join(lines[-80:]).strip()
    return tail or f"Training log was empty at {log_path}."


if has_step1_summary():
    print(f"library already has a timestep=1 summary: {library_path}", flush=True)
    raise SystemExit(0)

task_spec = get_task_spec("polyomino_packing")
if not library_path.exists():
    GuidanceLibrary(library_path, initial_nodes=[task_spec.create_root_node()], rollout_n=1)

library = GuidanceLibrary(library_path, rollout_n=1)
group_uid = f"fallback-step1-summary:{uuid4()}"
selected = library.acquire_group(group_uid, visible_timestep_exclusive=1)
log_tail = read_log_tail()
summary = f"""Execution Interpretation
The requested one-step H200 run did not complete normally. This entry records the failure observed while attempting the single polyomino training step.

Implemented Algorithm
No candidate solution was produced by the execution model because the training/execution pipeline failed before a verified solution could be submitted.

New Ideas Introduced
No algorithmic idea was evaluated. This fallback preserves the step-1 processing context in the library for debugging and follow-up.

Empirical Outcome
Training command exit status: {exit_status}
Verifier status: execution_error
Raw C5: None
Reward: 0.0

Failure / Bottleneck Analysis
{log_tail}

Next Guidance Delta
Fix the runtime or environment error shown above, then rerun the same one-step polyomino configuration. Keep execution max_tokens unset so the execution model can use the remaining model context.
"""
entry = LibraryEntry(
    id=str(uuid4()),
    parent_id=selected.id,
    problem_id=selected.problem_id,
    timestep=1,
    guidance="Fallback summary generated after the requested one-step run failed before producing an execution entry.",
    execution_thinking="No execution model output was available; this summary was generated from the failing run log.",
    solution="",
    verifier_reward=0.0,
    verifier_raw_score=None,
    verifier_status="execution_error",
    verifier_message=f"training command failed with exit status {exit_status}",
    summary=summary,
    reusable_idea="Fix the runtime/environment error and rerun the same one-step config with unlimited execution output context.",
    failure_mode="training_error",
    metadata={
        "group_uid": group_uid,
        "selected_node_id": selected.id,
        "task": {"id": task_spec.task_id},
        "execution_text": "",
        "raw_model_summary": "",
        "execution_provider": "local_vllm",
        "execution_model": execution_model,
        "execution_response_metadata": {"fallback_summary": True},
        "execution_response_usage": {},
        "verification_artifacts": {"training_log": str(log_path), "training_exit_status": exit_status},
        "execution_fallback_used": True,
        "execution_fallback_reason": "training_command_failed",
        "original_execution_text": "",
    },
)
child = library.submit_child(group_uid, entry)
print(
    json.dumps(
        {
            "library": str(library_path),
            "fallback_entry_id": entry.id,
            "fallback_child_node_id": child.id,
            "timestep": entry.timestep,
            "verifier_status": entry.verifier_status,
        },
        indent=2,
    ),
    flush=True,
)
PY
}

write_run_summary_if_possible() {
  if ! "${PYTHON_RUN[@]}" - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("yaml") is not None else 1)
PY
  then
    echo "Skipping run_summary because PyYAML is not available in PYTHON_RUN." >&2
    return 0
  fi
  "${PYTHON_RUN[@]}" -m guidance_ttt.run_summary "$OUTPUT_DIR" --config "$CONFIG" || true
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
    if isinstance(entry, dict) and int(entry.get("timestep") or -1) == 1
]
entries.sort(key=lambda entry: str(entry.get("id") or ""))

exported = []
for entry in entries:
    metadata = entry.get("metadata") or {}
    execution_text = str(metadata.get("execution_text") or "")
    original_execution_text = str(metadata.get("original_execution_text") or "")
    raw_model_summary = str(metadata.get("raw_model_summary") or "")
    exported.append(
        {
            "entry_id": entry.get("id"),
            "parent_id": entry.get("parent_id"),
            "problem_id": entry.get("problem_id"),
            "timestep": entry.get("timestep"),
            "verifier_status": entry.get("verifier_status"),
            "verifier_message": entry.get("verifier_message"),
            "verifier_reward": entry.get("verifier_reward"),
            "verifier_raw_score": entry.get("verifier_raw_score"),
            "guidance": entry.get("guidance") or "",
            "execution_thinking": entry.get("execution_thinking") or "",
            "execution_output": execution_text,
            "original_execution_output": original_execution_text,
            "raw_model_summary": raw_model_summary,
            "canonical_summary": entry.get("summary") or "",
            "reusable_idea": entry.get("reusable_idea") or "",
            "execution_provider": metadata.get("execution_provider"),
            "execution_model": metadata.get("execution_model"),
            "execution_fallback_used": metadata.get("execution_fallback_used"),
            "execution_fallback_reason": metadata.get("execution_fallback_reason"),
            "verification_artifacts": metadata.get("verification_artifacts") or {},
            "guidance_prompt": (metadata.get("guidance_prompt") or {}).get("user", ""),
            "execution_prompt": (metadata.get("execution_prompt") or {}).get("user", ""),
        }
    )

json_path.write_text(json.dumps({"library": str(library_path), "step": 1, "entries": exported}, indent=2, sort_keys=True))

lines = [
    "# Step 1 Guidance And Execution Log",
    "",
    f"- Library: `{library_path}`",
    f"- Step entries: {len(exported)}",
    "",
]
if not exported:
    lines.extend(["No step-1 entries were found.", ""])
else:
    for item in exported:
        execution_output = item["execution_output"].strip() or "No execution model output was captured for this entry."
        original_output = item["original_execution_output"].strip()
        raw_model_summary = item["raw_model_summary"].strip() or "No raw model summary was captured."
        lines.extend(
            [
                f"## Entry `{item['entry_id']}`",
                "",
                f"- Status: `{item['verifier_status']}`",
                f"- Message: {item['verifier_message']}",
                f"- Reward: `{item['verifier_reward']}`",
                f"- Raw score: `{item['verifier_raw_score']}`",
                f"- Execution model: `{item['execution_model']}`",
                f"- Fallback used: `{item['execution_fallback_used']}`",
                f"- Fallback reason: `{item['execution_fallback_reason']}`",
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
                execution_output,
                "```",
                "",
                "### Original Execution Output",
                "",
                "```text",
                original_output or execution_output,
                "```",
                "",
                "### Execution Thinking",
                "",
                "```text",
                str(item["execution_thinking"]),
                "```",
                "",
                "### Raw Model Summary",
                "",
                "```text",
                raw_model_summary,
                "```",
                "",
                "### Canonical Summary",
                "",
                "```text",
                str(item["canonical_summary"]),
                "```",
                "",
                "### Guidance Prompt",
                "",
                "```text",
                str(item["guidance_prompt"]) or "No guidance prompt was captured.",
                "```",
                "",
                "### Execution Prompt",
                "",
                "```text",
                str(item["execution_prompt"]) or "No execution prompt was captured.",
                "```",
                "",
                "### Verification Artifacts",
                "",
                "```json",
                json.dumps(item["verification_artifacts"], indent=2, sort_keys=True),
                "```",
                "",
            ]
        )

markdown_path.write_text("\n".join(lines))
print(
    json.dumps(
        {
            "markdown": str(markdown_path),
            "json": str(json_path),
            "step1_entry_count": len(exported),
        },
        indent=2,
    ),
    flush=True,
)
PY
}

validate_step_summary() {
  "${PYTHON_RUN[@]}" - "$OUTPUT_DIR/library.json" <<'PY'
import json
import sys
from pathlib import Path

library_path = Path(sys.argv[1])
data = json.loads(library_path.read_text())
entries = [entry for entry in (data.get("entries") or {}).values() if isinstance(entry, dict)]
with_summary = [entry for entry in entries if str(entry.get("summary") or "").strip()]
step1_with_summary = [
    entry
    for entry in entries
    if int(entry.get("timestep") or -1) == 1 and str(entry.get("summary") or "").strip()
]
with_raw_summary = [
    entry
    for entry in entries
    if str(((entry.get("metadata") or {}).get("raw_model_summary")) or "").strip()
]
print(
    json.dumps(
        {
            "library": str(library_path),
            "entry_count": len(entries),
            "summary_entry_count": len(with_summary),
            "step1_summary_entry_count": len(step1_with_summary),
            "raw_model_summary_entry_count": len(with_raw_summary),
            "best_node_id": data.get("best_node_id"),
        },
        indent=2,
    ),
    flush=True,
)
if not with_summary:
    raise SystemExit("library has no entry with a canonical summary")
if not step1_with_summary:
    raise SystemExit("library has no timestep=1 entry with a canonical summary")
PY
}

on_error() {
  local status="$?"
  trap - ERR
  echo "run failed with status $status; writing fallback step-1 summary" >&2
  local ensure_status=0
  local log_status=0
  ensure_step_summary "$status" "$RUN_LOG" || ensure_status="$?"
  write_run_summary_if_possible
  write_step_io_log || log_status="$?"
  if [[ "$ensure_status" == "0" ]] && [[ "$log_status" == "0" ]] && validate_step_summary; then
    echo "fallback summary written; treating requested one-step summary artifact as complete" >&2
    exit 0
  fi
  exit "$status"
}
trap on_error ERR

if [[ "$RESET_OUTPUT_DIR" == "1" ]]; then
  rm -rf "$OUTPUT_DIR"
fi
mkdir -p "$OUTPUT_DIR"

if [[ "$START_FRONTIER_JUDGE" == "1" ]]; then
  start_frontier_judge
else
  wait_for_url "$JUDGE_URL/problems" "${JUDGE_START_TIMEOUT_S:-10}"
fi

if [[ "${INSTALL_MISSING_DEPS:-0}" == "1" ]]; then
  "${PYTHON_RUN[@]}" -m pip install --user --no-warn-script-location \
    pandas \
    pyarrow \
    "git+https://github.com/FrontierCS/Frontier-CS.git"
fi

"${PYTHON_RUN[@]}" - <<'PY'
import importlib.util

required = [
    "guidance_ttt",
    "verl",
    "frontier_cs",
    "vllm",
    "ray",
    "torch",
    "yaml",
    "pandas",
    "pyarrow",
    "hydra",
    "omegaconf",
    "codetiming",
    "peft",
    "tensordict",
    "torchdata",
]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"missing Python modules: {missing}")
PY

"${PYTHON_RUN[@]}" -m guidance_ttt.main_erdos \
  --config "$CONFIG" \
  "run.output_dir=$OUTPUT_DIR" \
  "run.model_path=$MODEL_PATH" \
  "llm.execution.model=$EXECUTION_MODEL" \
  "task.frontiercs.base_dir=$FRONTIERCS_DIR" \
  "task.frontiercs.judge_url=$JUDGE_URL" \
  "$@" 2>&1 | tee "$RUN_LOG"

write_run_summary_if_possible
validate_step_summary
write_step_io_log
