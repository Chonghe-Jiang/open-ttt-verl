#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENEVOLVE_ROOT="$ROOT/openevolve"

if [[ ! -f "$OPENEVOLVE_ROOT/openevolve-run.py" ]]; then
  echo "Missing local OpenEvolve clone at $OPENEVOLVE_ROOT" >&2
  exit 1
fi

ITERATIONS=20
CONFIG="configs/erdos_openai_compatible.example.yaml"
PASS_ITERATIONS=1
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --iterations|-i)
      ITERATIONS="${2:-20}"
      if [[ "$ITERATIONS" == "0" ]]; then
        CONFIG="configs/erdos_no_api.yaml"
        PASS_ITERATIONS=0
      fi
      shift 2
      ;;
    --smoke)
      CONFIG="configs/erdos_no_api.yaml"
      PASS_ITERATIONS=0
      shift
      ;;
    --config|-c)
      CONFIG="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "$CONFIG" == "configs/erdos_openai_compatible.example.yaml" ]]; then
  if [[ -z "${OPENAI_API_KEY:-}" && -n "${API_KEY:-}" ]]; then
    export OPENAI_API_KEY="$API_KEY"
  fi

  if [[ -z "${OPENAI_API_BASE:-}" && -n "${ENDPOINT:-}" ]]; then
    export OPENAI_API_BASE="$ENDPOINT"
  fi
fi

cd "$ROOT"
CMD=(python "$OPENEVOLVE_ROOT/openevolve-run.py" initial_program.py evaluator.py --config "$CONFIG")
if [[ "$PASS_ITERATIONS" == "1" ]]; then
  CMD+=(--iterations "$ITERATIONS")
fi
CMD+=("${EXTRA_ARGS[@]}")

"${CMD[@]}"
