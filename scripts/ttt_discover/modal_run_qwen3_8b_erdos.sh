#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${MODAL_ENV_FILE:-${ROOT}/.env}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ACTION="${1:-run}"

if [[ $# -gt 0 ]]; then
  shift
fi

if [[ -f "${ENV_FILE}" ]]; then
  eval "$(
    "${PYTHON_BIN}" - "${ENV_FILE}" <<'PY'
import shlex
import sys
from pathlib import Path

env_file = Path(sys.argv[1])
keys = {"MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "MODAL_SECRET_KEY"}
values = {}

for raw_line in env_file.read_text().splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    key, value = line.split("=", 1)
    key = key.strip()
    if key.startswith("export "):
        key = key.removeprefix("export ").strip()
    if key not in keys:
        continue
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    elif " #" in value:
        value = value.split(" #", 1)[0].rstrip()
    values[key] = value

for key, value in values.items():
    print(f"export {key}={shlex.quote(value)}")

if "MODAL_TOKEN_SECRET" not in values and values.get("MODAL_SECRET_KEY"):
    print(f"export MODAL_TOKEN_SECRET={shlex.quote(values['MODAL_SECRET_KEY'])}")
PY
  )"
fi

if [[ -z "${MODAL_TOKEN_ID:-}" || -z "${MODAL_TOKEN_SECRET:-}" ]]; then
  echo "Missing MODAL_TOKEN_ID or MODAL_TOKEN_SECRET. Put MODAL_TOKEN_ID and MODAL_SECRET_KEY in ${ENV_FILE}." >&2
  exit 2
fi

exec modal run --detach --timestamps --name polynomino-qwen3-8b-h200-g4n16-train \
  "${ROOT}/scripts/ttt_discover/modal_qwen3_8b_erdos.py" \
  --action "${ACTION}" \
  "$@"
