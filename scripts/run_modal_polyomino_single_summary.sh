#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
  eval "$(
    python - <<'PY'
from pathlib import Path
import os
import shlex
import tomllib

keys = {"WORKSPACE", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "MODAL_SECRET_KEY"}
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

if "MODAL_TOKEN_SECRET" not in values and values.get("MODAL_SECRET_KEY"):
    values["MODAL_TOKEN_SECRET"] = values["MODAL_SECRET_KEY"]
if "MODAL_TOKEN_SECRET" not in os.environ and os.environ.get("MODAL_SECRET_KEY"):
    values["MODAL_TOKEN_SECRET"] = os.environ["MODAL_SECRET_KEY"]

token_id = os.environ.get("MODAL_TOKEN_ID") or values.get("MODAL_TOKEN_ID")
token_secret = os.environ.get("MODAL_TOKEN_SECRET") or values.get("MODAL_TOKEN_SECRET")
if token_id and not token_secret:
    config_path = Path.home() / ".modal.toml"
    if config_path.exists():
        data = tomllib.loads(config_path.read_text())
        for section in data.values():
            if isinstance(section, dict) and section.get("token_id") == token_id and section.get("token_secret"):
                values["MODAL_TOKEN_SECRET"] = str(section["token_secret"])
                break

for key in ("WORKSPACE", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"):
    if values.get(key):
        print(f"export {key}={shlex.quote(values[key])}")
PY
  )"
fi

if [[ -n "${MODAL_TOKEN_ID:-}" && -z "${MODAL_TOKEN_SECRET:-}" ]]; then
  {
    echo "ERROR: .env defines MODAL_TOKEN_ID but not MODAL_TOKEN_SECRET."
    echo "No matching token secret for that id was found in ~/.modal.toml."
    echo "Modal API token auth requires both values."
  } >&2
  exit 1
fi

if [[ -z "${MODAL_TOKEN_ID:-}" && -n "${MODAL_TOKEN_SECRET:-}" ]]; then
  {
    echo "ERROR: .env defines MODAL_TOKEN_SECRET but not MODAL_TOKEN_ID."
    echo "Modal API token auth requires both values."
  } >&2
  exit 1
fi

if [[ -z "${MODAL_PROFILE:-}" && -f .env ]]; then
  workspace="$(python - <<'PY'
from pathlib import Path

workspace = ""
for raw_line in Path(".env").read_text().splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    key, value = line.split("=", 1)
    key = key.strip()
    if key.startswith("export "):
        key = key.removeprefix("export ").strip()
    if key == "WORKSPACE":
        workspace = value.strip().strip("'\"")
        break

profiles = set()
config_path = Path.home() / ".modal.toml"
if config_path.exists():
    import tomllib

    profiles = set(tomllib.loads(config_path.read_text()).keys())

if workspace in profiles:
    print(workspace)
PY
)"
  if [[ -n "${workspace:-}" ]]; then
    export MODAL_PROFILE="$workspace"
  fi
fi

expected_workspace="$(python - <<'PY'
from pathlib import Path

workspace = ""
if Path(".env").exists():
    for raw_line in Path(".env").read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key.removeprefix("export ").strip()
        if key == "WORKSPACE":
            workspace = value.strip().strip("'\"")
            break
print(workspace)
PY
)"

if [[ -z "${expected_workspace:-}" ]]; then
  echo "ERROR: .env must contain WORKSPACE=<modal-workspace-name-or-id>." >&2
  exit 1
fi

token_info="$(modal token info)"
actual_workspace_line="$(printf '%s\n' "$token_info" | awk -F': ' '$1=="Workspace"{print $2}')"
actual_workspace_name="$(printf '%s' "$actual_workspace_line" | sed -E 's/[[:space:]]*\([^)]*\)[[:space:]]*$//')"
actual_workspace_id="$(printf '%s' "$actual_workspace_line" | sed -nE 's/^.*\(([^)]*)\)[[:space:]]*$/\1/p')"

if [[ "$expected_workspace" != "$actual_workspace_name" && "$expected_workspace" != "$actual_workspace_id" ]]; then
  {
    echo "ERROR: Modal token workspace does not match .env WORKSPACE."
    echo "  .env WORKSPACE: $expected_workspace"
    echo "  token workspace: $actual_workspace_name ($actual_workspace_id)"
    echo
    echo "Create or set a Modal token that is actually connected to the .env workspace, then retry:"
    echo "  modal token new --profile \"$expected_workspace\" --activate"
    echo "  modal token info"
    echo
    echo "The token-info Workspace line must show the .env workspace name or id."
  } >&2
  exit 1
fi

modal run scripts/modal_polyomino_h200_smoke.py --action bootstrap_single_summary
modal run --detach scripts/modal_polyomino_h200_smoke.py --action train_single_summary "$@"
