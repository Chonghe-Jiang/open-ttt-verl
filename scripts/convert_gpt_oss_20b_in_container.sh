#!/bin/bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"

export WORKSPACE=${WORKSPACE:-/root/workspace}
export ERDOS_DIR=${ERDOS_DIR:-${REPO_DIR}}

exec "${SCRIPT_DIR}/prepare_gpt_oss_20b_slime.sh"
