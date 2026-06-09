#!/bin/bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export DOCKER_GPUS=${DOCKER_GPUS:-all}
export SHM_SIZE=${SHM_SIZE:-128g}
export CONTAINER_NAME=${CONTAINER_NAME:-open-ttt-qwen3-8b-8b200}

exec "${SCRIPT_DIR}/run_open_ttt_slime.sh" scripts/docker_qwen3_8b_8b200_rl.sh
