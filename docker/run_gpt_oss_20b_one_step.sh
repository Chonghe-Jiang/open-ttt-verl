#!/bin/bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export DOCKER_GPUS=${DOCKER_GPUS:-all}
export SHM_SIZE=${SHM_SIZE:-128g}
export CONTAINER_NAME=${CONTAINER_NAME:-open-ttt-gpt-oss-20b-one-step}

exec "${SCRIPT_DIR}/run_open_ttt_slime.sh" ./run_gpt_oss_20b_erdos.sh
