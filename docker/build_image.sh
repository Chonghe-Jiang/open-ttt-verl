#!/bin/bash
set -Eeuo pipefail

REPO_DIR=${REPO_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)}
IMAGE=${IMAGE:-open-ttt-slime:latest}
BASE_IMAGE=${BASE_IMAGE:-slimerl/slime:latest}

cd "${REPO_DIR}"

docker build \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  -t "${IMAGE}" \
  -f docker/Dockerfile \
  .
