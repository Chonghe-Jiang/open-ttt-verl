#!/usr/bin/env bash
set -euo pipefail

# Build the portable TTT-Discover runtime image.
# Override IMAGE_TAG or BASE_IMAGE from the shell.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

IMAGE_TAG="${IMAGE_TAG:-open-ttt-verl:ttt-vllm}"
BASE_IMAGE="${BASE_IMAGE:-verlai/verl:vllm017.latest}"
DOCKERFILE="${DOCKERFILE:-docker/Dockerfile.ttt-vllm}"
CONTEXT="${CONTEXT:-${REPO_ROOT}}"

cd "${REPO_ROOT}"

docker build \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  -f "${DOCKERFILE}" \
  -t "${IMAGE_TAG}" \
  "$@" \
  "${CONTEXT}"
