#!/usr/bin/env bash
# Clone the two third-party repos that this work depends on, pinned to the
# exact commits we used. Run this once after cloning Xuan-1998/ttt.
#
# Why we do not vendor them:
# - frontier-cs is 1.4 GB (includes evaluator Docker images and datasets) and
#   is the property of the FrontierCS team.
# - ttt-discover (paper code) is the property of test-time-training / Thinking
#   Machines and depends on the Tinker SDK, which we do not use.
#
# Our scripts/ttt_iterative_4tasks.py is an independent re-implementation of
# the TTT-Discover algorithm on top of HF transformers + PEFT + vLLM, with no
# dependency on Tinker.

set -euo pipefail
SRC_DIR="${SRC_DIR:-./src}"
mkdir -p "$SRC_DIR"

echo "[setup] cloning Frontier-CS @ 307d5209 (evaluator)"
if [ ! -d "$SRC_DIR/frontier-cs" ]; then
  git clone https://github.com/FrontierCS/Frontier-CS.git "$SRC_DIR/frontier-cs"
fi
git -C "$SRC_DIR/frontier-cs" fetch origin
git -C "$SRC_DIR/frontier-cs" checkout 307d5209

echo "[setup] cloning TTT-Discover paper code @ bf20511 (reference only)"
if [ ! -d "$SRC_DIR/ttt-discover" ]; then
  git clone https://github.com/test-time-training/discover.git "$SRC_DIR/ttt-discover"
fi
git -C "$SRC_DIR/ttt-discover" fetch origin
git -C "$SRC_DIR/ttt-discover" checkout bf20511

echo "[setup] installing frontier-cs CLI (editable, --no-deps to avoid resolver issues)"
pip install --user --no-deps -e "$SRC_DIR/frontier-cs" || {
  echo "[setup] WARN: editable install failed; falling back to PYTHONPATH"
  echo "[setup] export PYTHONPATH=\"$SRC_DIR/frontier-cs/src:\$PYTHONPATH\""
}

echo "[setup] done. frontier eval CLI should be on PATH; data lives at $SRC_DIR/frontier-cs/research/problems/"
echo "[setup] also need: pip install --user vllm transformers peft accelerate datasets pyyaml"
