#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python scripts/evaluate_initial.py
pytest -q tests
python -m compileall -q erdos_evolve evaluator.py initial_program.py scripts/evaluate_initial.py scripts/check_vllm_server.py
