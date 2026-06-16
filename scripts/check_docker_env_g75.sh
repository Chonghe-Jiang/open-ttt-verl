#!/usr/bin/env bash
set -euo pipefail
"$(dirname "$0")/docker_env_g75.sh" -lc 'python3 - <<"PY"
import importlib
mods = [
    "guidance_ttt",
    "guidance_ttt.agent_loop",
    "verl",
    "verl.trainer.main_ppo",
    "vllm",
    "ray",
    "hydra",
    "omegaconf",
    "torch",
    "datasets",
    "pandas",
    "pyarrow",
]
for mod in mods:
    importlib.import_module(mod)
    print(f"{mod}: OK")
PY'
