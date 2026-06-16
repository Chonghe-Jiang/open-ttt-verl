#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env_g75.sh"
python - <<'PY'
import importlib.util
mods = [
    "guidance_ttt",
    "verl",
    "verl.trainer.main_ppo",
    "vllm",
    "ray",
    "hydra",
    "omegaconf",
    "torch",
    "numpy",
    "yaml",
    "pandas",
    "pyarrow",
]
missing = []
for mod in mods:
    ok = importlib.util.find_spec(mod) is not None
    print(f"{mod}: {'OK' if ok else 'MISSING'}")
    if not ok:
        missing.append(mod)
if missing:
    raise SystemExit(f"missing modules: {missing}")
PY
