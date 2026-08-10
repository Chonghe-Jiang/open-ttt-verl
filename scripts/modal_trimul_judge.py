from __future__ import annotations

import os
import sys
from pathlib import Path

import modal
from fastapi import Request


APP_NAME = "guidance-ttt-trimul-judge"
SECRET_NAME = os.environ.get(
    "GUIDANCE_TTT_TRIMUL_SECRET",
    "guidance-ttt-trimul-secrets",
)
AUTH_TOKEN_ENV = "TRIMUL_JUDGE_TOKEN"
EVALUATION_GPU_CONFIG = "H100!"
EVALUATION_MAX_CONTAINERS = max(
    1,
    int(os.environ.get("TRIMUL_JUDGE_MAX_CONTAINERS", "16")),
)
REMOTE_PACKAGE_DIR = "/opt/guidance_ttt"
REMOTE_EVALUATOR_DIR = f"{REMOTE_PACKAGE_DIR}/tasks/assets/trimul_evaluator"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _judge_subprocess_env() -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"EVOLVENT_API_KEY", AUTH_TOKEN_ENV, "VLIW_JUDGE_TOKEN"}
    }
    env.update({"NO_PROXY": "*", "no_proxy": "*", "PYTHONUNBUFFERED": "1"})
    return env


app = modal.App(APP_NAME)
runtime_secret = modal.Secret.from_name(SECRET_NAME)

judge_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu24.04", add_python="3.13")
    .apt_install("git", "gcc-13", "g++-13", "clang-18")
    .pip_install(
        "ninja~=1.11",
        "wheel~=0.45",
        "requests~=2.32.4",
        "packaging~=25.0",
        "numpy~=2.3",
        "pytest",
        "PyYAML",
    )
    .pip_install(
        "torch>=2.7.0,<2.8.0",
        "torchvision~=0.22",
        "torchaudio>=2.7.0,<2.8.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install("triton==3.3.1")
    .pip_install("fastapi>=0.115", "pydantic>=2")
    .env(
        {
            "PYTHONPATH": "/opt",
            "PYTHONUNBUFFERED": "1",
            "TORCH_CUDA_ARCH_LIST": "9.0 9.0a",
        }
    )
    .add_local_dir(REPO_ROOT / "guidance_ttt", REMOTE_PACKAGE_DIR)
)


@app.function(
    image=judge_image,
    gpu=EVALUATION_GPU_CONFIG,
    secrets=[runtime_secret],
    timeout=1200,
    cpu=8,
    memory=131072,
    max_containers=EVALUATION_MAX_CONTAINERS,
    buffer_containers=1,
    scaledown_window=300,
    block_network=True,
)
@modal.concurrent(max_inputs=1)
@modal.fastapi_endpoint(method="POST", docs=False)
async def evaluate_trimul_candidate(request: Request):
    from fastapi import HTTPException

    from guidance_ttt.verifier.trimul_official_runner import run_official_trimul_evaluation

    expected_token = os.environ.get(AUTH_TOKEN_ENV, "")
    supplied_auth = str(request.headers.get("authorization") or "")
    if not expected_token or supplied_auth != f"Bearer {expected_token}":
        raise HTTPException(status_code=401, detail="unauthorized")
    payload = await request.json()
    solution = payload.get("solution") if isinstance(payload, dict) else None
    if not isinstance(solution, str) or not solution.strip():
        raise HTTPException(status_code=400, detail="solution must be a non-empty string")
    runner_timeout_s = max(1, min(int(payload.get("runner_timeout_s", 520)), 520))
    result = run_official_trimul_evaluation(
        solution,
        evaluator_dir=REMOTE_EVALUATOR_DIR,
        timeout_s=runner_timeout_s,
        subprocess_env=_judge_subprocess_env(),
    )

    import torch
    import triton

    result.update(
        {
            "provider": "modal_http",
            "evaluation_gpu": torch.cuda.get_device_name(0),
            "runtime_versions": {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "triton": triton.__version__,
                "cuda": torch.version.cuda,
            },
        }
    )
    return result
