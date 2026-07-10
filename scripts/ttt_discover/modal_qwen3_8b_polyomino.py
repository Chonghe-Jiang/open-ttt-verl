"""Modal app for the TTT-Discover Frontier-CS Polyomino Qwen3-8B 2-GPU H200 run.

Usage:
    modal run scripts/ttt_discover/modal_qwen3_8b_polyomino.py --action preflight
    modal run scripts/ttt_discover/modal_qwen3_8b_polyomino.py --action prepare
    modal run scripts/ttt_discover/modal_qwen3_8b_polyomino.py --action run
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


REMOTE_ROOT = "/workspace/open-ttt-verl"
APP_NAME = "polynomino-qwen3-8b-polyomino"
DEFAULT_CONFIG = "verl_ttt_discover/config/polyomino_2gpu_h200_qwen3_8b_g4_n16.yaml"
DEFAULT_BASE_IMAGE = "verlai/verl:vllm017.latest"
DEFAULT_GPUS = "0,1"
SOURCE_IGNORE = ["**/__pycache__/**", "**/*.pyc"]


def _load_modal_credentials_from_dotenv() -> None:
    path = Path(__file__).resolve()
    env_path = path.parents[2] / ".env" if len(path.parents) > 2 else path.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in {"MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "MODAL_SECRET_KEY"} and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")
    if "MODAL_TOKEN_SECRET" not in os.environ and "MODAL_SECRET_KEY" in os.environ:
        os.environ["MODAL_TOKEN_SECRET"] = os.environ["MODAL_SECRET_KEY"]


_load_modal_credentials_from_dotenv()

import modal  # noqa: E402


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _optional_hf_secret() -> list[modal.Secret]:
    secret_name = os.environ.get("MODAL_HF_SECRET")
    if not secret_name:
        return []
    return [modal.Secret.from_name(secret_name)]


def _resolve_repo_root() -> Path:
    path = Path(__file__).resolve()
    default_root = path.parents[2] if len(path.parents) > 2 else path.parent
    return Path(os.environ.get("MODAL_REPO_ROOT", default_root))


def _add_repo_sources(image: modal.Image) -> modal.Image:
    root = _resolve_repo_root()
    for rel in ["verl", "verl_ttt_discover", "scripts", "tests", "docs"]:
        path = root / rel
        if path.exists():
            image = image.add_local_dir(path, remote_path=f"{REMOTE_ROOT}/{rel}", ignore=SOURCE_IGNORE)
    for rel in ["pyproject.toml", "setup.py", "requirements-ttt.txt", "requirements-test.txt", "README.md"]:
        path = root / rel
        if path.exists():
            image = image.add_local_file(path, remote_path=f"{REMOTE_ROOT}/{rel}")
    return image


image = modal.Image.from_registry(_env("MODAL_BASE_IMAGE", DEFAULT_BASE_IMAGE))
image = image.env(
    {
        "PYTHONPATH": REMOTE_ROOT,
        "HF_HOME": "/hf_cache",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    }
)
image = image.run_commands(
    "python -m pip install --upgrade pip",
    "python -m pip install 'huggingface_hub[cli]' hf_transfer protobuf sentencepiece tiktoken math-verify",
    "python - <<'PY'\n"
    "import importlib.util\n"
    "print('flash_attn_version=', importlib.import_module('flash_attn').__version__ if importlib.util.find_spec('flash_attn') else 'missing')\n"
    "print(\"attn_implementation='flash_attention_2'\")\n"
    "PY",
)
image = _add_repo_sources(image)

app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name("polynomino-qwen3-8b-hf-cache", create_if_missing=True)
outputs = modal.Volume.from_name("polynomino-qwen3-8b-outputs", create_if_missing=True)


def _run_command(command: list[str], *, model_path: str = "", attn_impl: str = "") -> str:
    env = os.environ.copy()
    env.update(
        {
            "GPUS": _env("MODAL_GPUS", DEFAULT_GPUS),
            "HF_HOME": "/hf_cache",
            "HYDRA_FULL_ERROR": "1",
            "RAY_DEDUP_LOGS": "0",
        }
    )
    if model_path:
        env["MODEL_PATH"] = model_path
    if attn_impl:
        env["ATTN_IMPL"] = attn_impl
    proc = subprocess.Popen(
        command,
        cwd=REMOTE_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    chunks: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        chunks.append(line)
    proc.wait()
    outputs.commit()
    hf_cache.commit()
    if proc.returncode:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(command)}")
    return "".join(chunks)


@app.function(image=image, gpu=_env("MODAL_GPU", "H200:2"), cpu=float(_env("MODAL_CPU", "32")), memory=int(_env("MODAL_MEMORY_MB", "262144")), timeout=3600, volumes={"/hf_cache": hf_cache, f"{REMOTE_ROOT}/outputs": outputs}, secrets=_optional_hf_secret())
def preflight(model_path: str = "", attn_impl: str = "") -> str:
    command = [
        "python",
        "-c",
        "import torch, flash_attn; print('cuda_device_count=', torch.cuda.device_count()); print('flash_attn_version=', flash_attn.__version__); print(\"attn_implementation='flash_attention_2'\")",
    ]
    return _run_command(command, model_path=model_path, attn_impl=attn_impl)


def _split_extra_args(extra_args: str) -> list[str]:
    return extra_args.split() if extra_args else []


@app.function(image=image, gpu=_env("MODAL_GPU", "H200:2"), cpu=float(_env("MODAL_CPU", "32")), memory=int(_env("MODAL_MEMORY_MB", "262144")), timeout=int(_env("MODAL_TIMEOUT_SECONDS", "86400")), volumes={"/hf_cache": hf_cache, f"{REMOTE_ROOT}/outputs": outputs}, secrets=_optional_hf_secret())
def prepare(model_path: str = "", attn_impl: str = "", extra_args: str = "") -> str:
    command = ["bash", "scripts/ttt_discover/run_polyomino_qwen3_8b_2gpu_h200.sh", "--prepare-only"]
    command.extend(_split_extra_args(extra_args))
    return _run_command(command, model_path=model_path, attn_impl=attn_impl)


@app.function(image=image, gpu=_env("MODAL_GPU", "H200:2"), cpu=float(_env("MODAL_CPU", "32")), memory=int(_env("MODAL_MEMORY_MB", "262144")), timeout=int(_env("MODAL_TIMEOUT_SECONDS", "86400")), volumes={"/hf_cache": hf_cache, f"{REMOTE_ROOT}/outputs": outputs}, secrets=_optional_hf_secret())
def run(model_path: str = "", attn_impl: str = "", extra_args: str = "") -> str:
    command = ["bash", "scripts/ttt_discover/run_polyomino_qwen3_8b_2gpu_h200.sh"]
    command.extend(_split_extra_args(extra_args))
    return _run_command(command, model_path=model_path, attn_impl=attn_impl)


@app.local_entrypoint()
def main(action: str = "run", model_path: str = "", attn_impl: str = "", extra_args: str = "") -> None:
    if action == "preflight":
        print(preflight.remote(model_path=model_path, attn_impl=attn_impl))
    elif action == "prepare":
        print(prepare.remote(model_path=model_path, attn_impl=attn_impl, extra_args=extra_args))
    elif action == "run":
        print(run.remote(model_path=model_path, attn_impl=attn_impl, extra_args=extra_args))
    else:
        raise ValueError(f"Unknown action: {action}")
