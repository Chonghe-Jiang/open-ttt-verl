"""Modal app for the TTT-Discover Erdos Qwen3-8B 2-GPU H200 run.

Usage:
    modal run scripts/ttt_discover/modal_qwen3_8b_erdos.py --action prepare
    modal run scripts/ttt_discover/modal_qwen3_8b_erdos.py --action run

Optional local environment knobs before `modal run` or `modal deploy`:
    MODAL_BASE_IMAGE=verlai/verl:vllm017.latest
    MODAL_GPU=H200:2
    MODAL_HF_SECRET=<modal secret name with HF_TOKEN>

The app also loads Modal API credentials from the repository `.env` before the
Modal SDK is imported. `MODAL_SECRET_KEY` is treated as the token secret and is
mirrored to `MODAL_TOKEN_SECRET` when needed.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

REMOTE_ROOT = "/workspace/open-ttt-verl"


def _resolve_repo_root() -> Path:
    current_file = Path(__file__).resolve()
    for parent in (current_file.parent, *current_file.parents):
        if (parent / "pyproject.toml").exists() and (parent / "verl_ttt_discover").exists():
            return parent
    return Path(os.environ.get("MODAL_REPO_ROOT", REMOTE_ROOT))


LOCAL_ROOT = _resolve_repo_root()


def _strip_dotenv_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    if " #" in value:
        value = value.split(" #", 1)[0].rstrip()
    return value


def _load_modal_credentials_from_dotenv() -> None:
    env_file = Path(os.environ.get("MODAL_ENV_FILE", LOCAL_ROOT / ".env"))
    if not env_file.exists():
        return

    dotenv_values: dict[str, str] = {}
    for raw_line in env_file.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key.removeprefix("export ").strip()
        dotenv_values[key] = _strip_dotenv_value(value)

    for key in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "MODAL_SECRET_KEY"):
        if dotenv_values.get(key) and not os.environ.get(key):
            os.environ[key] = dotenv_values[key]

    if not os.environ.get("MODAL_TOKEN_SECRET") and os.environ.get("MODAL_SECRET_KEY"):
        os.environ["MODAL_TOKEN_SECRET"] = os.environ["MODAL_SECRET_KEY"]


_load_modal_credentials_from_dotenv()

import modal  # noqa: E402

APP_NAME = "polynomino-qwen3-8b-erdos"
DEFAULT_CONFIG = "verl_ttt_discover/config/erdos_2gpu_h200_qwen3_8b_g4_n16.yaml"
DEFAULT_BASE_IMAGE = "verlai/verl:vllm017.latest"
DEFAULT_GPUS = "0,1"

SOURCE_IGNORE = ["**/__pycache__/**", "**/*.pyc"]


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _optional_hf_secret() -> list[modal.Secret]:
    secret_name = os.environ.get("MODAL_HF_SECRET")
    if not secret_name:
        return []
    return [modal.Secret.from_name(secret_name)]


def _add_repo_sources(image: modal.Image) -> modal.Image:
    """Copy only the runtime sources needed by the TTT entrypoint."""
    for filename in ("pyproject.toml", "setup.py", "README.md", "requirements-ttt.txt"):
        image = image.add_local_file(
            str(LOCAL_ROOT / filename),
            f"{REMOTE_ROOT}/{filename}",
            copy=True,
        )

    for dirname in ("verl", "verl_ttt_discover"):
        image = image.add_local_dir(
            str(LOCAL_ROOT / dirname),
            f"{REMOTE_ROOT}/{dirname}",
            copy=True,
            ignore=SOURCE_IGNORE,
        )

    image = image.add_local_dir(
        str(LOCAL_ROOT / "scripts" / "ttt_discover"),
        f"{REMOTE_ROOT}/scripts/ttt_discover",
        copy=True,
        ignore=SOURCE_IGNORE,
    )
    return image


image = modal.Image.from_registry(_env("MODAL_BASE_IMAGE", DEFAULT_BASE_IMAGE))
image = _add_repo_sources(image)
image = (
    image.env(
        {
            "HF_HOME": "/hf_cache",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HYDRA_FULL_ERROR": "1",
            "RAY_DEDUP_LOGS": "0",
            "NCCL_P2P_DISABLE": "0",
            "NCCL_SHM_DISABLE": "0",
            "NCCL_IB_DISABLE": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .workdir(REMOTE_ROOT)
    .run_commands(
        "python -m pip install wheel",
        "python -m pip install --no-deps -e .",
        "python -m pip install 'huggingface_hub[cli]' hf_transfer protobuf sentencepiece tiktoken math-verify",
        "python -m compileall -q verl_ttt_discover scripts/ttt_discover",
        "mkdir -p /hf_cache outputs",
    )
)

app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name("polynomino-qwen3-8b-hf-cache", create_if_missing=True)
outputs = modal.Volume.from_name("polynomino-qwen3-8b-outputs", create_if_missing=True)


def _run_command(
    command: list[str],
    *,
    config: str = DEFAULT_CONFIG,
    model_path: str = "",
    attn_impl: str = "",
    extra_env: dict[str, str] | None = None,
) -> str:
    env = os.environ.copy()
    env.update(
        {
            "CONFIG": config,
            "GPUS": DEFAULT_GPUS,
            "HF_HOME": "/hf_cache",
            "HYDRA_FULL_ERROR": "1",
            "RAY_DEDUP_LOGS": "0",
            "NCCL_P2P_DISABLE": "0",
            "NCCL_SHM_DISABLE": "0",
            "NCCL_IB_DISABLE": "1",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        }
    )
    if model_path:
        env["MODEL_PATH"] = model_path
    if attn_impl:
        env["ATTN_IMPL"] = attn_impl
    if extra_env:
        env.update(extra_env)

    subprocess.run(command, cwd=REMOTE_ROOT, env=env, check=True)
    hf_cache.commit()
    outputs.commit()
    return "ok"


@app.function(
    image=image,
    gpu=_env("MODAL_GPU", "H200:2"),
    cpu=float(_env("MODAL_CPU", "32")),
    memory=int(_env("MODAL_MEMORY_MB", "262144")),
    timeout=int(_env("MODAL_TIMEOUT_SECONDS", "86400")),
    volumes={
        "/hf_cache": hf_cache,
        f"{REMOTE_ROOT}/outputs": outputs,
    },
    secrets=_optional_hf_secret(),
)
def preflight() -> str:
    command = [
        "python",
        "-c",
        (
            "import importlib, os, torch; "
            "print(f'torch={torch.__version__}'); "
            "print(f'torch_cuda={torch.version.cuda}'); "
            "print(f'cuda_available={torch.cuda.is_available()}'); "
            "print(f'cuda_device_count={torch.cuda.device_count()}'); "
            "print(f'HF_HOME={os.environ.get(\"HF_HOME\")}'); "
            "mods = ('verl', 'verl_ttt_discover', 'vllm', 'flash_attn'); "
            "[importlib.import_module(m) for m in mods]; "
            "[print(f'import {m}: ok') for m in mods]; "
            "flash_attn = importlib.import_module('flash_attn'); "
            "print(f'flash_attn_version={getattr(flash_attn, \"__version__\", \"unknown\")}'); "
            "from transformers import AutoConfig; "
            "cfg = AutoConfig.from_pretrained(os.environ.get('MODEL_PATH', 'Qwen/Qwen3-8B'), "
            "attn_implementation='flash_attention_2'); "
            "print(f'attn_implementation={getattr(cfg, \"_attn_implementation\", None)}')"
        ),
    ]
    return _run_command(command)


@app.function(
    image=image,
    gpu=_env("MODAL_GPU", "H200:2"),
    cpu=float(_env("MODAL_CPU", "32")),
    memory=int(_env("MODAL_MEMORY_MB", "262144")),
    timeout=int(_env("MODAL_TIMEOUT_SECONDS", "86400")),
    volumes={
        "/hf_cache": hf_cache,
        f"{REMOTE_ROOT}/outputs": outputs,
    },
    secrets=_optional_hf_secret(),
)
def prepare(model_path: str = "", attn_impl: str = "", extra_args: list[str] | None = None) -> str:
    command = ["bash", "scripts/ttt_discover/run_erdos_qwen3_8b_2gpu_h200.sh", "--prepare-only"]
    command.extend(extra_args or [])
    return _run_command(command, model_path=model_path, attn_impl=attn_impl)


@app.function(
    image=image,
    gpu=_env("MODAL_GPU", "H200:2"),
    cpu=float(_env("MODAL_CPU", "32")),
    memory=int(_env("MODAL_MEMORY_MB", "262144")),
    timeout=int(_env("MODAL_TIMEOUT_SECONDS", "86400")),
    volumes={
        "/hf_cache": hf_cache,
        f"{REMOTE_ROOT}/outputs": outputs,
    },
    secrets=_optional_hf_secret(),
)
def run(model_path: str = "", attn_impl: str = "", extra_args: list[str] | None = None) -> str:
    command = ["bash", "scripts/ttt_discover/run_erdos_qwen3_8b_2gpu_h200.sh"]
    command.extend(extra_args or [])
    return _run_command(command, model_path=model_path, attn_impl=attn_impl)


@app.local_entrypoint()
def main(
    action: str = "prepare",
    model_path: str = "",
    attn_impl: str = "",
    extra_args: str = "",
) -> None:
    parsed_extra_args = shlex.split(extra_args)
    if action == "preflight":
        preflight.remote()
    elif action == "prepare":
        prepare.remote(model_path=model_path, attn_impl=attn_impl, extra_args=parsed_extra_args)
    elif action == "run":
        run.remote(model_path=model_path, attn_impl=attn_impl, extra_args=parsed_extra_args)
    else:
        raise ValueError("action must be one of: preflight, prepare, run")
