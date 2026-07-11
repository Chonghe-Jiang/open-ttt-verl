from __future__ import annotations

import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
import time
from pathlib import Path


APP_NAME = "guidance-ttt-polyomino-h200-smoke"
GPU_CONFIG = "H200:4"
HISTORY_GPU_CONFIG = "H200:2"
GPT_OSS_120B_GPU_CONFIG = "H200:5"
GPT_OSS_120B_3GPU_CONFIG = "H200:3"
GPT_OSS_120B_SINGLE_GPU_CONFIG = "H200:1"
REMOTE_REPO_DIR = "/root/guidance"
REMOTE_FRONTIER_DIR = "/opt/Frontier-CS"
REMOTE_CONFIG_PATH = f"{REMOTE_REPO_DIR}/guidance_ttt/config/backup/polyomino_modal_h200_4gpu_smoke.yaml"
REMOTE_OUTPUT_DIR = "/runs/guidance_ttt/polyomino_modal_h200_4gpu_smoke"
REMOTE_HISTORY_CONFIG_PATH = f"{REMOTE_REPO_DIR}/guidance_ttt/config/backup/polyomino_modal_h200_2gpu_history_smoke.yaml"
REMOTE_HISTORY_OUTPUT_DIR = "/runs/guidance_ttt/polyomino_modal_h200_2gpu_history_smoke"
REMOTE_SINGLE_SUMMARY_CONFIG_PATH = f"{REMOTE_REPO_DIR}/guidance_ttt/config/backup/polyomino_modal_h200_2gpu_single_summary.yaml"
REMOTE_SINGLE_SUMMARY_OUTPUT_DIR = "/runs/guidance_ttt/polyomino_modal_h200_2gpu_single_summary"
REMOTE_QWEN_EXEC_SINGLE_SUMMARY_CONFIG_PATH = (
    f"{REMOTE_REPO_DIR}/guidance_ttt/config/backup/polyomino_modal_h200_2gpu_qwen_exec_single_summary.yaml"
)
REMOTE_QWEN_EXEC_SINGLE_SUMMARY_OUTPUT_DIR = "/runs/guidance_ttt/polyomino_modal_h200_2gpu_qwen_exec_single_summary"
REMOTE_OPENROUTER_GPT55_SINGLE_SUMMARY_CONFIG_PATH = (
    f"{REMOTE_REPO_DIR}/guidance_ttt/config/backup/polyomino_modal_h200_2gpu_openrouter_gpt55_single_summary.yaml"
)
REMOTE_OPENROUTER_GPT55_SINGLE_SUMMARY_OUTPUT_DIR = (
    "/runs/guidance_ttt/polyomino_modal_h200_2gpu_openrouter_gpt55_single_summary"
)
REMOTE_OPENROUTER_GPT55_4GPU_FULL_BATCH_CONFIG_PATH = (
    f"{REMOTE_REPO_DIR}/guidance_ttt/config/backup/polyomino_modal_h200_4gpu_openrouter_gpt55_full_batch.yaml"
)
REMOTE_OPENROUTER_GPT55_4GPU_FULL_BATCH_OUTPUT_DIR = (
    "/runs/guidance_ttt/polyomino_modal_h200_4gpu_openrouter_gpt55_full_batch"
)
REMOTE_EVOLVENT_GPT55_4GPU_SHORT_RESPONSE_CONFIG_PATH = (
    f"{REMOTE_REPO_DIR}/guidance_ttt/config/backup/polyomino_modal_h200_4gpu_evolvent_gpt55_short_response.yaml"
)
REMOTE_EVOLVENT_GPT55_4GPU_SHORT_RESPONSE_OUTPUT_DIR = (
    "/runs/guidance_ttt/polyomino_modal_h200_4gpu_evolvent_gpt55_short_response"
)
REMOTE_GPT_OSS_120B_5GPU_GROUP64_CONFIG_PATH = (
    f"{REMOTE_REPO_DIR}/guidance_ttt/config/backup/polyomino_modal_h200_5gpu_gpt_oss_120b_group64.yaml"
)
REMOTE_GPT_OSS_120B_5GPU_GROUP64_OUTPUT_DIR = (
    "/runs/guidance_ttt/polyomino_modal_h200_5gpu_gpt_oss_120b_group64"
)
REMOTE_GPT_OSS_120B_3GPU_GROUP16_CONFIG_PATH = (
    f"{REMOTE_REPO_DIR}/guidance_ttt/config/backup/polyomino_modal_h200_3gpu_gpt_oss_120b_group16.yaml"
)
REMOTE_GPT_OSS_120B_3GPU_GROUP16_OUTPUT_DIR = (
    "/runs/guidance_ttt/polyomino_modal_h200_3gpu_gpt_oss_120b_group16"
)
REMOTE_GPT_OSS_120B_3GPU_GROUP16_H200_TUNED_CONFIG_PATH = (
    f"{REMOTE_REPO_DIR}/guidance_ttt/config/polyomino_modal_h200_3gpu_gpt_oss_120b_group16_h200_tuned.yaml"
)
REMOTE_GPT_OSS_120B_3GPU_GROUP16_H200_TUNED_OUTPUT_DIR = (
    "/runs/guidance_ttt/polyomino_modal_h200_3gpu_gpt_oss_120b_group16_h200_tuned_50step"
)
REMOTE_GPT_OSS_120B_3GPU_BATCH8_GROUP8_TEMP09_CONFIG_PATH = (
    f"{REMOTE_REPO_DIR}/guidance_ttt/config/polyomino_modal_h200_3gpu_gpt_oss_120b_batch8_group8_temp09.yaml"
)
REMOTE_GPT_OSS_120B_3GPU_BATCH8_GROUP8_TEMP09_OUTPUT_DIR = (
    "/runs/guidance_ttt/polyomino_modal_h200_3gpu_gpt_oss_120b_batch8_group8_temp09_20step"
)
REMOTE_GPT_OSS_120B_3GPU_BATCH8_GROUP16_CONCURRENCY16_CONFIG_PATH = (
    f"{REMOTE_REPO_DIR}/guidance_ttt/config/"
    "polyomino_modal_h200_3gpu_gpt_oss_120b_batch8_group16_concurrency16_1step.yaml"
)
REMOTE_GPT_OSS_120B_3GPU_BATCH8_GROUP16_CONCURRENCY16_OUTPUT_DIR = (
    "/runs/guidance_ttt/polyomino_modal_h200_3gpu_gpt_oss_120b_batch8_group16_concurrency16_1step"
)
REMOTE_GPT_OSS_120B_3GPU_BATCH8_GROUP16_PUCT_FIX_CONFIG_PATH = (
    f"{REMOTE_REPO_DIR}/guidance_ttt/config/"
    "polyomino_modal_h200_3gpu_gpt_oss_120b_batch8_group16_puct_fix_1step.yaml"
)
REMOTE_GPT_OSS_120B_3GPU_BATCH8_GROUP16_PUCT_FIX_OUTPUT_DIR = (
    "/runs/guidance_ttt/polyomino_modal_h200_3gpu_gpt_oss_120b_batch8_group16_puct_fix_1step"
)
REMOTE_GPT_OSS_120B_3GPU_BATCH8_GROUP16_CONCURRENCY16_50STEP_CONFIG_PATH = (
    f"{REMOTE_REPO_DIR}/guidance_ttt/config/"
    "polyomino_modal_h200_3gpu_gpt_oss_120b_batch8_group16_concurrency16_50step.yaml"
)
REMOTE_GPT_OSS_120B_3GPU_BATCH8_GROUP16_CONCURRENCY16_50STEP_OUTPUT_DIR = (
    "/runs/guidance_ttt/polyomino_modal_h200_3gpu_gpt_oss_120b_batch8_group16_concurrency16_50step"
)
REMOTE_GPT_OSS_120B_3GPU_BATCH8_GROUP16_PROMPT_REFINEMENT_50STEP_CONFIG_PATH = (
    f"{REMOTE_REPO_DIR}/guidance_ttt/config/"
    "polyomino_modal_h200_3gpu_gpt_oss_120b_batch8_group16_prompt_refinement_50step.yaml"
)
REMOTE_GPT_OSS_120B_3GPU_BATCH8_GROUP16_PROMPT_REFINEMENT_50STEP_OUTPUT_DIR = (
    "/runs/guidance_ttt/polyomino_modal_h200_3gpu_gpt_oss_120b_batch8_group16_prompt_refinement_50step"
)
REMOTE_GPT_OSS_120B_BOOTSTRAP_SEED_CONFIG_PATH = (
    f"{REMOTE_REPO_DIR}/guidance_ttt/config/polyomino_modal_h200_gpt_oss_120b_bootstrap_seed.yaml"
)
REMOTE_GPT_OSS_120B_BOOTSTRAP_SEED_OUTPUT_DIR = (
    "/runs/guidance_ttt/polyomino_modal_h200_gpt_oss_120b_bootstrap_seed"
)
JUDGE_LOG_PATH = "/tmp/frontier_judge.log"
GPT_OSS_120B_SERVER_LOG_PATH = "/tmp/gpt_oss_120b_vllm.log"
GPT_OSS_120B_XML_PROBE_OUTPUT_PATH = "/runs/guidance_ttt/gpt_oss_120b_xml_probe.json"
FLASH_ATTN_TORCH29_CU12_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
    "flash_attn-2.8.3%2Bcu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _strip_env_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _load_dotenv_key(path: Path, key: str) -> str | None:
    if not path.exists():
        return None
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if name.startswith("export "):
            name = name.removeprefix("export ").strip()
        if name == key:
            return _strip_env_value(value)
    return None


def _load_dotenv_keys(path: Path, keys: set[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if name.startswith("export "):
            name = name.removeprefix("export ").strip()
        if name in keys:
            values[name] = _strip_env_value(value)
    return values


def _configure_modal_auth(env_path: Path = REPO_ROOT / ".env") -> dict[str, str]:
    dotenv_values = _load_dotenv_keys(env_path, {"WORKSPACE", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "MODAL_SECRET_KEY"})
    if "MODAL_TOKEN_SECRET" not in dotenv_values and dotenv_values.get("MODAL_SECRET_KEY"):
        dotenv_values["MODAL_TOKEN_SECRET"] = dotenv_values["MODAL_SECRET_KEY"]
    if "MODAL_TOKEN_SECRET" not in os.environ and os.environ.get("MODAL_SECRET_KEY"):
        os.environ["MODAL_TOKEN_SECRET"] = os.environ["MODAL_SECRET_KEY"]
    for key in ("WORKSPACE", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"):
        if key not in os.environ and dotenv_values.get(key):
            os.environ[key] = dotenv_values[key]
    if os.environ.get("MODAL_TOKEN_ID") and not os.environ.get("MODAL_TOKEN_SECRET"):
        token_secret = _modal_token_secret_for_id(os.environ["MODAL_TOKEN_ID"])
        if token_secret:
            os.environ["MODAL_TOKEN_SECRET"] = token_secret
    return {key: os.environ[key] for key in ("WORKSPACE", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET") if key in os.environ}


def _validate_modal_token_env() -> None:
    has_token_id = bool(os.environ.get("MODAL_TOKEN_ID"))
    has_token_secret = bool(os.environ.get("MODAL_TOKEN_SECRET"))
    if has_token_id != has_token_secret:
        missing = "MODAL_TOKEN_SECRET" if has_token_id else "MODAL_TOKEN_ID"
        present = "MODAL_TOKEN_ID" if has_token_id else "MODAL_TOKEN_SECRET"
        raise RuntimeError(
            f".env defines {present} but not {missing}. Modal API token auth requires both values; "
            "no matching token secret for that id was found in ~/.modal.toml."
        )


def _modal_token_secret_for_id(token_id: str, path: Path | None = None) -> str | None:
    config_path = path or (Path.home() / ".modal.toml")
    if not config_path.exists():
        return None
    data = tomllib.loads(config_path.read_text())
    for section in data.values():
        if isinstance(section, dict) and section.get("token_id") == token_id:
            token_secret = section.get("token_secret")
            return str(token_secret) if token_secret else None
    return None


def _sync_modal_profile(profile: str) -> None:
    modal_config = importlib.import_module("modal.config")
    if hasattr(modal_config, "_profile"):
        setattr(modal_config, "_profile", profile)


def _modal_profile_names(path: Path | None = None) -> set[str]:
    config_path = path or (Path.home() / ".modal.toml")
    if not config_path.exists():
        return set()
    data = tomllib.loads(config_path.read_text())
    return {str(name) for name, section in data.items() if isinstance(section, dict)}


def _configure_modal_profile(env_path: Path = REPO_ROOT / ".env") -> str | None:
    workspace = os.environ.get("WORKSPACE") or _load_dotenv_key(env_path, "WORKSPACE")
    if workspace and "MODAL_PROFILE" not in os.environ and workspace in _modal_profile_names():
        os.environ["MODAL_PROFILE"] = workspace
        _sync_modal_profile(workspace)
    return workspace


def _parse_modal_token_workspace(token_info: str) -> tuple[str, str]:
    match = re.search(r"^Workspace:\s*(?P<name>.*?)\s*\((?P<id>[^)]+)\)\s*$", token_info, re.MULTILINE)
    if not match:
        raise RuntimeError(f"Could not parse workspace from modal token info:\n{token_info}")
    return match.group("name"), match.group("id")


def _current_modal_token_workspace() -> tuple[str, str]:
    result = subprocess.run(
        ["modal", "token", "info"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return _parse_modal_token_workspace(result.stdout)


def _should_validate_modal_workspace() -> bool:
    if os.environ.get("MODAL_IS_REMOTE") == "1":
        return False
    if os.environ.get("GUIDANCE_TTT_SKIP_MODAL_WORKSPACE_CHECK"):
        return False
    if os.environ.get("GUIDANCE_TTT_ENFORCE_MODAL_WORKSPACE"):
        return True
    return "modal_polyomino_h200_smoke.py" in "\n".join(sys.argv)


def _validate_modal_workspace(expected_workspace: str | None) -> None:
    if not expected_workspace:
        raise RuntimeError(".env must contain WORKSPACE=<modal-workspace-name-or-id>.")
    actual_name, actual_id = _current_modal_token_workspace()
    if expected_workspace not in {actual_name, actual_id}:
        raise RuntimeError(
            "Modal token workspace does not match .env WORKSPACE.\n"
            f"  .env WORKSPACE: {expected_workspace}\n"
            f"  token workspace: {actual_name} ({actual_id})\n\n"
            "Create or set a Modal token that is actually connected to the .env workspace, then retry:\n"
            f"  modal token new --profile {expected_workspace!r} --activate\n"
            "  modal token info\n\n"
            "The token-info Workspace line must show the .env workspace name or id."
        )


MODAL_AUTH = _configure_modal_auth()
MODAL_WORKSPACE = _configure_modal_profile()
if _should_validate_modal_workspace():
    _validate_modal_token_env()
    _validate_modal_workspace(MODAL_WORKSPACE)

import modal  # noqa: E402

app = modal.App(APP_NAME)
runs_volume = modal.Volume.from_name("guidance-ttt-runs", create_if_missing=True)
cache_volume = modal.Volume.from_name("guidance-ttt-cache", create_if_missing=True)
openrouter_secret = modal.Secret.from_name("openrouter-api-key")
evolvent_secret = modal.Secret.from_name("evolvent-api-key")


def training_command() -> list[str]:
    return [
        "python",
        "-m",
        "guidance_ttt.main_erdos",
        "--config",
        REMOTE_CONFIG_PATH,
    ]


def history_training_command() -> list[str]:
    return [
        "python",
        "-m",
        "guidance_ttt.main_erdos",
        "--config",
        REMOTE_HISTORY_CONFIG_PATH,
    ]


def single_summary_training_command() -> list[str]:
    return [
        "python",
        "-m",
        "guidance_ttt.main_erdos",
        "--config",
        REMOTE_SINGLE_SUMMARY_CONFIG_PATH,
    ]


def single_summary_bootstrap_command() -> list[str]:
    return [
        "python",
        "-m",
        "guidance_ttt.main_erdos",
        "--config",
        REMOTE_SINGLE_SUMMARY_CONFIG_PATH,
        "--bootstrap-only",
    ]


def qwen_exec_single_summary_training_command() -> list[str]:
    return [
        "python",
        "-m",
        "guidance_ttt.main_erdos",
        "--config",
        REMOTE_QWEN_EXEC_SINGLE_SUMMARY_CONFIG_PATH,
    ]


def qwen_exec_single_summary_bootstrap_command() -> list[str]:
    return [
        "python",
        "-m",
        "guidance_ttt.main_erdos",
        "--config",
        REMOTE_QWEN_EXEC_SINGLE_SUMMARY_CONFIG_PATH,
        "--bootstrap-only",
    ]


def openrouter_gpt55_single_summary_training_command() -> list[str]:
    return [
        "python",
        "-m",
        "guidance_ttt.main_erdos",
        "--config",
        REMOTE_OPENROUTER_GPT55_SINGLE_SUMMARY_CONFIG_PATH,
    ]


def openrouter_gpt55_single_summary_bootstrap_command() -> list[str]:
    return [
        "python",
        "-m",
        "guidance_ttt.main_erdos",
        "--config",
        REMOTE_OPENROUTER_GPT55_SINGLE_SUMMARY_CONFIG_PATH,
        "--bootstrap-only",
    ]


def openrouter_gpt55_4gpu_full_batch_training_command() -> list[str]:
    return [
        "python",
        "-m",
        "guidance_ttt.main_erdos",
        "--config",
        REMOTE_OPENROUTER_GPT55_4GPU_FULL_BATCH_CONFIG_PATH,
    ]


def evolvent_gpt55_4gpu_short_response_training_command() -> list[str]:
    return [
        "python",
        "-m",
        "guidance_ttt.main_erdos",
        "--config",
        REMOTE_EVOLVENT_GPT55_4GPU_SHORT_RESPONSE_CONFIG_PATH,
    ]


def gpt_oss_120b_5gpu_group64_training_command() -> list[str]:
    return [
        "python",
        "-m",
        "guidance_ttt.main_erdos",
        "--config",
        REMOTE_GPT_OSS_120B_5GPU_GROUP64_CONFIG_PATH,
    ]


def gpt_oss_120b_3gpu_group16_training_command() -> list[str]:
    return [
        "python",
        "-m",
        "guidance_ttt.main_erdos",
        "--config",
        REMOTE_GPT_OSS_120B_3GPU_GROUP16_CONFIG_PATH,
    ]


def gpt_oss_120b_3gpu_group16_h200_tuned_training_command() -> list[str]:
    return [
        "python",
        "-m",
        "guidance_ttt.main_erdos",
        "--config",
        REMOTE_GPT_OSS_120B_3GPU_GROUP16_H200_TUNED_CONFIG_PATH,
    ]


def gpt_oss_120b_3gpu_batch8_group8_temp09_training_command() -> list[str]:
    return [
        "python",
        "-m",
        "guidance_ttt.main_erdos",
        "--config",
        REMOTE_GPT_OSS_120B_3GPU_BATCH8_GROUP8_TEMP09_CONFIG_PATH,
    ]


def gpt_oss_120b_3gpu_batch8_group16_concurrency16_training_command() -> list[str]:
    return [
        "python",
        "-m",
        "guidance_ttt.main_erdos",
        "--config",
        REMOTE_GPT_OSS_120B_3GPU_BATCH8_GROUP16_CONCURRENCY16_CONFIG_PATH,
    ]


def gpt_oss_120b_3gpu_batch8_group16_puct_fix_training_command() -> list[str]:
    return [
        "python",
        "-m",
        "guidance_ttt.main_erdos",
        "--config",
        REMOTE_GPT_OSS_120B_3GPU_BATCH8_GROUP16_PUCT_FIX_CONFIG_PATH,
    ]


def gpt_oss_120b_3gpu_batch8_group16_concurrency16_50step_training_command() -> list[str]:
    return [
        "python",
        "-m",
        "guidance_ttt.main_erdos",
        "--config",
        REMOTE_GPT_OSS_120B_3GPU_BATCH8_GROUP16_CONCURRENCY16_50STEP_CONFIG_PATH,
    ]


def gpt_oss_120b_3gpu_batch8_group16_prompt_refinement_50step_training_command() -> list[str]:
    return [
        "python",
        "-m",
        "guidance_ttt.main_erdos",
        "--config",
        REMOTE_GPT_OSS_120B_3GPU_BATCH8_GROUP16_PROMPT_REFINEMENT_50STEP_CONFIG_PATH,
    ]


def gpt_oss_120b_bootstrap_seed_command() -> list[str]:
    return [
        "python",
        "-m",
        "guidance_ttt.main_erdos",
        "--config",
        REMOTE_GPT_OSS_120B_BOOTSTRAP_SEED_CONFIG_PATH,
        "--bootstrap-only",
    ]


def _assert_training_packages_available() -> None:
    checks = [
        (
            "flash_attn",
            "import flash_attn; print('flash_attn', getattr(flash_attn, '__version__', 'unknown'))",
        ),
        ("torch", "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"),
        ("vllm", "import vllm; print('vllm', vllm.__version__)"),
    ]
    for name, code in checks:
        try:
            subprocess.run([sys.executable, "-c", code], check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Required training package {name!r} is not importable in the Modal image.") from exc


def _reset_output_dir(output_dir: str = REMOTE_OUTPUT_DIR) -> None:
    path = Path(output_dir)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _repo_ignore(path: Path) -> bool:
    try:
        rel = path.relative_to(REPO_ROOT)
    except ValueError:
        return False
    parts = rel.parts
    if not parts:
        return False
    ignored_roots = {
        ".git",
        ".hf_cache",
        ".pytest_cache",
        ".ray_tmp",
        ".tmp",
        ".triton_cache",
        "outputs",
        "reference",
        "verl.egg-info",
    }
    return parts[0] in ignored_roots or "__pycache__" in parts


def _run_streaming(cmd: list[str], *, cwd: str | None = None, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    process = subprocess.Popen(cmd, cwd=cwd, env=env)
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}: {' '.join(cmd)}")


def _wait_for_judge(url: str = "http://127.0.0.1:8081/problems", *, timeout_s: int = 120) -> None:
    import urllib.request

    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                payload = response.read().decode()
            print(f"Judge ready: {payload[:300]}", flush=True)
            return
        except Exception as exc:
            last_error = exc
            time.sleep(2)
    raise RuntimeError(f"Judge did not become ready within {timeout_s}s: {last_error}")


def _wait_for_tcp(host: str, port: int, *, timeout_s: int = 120, process: subprocess.Popen | None = None) -> None:
    import socket

    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            _print_judge_log()
            raise RuntimeError(f"Judge process exited before {host}:{port} became ready")
        try:
            with socket.create_connection((host, port), timeout=5):
                print(f"TCP ready: {host}:{port}", flush=True)
                return
        except Exception as exc:
            last_error = exc
            time.sleep(2)
    _print_judge_log()
    raise RuntimeError(f"TCP endpoint {host}:{port} did not become ready within {timeout_s}s: {last_error}")


def _print_log_tail(path: str, *, label: str, max_bytes: int = 20000) -> None:
    log_path = Path(path)
    if not log_path.exists():
        print(f"{label} log does not exist: {path}", flush=True)
        return
    payload = log_path.read_bytes()[-max_bytes:].decode(errors="replace")
    print(f"----- {label} log tail ({path}) -----\n{payload}\n----- end {label} log -----", flush=True)


def _print_judge_log(max_bytes: int = 20000) -> None:
    _print_log_tail(JUDGE_LOG_PATH, label="judge", max_bytes=max_bytes)


def _start_judge(*, workers: int = 8) -> subprocess.Popen:
    workers = max(1, int(workers))
    env = {
        **__import__("os").environ,
        "PORT": "8081",
        "GJ_ADDR": "http://127.0.0.1:5050",
        "JUDGE_WORKERS": str(workers),
        "GJ_PARALLELISM": str(workers),
        "ES_PRE_FORK": "0",
        "SAVE_OUTPUTS": "false",
    }
    log_file = open(JUDGE_LOG_PATH, "wb")
    process = subprocess.Popen(
        ["node", "/app/server.js"],
        cwd="/app",
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    process._guidance_ttt_log_file = log_file  # type: ignore[attr-defined]
    _wait_for_judge()
    return process


def _stop_judge(process: subprocess.Popen) -> None:
    process.terminate()
    process.wait(timeout=10)
    log_file = getattr(process, "_guidance_ttt_log_file", None)
    if log_file is not None:
        log_file.close()


def _start_gpt_oss_120b_server(*, cuda_visible_devices: str = "4", max_num_seqs: int = 8) -> subprocess.Popen:
    max_num_seqs = max(1, int(max_num_seqs))
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
        "VLLM_NO_USAGE_STATS": "1",
        "HF_HOME": "/cache/huggingface",
        "HF_DATASETS_CACHE": "/cache/huggingface/datasets",
        "HUGGINGFACE_HUB_CACHE": "/cache/huggingface/hub",
        "TRANSFORMERS_CACHE": "/cache/huggingface/hub",
    }
    cmd = [
        "vllm",
        "serve",
        "openai/gpt-oss-120b",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--dtype",
        "auto",
        "--trust-remote-code",
        "--tensor-parallel-size",
        "1",
        "--gpu-memory-utilization",
        "0.88",
        "--max-model-len",
        "32768",
        "--max-num-seqs",
        str(max_num_seqs),
        "--enforce-eager",
        "--download-dir",
        "/cache/huggingface/hub",
    ]
    print(f"+ CUDA_VISIBLE_DEVICES={cuda_visible_devices} " + " ".join(cmd), flush=True)
    log_file = open(GPT_OSS_120B_SERVER_LOG_PATH, "wb")
    process = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    process._guidance_ttt_log_file = log_file  # type: ignore[attr-defined]
    try:
        _wait_for_tcp("127.0.0.1", 8000, timeout_s=30 * 60, process=process)
    except Exception:
        _print_log_tail(GPT_OSS_120B_SERVER_LOG_PATH, label="gpt-oss-120b-vllm")
        raise
    return process


def _stop_process(process: subprocess.Popen) -> None:
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)
    log_file = getattr(process, "_guidance_ttt_log_file", None)
    if log_file is not None:
        log_file.close()


def _gojudge_run_one(command: dict[str, object]) -> dict[str, object]:
    import urllib.request

    payload = json.dumps({"cmd": [command]}).encode()
    request = urllib.request.Request(
        "http://127.0.0.1:5050/run",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        data = json.loads(response.read().decode())
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        raise RuntimeError(f"Unexpected go-judge response: {data!r}")
    return data[0]


def _post_openai_chat_completion(payload: dict[str, object], *, timeout_s: int = 600) -> dict[str, object]:
    import urllib.request

    request = urllib.request.Request(
        "http://127.0.0.1:8000/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": "Bearer local-vllm",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode())


def _xml_probe_payload(user_prompt: str, *, max_tokens: int = 2048) -> dict[str, object]:
    return {
        "model": "openai/gpt-oss-120b",
        "messages": [
            {
                "role": "system",
                "content": "You are an execution model. Output exactly the requested XML blocks and no extra text.",
            },
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }


def _run_gpt_oss_xml_probe() -> dict[str, object]:
    prompts = [
        """Return exactly three top-level XML blocks and no extra text.

<execution_thinking>
Say one short sentence.
</execution_thinking>

<solution>
```cpp
#include <bits/stdc++.h>
int main(){return 0;}
```
</solution>

<summary>
Say one short sentence and close this XML block.
</summary>""",
        """Your response must contain exactly these blocks:
<execution_thinking>
Briefly describe a simple algorithm.
</execution_thinking>
<solution>
```cpp
#include <bits/stdc++.h>
using namespace std;
int main(){ cout << "ok\\n"; }
```
</solution>
<summary>
Summarize the algorithm in one sentence.
</summary>

Do not omit any closing XML tag. The final characters of your answer must be </summary>.""",
        """Write three XML blocks: execution_thinking, solution, summary.
The summary is the final block and must end with a literal closing tag.
No markdown outside the solution fence.""",
    ]
    results = []
    for index, prompt in enumerate(prompts):
        response = _post_openai_chat_completion(_xml_probe_payload(prompt), timeout_s=900)
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        text = str(message.get("content") or choice.get("text") or "")
        results.append(
            {
                "index": index,
                "prompt": prompt,
                "finish_reason": choice.get("finish_reason"),
                "usage": response.get("usage"),
                "text": text,
                "text_len": len(text),
                "has_execution_thinking_close": "</execution_thinking>" in text,
                "has_solution_close": "</solution>" in text,
                "has_summary_open": "<summary>" in text,
                "has_summary_close": "</summary>" in text,
                "tail": text[-1000:],
            }
        )
    summary = {
        "model": "openai/gpt-oss-120b",
        "result_count": len(results),
        "summary_close_count": sum(1 for result in results if result["has_summary_close"]),
        "results": results,
    }
    output_path = Path(GPT_OSS_120B_XML_PROBE_OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def _direct_gojudge_smoke() -> None:
    true_result = _gojudge_run_one(
        {
            "args": ["/bin/true"],
            "env": ["PATH=/usr/bin:/bin"],
            "files": [{"content": ""}, {"name": "stdout", "max": 1024}, {"name": "stderr", "max": 1024}],
            "cpuLimit": int(1e9),
            "memoryLimit": 16 << 20,
            "procLimit": 5,
        }
    )
    print("Direct go-judge /bin/true response:", json.dumps(true_result, indent=2), flush=True)

    compile_result = _gojudge_run_one(
        {
            "args": ["/usr/bin/g++", "main.cpp", "-O2", "-pipe", "-std=gnu++17", "-o", "a"],
            "env": ["PATH=/usr/bin:/bin"],
            "files": [{"content": ""}, {"name": "stdout", "max": 65536}, {"name": "stderr", "max": 65536}],
            "copyIn": {"main.cpp": {"content": "int main(){return 0;}"}},
            "copyOut": ["stdout", "stderr"],
            "copyOutCached": ["a"],
            "cpuLimit": int(10e9),
            "memoryLimit": 512 << 20,
            "procLimit": 50,
        }
    )
    print("Direct go-judge C++ compile response:", json.dumps(compile_result, indent=2), flush=True)


def _summarize_output(output_dir: str = REMOTE_OUTPUT_DIR) -> dict[str, object]:
    library_path = Path(output_dir) / "library.json"
    summary: dict[str, object] = {
        "output_dir": output_dir,
        "library_exists": library_path.exists(),
    }
    if library_path.exists():
        data = json.loads(library_path.read_text())
        entries = data.get("entries") or {}
        nodes = data.get("nodes") or {}
        summary.update(
            {
                "entry_count": len(entries),
                "node_count": len(nodes),
                "best_node_id": data.get("best_node_id"),
                "statuses": sorted(
                    {
                        str(entry.get("verifier_status"))
                        for entry in entries.values()
                        if isinstance(entry, dict)
                    }
                ),
            }
        )
    return summary


def _summarize_puct_group_accounting(output_dir: str) -> dict[str, object]:
    library_path = Path(output_dir) / "library.json"
    if not library_path.exists():
        return {"output_dir": output_dir, "library_exists": False}

    data = json.loads(library_path.read_text())
    config = data.get("config") or {}
    groups = data.get("groups") or {}
    nodes = data.get("nodes") or {}
    puct_n = data.get("puct_n") or {}
    selected_group_counts: dict[str, int] = {}
    group_details: list[dict[str, object]] = []
    for group_uid, group in sorted(groups.items()):
        if not isinstance(group, dict):
            group_details.append(
                {
                    "group_uid": str(group_uid),
                    "selected_node_id": None,
                    "submitted": None,
                    "finalized": False,
                    "child_count": None,
                }
            )
            continue
        selected_node_id = str(group.get("selected_node_id") or "")
        selected_group_counts[selected_node_id] = selected_group_counts.get(selected_node_id, 0) + 1
        children = group.get("children") or []
        group_details.append(
            {
                "group_uid": str(group_uid),
                "selected_node_id": selected_node_id,
                "submitted": int(group.get("submitted", 0)),
                "finalized": bool(group.get("finalized", False)),
                "child_count": len(children) if isinstance(children, list) else None,
            }
        )

    selected_node_stats = {}
    for node_id, group_count in selected_group_counts.items():
        node = nodes.get(node_id) or {}
        selected_node_stats[node_id] = {
            "group_count": group_count,
            "node_visits": int(node.get("visits", 0)) if isinstance(node, dict) else None,
            "puct_n": int(puct_n.get(node_id, 0)),
        }

    return {
        "output_dir": output_dir,
        "library_exists": True,
        "rollout_n": int(config.get("rollout_n", data.get("rollout_n", 0))),
        "group_count": len(groups),
        "finalized_group_count": sum(1 for group in group_details if group["finalized"]),
        "puct_T": int(data.get("puct_T", 0)),
        "group_details": group_details,
        "selected_node_stats": selected_node_stats,
    }


def _validate_puct_group_accounting(
    summary: dict[str, object],
    *,
    expected_groups: int,
    expected_rollout_n: int,
) -> None:
    errors: list[str] = []
    if not summary.get("library_exists"):
        errors.append("library.json was not created")
    if summary.get("rollout_n") != expected_rollout_n:
        errors.append(f"rollout_n={summary.get('rollout_n')!r}, expected {expected_rollout_n}")
    if summary.get("group_count") != expected_groups:
        errors.append(f"group_count={summary.get('group_count')!r}, expected {expected_groups}")
    if summary.get("finalized_group_count") != expected_groups:
        errors.append(
            f"finalized_group_count={summary.get('finalized_group_count')!r}, expected {expected_groups}"
        )
    if summary.get("puct_T") != expected_groups:
        errors.append(f"puct_T={summary.get('puct_T')!r}, expected {expected_groups}")

    group_details = summary.get("group_details") or []
    if isinstance(group_details, list):
        for group in group_details:
            if not isinstance(group, dict):
                errors.append(f"malformed group record: {group!r}")
                continue
            if group.get("submitted") != expected_rollout_n:
                errors.append(
                    f"group {group.get('group_uid')!r} submitted={group.get('submitted')!r}, "
                    f"expected {expected_rollout_n}"
                )
            if group.get("child_count") != expected_rollout_n:
                errors.append(
                    f"group {group.get('group_uid')!r} child_count={group.get('child_count')!r}, "
                    f"expected {expected_rollout_n}"
                )

    selected_node_stats = summary.get("selected_node_stats") or {}
    if isinstance(selected_node_stats, dict):
        for node_id, stats in selected_node_stats.items():
            if not isinstance(stats, dict):
                errors.append(f"malformed selected-node record for {node_id!r}: {stats!r}")
                continue
            group_count = stats.get("group_count")
            if stats.get("node_visits") != group_count:
                errors.append(
                    f"selected node {node_id!r} visits={stats.get('node_visits')!r}, "
                    f"expected one acquisition per group ({group_count!r})"
                )
            puct_n = stats.get("puct_n")
            if not isinstance(puct_n, int) or not isinstance(group_count, int):
                errors.append(f"selected node {node_id!r} has malformed PUCT statistics: {stats!r}")
            elif puct_n < group_count or puct_n > expected_groups:
                errors.append(
                    f"selected node {node_id!r} puct_n={puct_n}, expected between its "
                    f"direct group count {group_count} and total group count {expected_groups}"
                )
        if len(selected_node_stats) == 1:
            only_stats = next(iter(selected_node_stats.values()))
            if isinstance(only_stats, dict) and only_stats.get("puct_n") != expected_groups:
                errors.append(
                    f"single selected node puct_n={only_stats.get('puct_n')!r}, expected {expected_groups}"
                )

    if errors:
        raise RuntimeError("PUCT group-accounting acceptance failed: " + "; ".join(errors))


def _summarize_history_extraction(output_dir: str = REMOTE_HISTORY_OUTPUT_DIR) -> dict[str, object]:
    library_path = Path(output_dir) / "library.json"
    if not library_path.exists():
        return {"output_dir": output_dir, "library_exists": False}
    data = json.loads(library_path.read_text())
    entries = [entry for entry in (data.get("entries") or {}).values() if isinstance(entry, dict)]
    guidance_ok = 0
    raw_summary_ok = 0
    canonical_summary_ok = 0
    execution_text_ok = 0
    statuses: dict[str, int] = {}
    examples: list[dict[str, object]] = []
    for entry in entries:
        metadata = entry.get("metadata") or {}
        status = str(entry.get("verifier_status"))
        statuses[status] = statuses.get(status, 0) + 1
        if bool(metadata.get("guidance_format_ok")) and str(entry.get("guidance") or "").strip():
            guidance_ok += 1
        if str(metadata.get("raw_model_summary") or "").strip():
            raw_summary_ok += 1
        if str(entry.get("summary") or "").strip():
            canonical_summary_ok += 1
        if str(metadata.get("execution_text") or "").strip():
            execution_text_ok += 1
        if len(examples) < 3:
            examples.append(
                {
                    "entry_id": entry.get("id"),
                    "status": status,
                    "guidance_format_ok": bool(metadata.get("guidance_format_ok")),
                    "guidance_head": str(entry.get("guidance") or "")[:300],
                    "raw_summary_head": str(metadata.get("raw_model_summary") or "")[:300],
                    "canonical_summary_head": str(entry.get("summary") or "")[:300],
                }
            )
    return {
        "output_dir": output_dir,
        "library_exists": True,
        "entry_count": len(entries),
        "guidance_ok_count": guidance_ok,
        "raw_summary_ok_count": raw_summary_ok,
        "canonical_summary_ok_count": canonical_summary_ok,
        "execution_text_ok_count": execution_text_ok,
        "statuses": statuses,
        "examples": examples,
    }


def _dump_prompt_answer_artifacts(output_dir: str = REMOTE_SINGLE_SUMMARY_OUTPUT_DIR) -> dict[str, object]:
    output_path = Path(output_dir)
    library_path = output_path / "library.json"
    if not library_path.exists():
        return {"output_dir": output_dir, "library_exists": False}
    data = json.loads(library_path.read_text())
    entries = [
        _prompt_answer_entry_record(entry)
        for entry in (data.get("entries") or {}).values()
        if isinstance(entry, dict)
    ]
    entries.sort(key=lambda item: (int(item.get("timestep") or 0), str(item.get("entry_id") or "")))
    dump = {
        "output_dir": output_dir,
        "library_path": str(library_path),
        "entry_count": len(entries),
        "best_node_id": data.get("best_node_id"),
        "entries": entries,
    }
    json_path = output_path / "prompt_answer_dump.json"
    markdown_path = output_path / "prompt_answer_dump.md"
    json_path.write_text(json.dumps(dump, indent=2, sort_keys=True))
    markdown_path.write_text(_prompt_answer_dump_markdown(dump))
    return {
        "output_dir": output_dir,
        "library_exists": True,
        "entry_count": len(entries),
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
    }


def _prompt_answer_entry_record(entry: dict[str, object]) -> dict[str, object]:
    metadata = entry.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        "entry_id": entry.get("id"),
        "timestep": entry.get("timestep"),
        "status": entry.get("verifier_status"),
        "message": entry.get("verifier_message"),
        "reward": entry.get("verifier_reward"),
        "raw_score": entry.get("verifier_raw_score"),
        "guidance_format_ok": bool(metadata.get("guidance_format_ok")),
        "guidance": entry.get("guidance", ""),
        "raw_guidance_text": metadata.get("raw_guidance_text", ""),
        "raw_guidance_with_specials": metadata.get("raw_guidance_with_specials", ""),
        "guidance_prompt": metadata.get("guidance_prompt") or {},
        "execution_prompt": metadata.get("execution_prompt") or {},
        "execution_output": metadata.get("execution_text", ""),
        "raw_model_summary": metadata.get("raw_model_summary", ""),
        "canonical_summary": entry.get("summary", ""),
        "solution": entry.get("solution", ""),
        "execution_thinking": entry.get("execution_thinking", ""),
        "execution_provider": metadata.get("execution_provider"),
        "execution_model": metadata.get("execution_model"),
        "execution_finish_reason": metadata.get("execution_finish_reason"),
        "verification_artifacts": metadata.get("verification_artifacts") or {},
    }


def _prompt_answer_dump_markdown(dump: dict[str, object]) -> str:
    lines = [
        "# Polyomino Modal Prompt/Answer Dump",
        "",
        f"- Output dir: `{dump.get('output_dir')}`",
        f"- Library: `{dump.get('library_path')}`",
        f"- Entry count: `{dump.get('entry_count')}`",
        f"- Best node id: `{dump.get('best_node_id')}`",
        "",
    ]
    entries = dump.get("entries") or []
    if not isinstance(entries, list) or not entries:
        lines.extend(["No library entries were found.", ""])
        return "\n".join(lines)
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        lines.extend(
            [
                f"## Entry `{entry.get('entry_id')}`",
                "",
                f"- Timestep: `{entry.get('timestep')}`",
                f"- Status: `{entry.get('status')}`",
                f"- Reward: `{entry.get('reward')}`",
                f"- Raw score: `{entry.get('raw_score')}`",
                f"- Guidance format ok: `{entry.get('guidance_format_ok')}`",
                f"- Execution model: `{entry.get('execution_model')}`",
                f"- Execution finish reason: `{entry.get('execution_finish_reason')}`",
                "",
            ]
        )
        _append_prompt_answer_section(lines, "Guidance Prompt System", _prompt_part(entry, "guidance_prompt", "system"))
        _append_prompt_answer_section(lines, "Guidance Prompt User", _prompt_part(entry, "guidance_prompt", "user"))
        _append_prompt_answer_section(lines, "Guidance Answer", entry.get("guidance"))
        _append_prompt_answer_section(lines, "Raw Guidance Answer", entry.get("raw_guidance_text"))
        _append_prompt_answer_section(lines, "Raw Guidance Answer With Specials", entry.get("raw_guidance_with_specials"))
        _append_prompt_answer_section(lines, "Execution Prompt System", _prompt_part(entry, "execution_prompt", "system"))
        _append_prompt_answer_section(lines, "Execution Prompt User", _prompt_part(entry, "execution_prompt", "user"))
        _append_prompt_answer_section(lines, "Execution Answer", entry.get("execution_output"))
        _append_prompt_answer_section(lines, "Raw Model Summary", entry.get("raw_model_summary"))
        _append_prompt_answer_section(lines, "Canonical Library Summary", entry.get("canonical_summary"))
        lines.extend(
            [
                "### Verification Artifacts",
                "",
                "```json",
                json.dumps(entry.get("verification_artifacts") or {}, indent=2, sort_keys=True),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def _prompt_part(entry: dict[str, object], prompt_key: str, part_key: str) -> object:
    prompt = entry.get(prompt_key) or {}
    if not isinstance(prompt, dict):
        return ""
    return prompt.get(part_key, "")


def _append_prompt_answer_section(lines: list[str], title: str, value: object) -> None:
    lines.extend([f"### {title}", "", "```text", str(value or ""), "```", ""])


def _entry_single_summary_rank(entry: dict[str, object]) -> tuple[int, int, int, int, float, str]:
    metadata = entry.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    raw_score = entry.get("verifier_raw_score")
    reward = entry.get("verifier_reward")
    numeric_score = raw_score if raw_score is not None else reward
    try:
        score = float(numeric_score)
    except (TypeError, ValueError):
        score = float("-inf")
    return (
        1 if str(entry.get("verifier_status")) == "valid" else 0,
        1 if bool(metadata.get("guidance_format_ok")) and str(entry.get("guidance") or "").strip() else 0,
        1 if str(entry.get("summary") or "").strip() else 0,
        1 if str(metadata.get("execution_text") or "").strip() else 0,
        score,
        str(entry.get("id") or ""),
    )


def _prune_to_single_summary(output_dir: str = REMOTE_SINGLE_SUMMARY_OUTPUT_DIR) -> dict[str, object]:
    library_path = Path(output_dir) / "library.json"
    if not library_path.exists():
        return {"output_dir": output_dir, "library_exists": False}
    data = json.loads(library_path.read_text())
    entries = data.get("entries") or {}
    if not isinstance(entries, dict) or len(entries) <= 1:
        return {"output_dir": output_dir, "library_exists": True, "entry_count": len(entries)}

    chosen_id, chosen_entry = max(
        ((entry_id, entry) for entry_id, entry in entries.items() if isinstance(entry, dict)),
        key=lambda item: _entry_single_summary_rank(item[1]),
    )
    nodes = data.get("nodes") or {}
    chosen_node_id = None
    for node_id, node in nodes.items():
        if isinstance(node, dict) and node.get("entry_id") == chosen_id:
            chosen_node_id = node_id
            break

    keep_node_ids: set[str] = set()
    current_node_id = chosen_node_id
    while current_node_id and current_node_id in nodes:
        keep_node_ids.add(current_node_id)
        parent_id = nodes[current_node_id].get("parent_id") if isinstance(nodes[current_node_id], dict) else None
        current_node_id = str(parent_id) if parent_id else None
    for node_id, node in nodes.items():
        if isinstance(node, dict) and node.get("entry_id") is None:
            keep_node_ids.add(node_id)

    pruned_nodes = {
        node_id: node
        for node_id, node in nodes.items()
        if node_id in keep_node_ids and isinstance(node, dict)
    }
    for node_id, node in pruned_nodes.items():
        children = node.get("children") or []
        node["children"] = [child for child in children if child in pruned_nodes]
        if node.get("entry_id") not in {None, chosen_id}:
            node["entry_id"] = None
    if chosen_node_id:
        data["best_node_id"] = chosen_node_id

    groups = data.get("groups") or {}
    if isinstance(groups, dict):
        for group in groups.values():
            if not isinstance(group, dict):
                continue
            group["children"] = [child for child in (group.get("children") or []) if child == chosen_node_id]
            group["submitted"] = 1 if group["children"] else 0
            group["finalized"] = bool(group["children"])

    if isinstance(data.get("config"), dict):
        data["config"]["rollout_n"] = 1
    data["rollout_n"] = 1
    data["entries"] = {chosen_id: chosen_entry}
    data["nodes"] = pruned_nodes
    data.setdefault("metadata", {})
    if isinstance(data["metadata"], dict):
        data["metadata"]["single_summary_pruned_from_entries"] = len(entries)
        data["metadata"]["single_summary_chosen_entry_id"] = chosen_id
    library_path.write_text(json.dumps(data, indent=2, sort_keys=True))
    return {
        "output_dir": output_dir,
        "library_exists": True,
        "entry_count": 1,
        "chosen_entry_id": chosen_id,
        "chosen_node_id": chosen_node_id,
        "pruned_from_entry_count": len(entries),
    }


def _validate_history_extraction(summary: dict[str, object]) -> None:
    entry_count = int(summary.get("entry_count", 0))
    if entry_count <= 0:
        raise RuntimeError(f"History smoke did not write library entries: {summary}")
    for key in ("guidance_ok_count", "raw_summary_ok_count", "canonical_summary_ok_count", "execution_text_ok_count"):
        if int(summary.get(key, 0)) <= 0:
            raise RuntimeError(f"History smoke missing {key}: {summary}")


def _validate_single_summary_extraction(summary: dict[str, object]) -> None:
    if int(summary.get("entry_count", 0)) != 1:
        raise RuntimeError(f"Single-summary smoke should write exactly one library entry: {summary}")
    for key in ("guidance_ok_count", "canonical_summary_ok_count", "execution_text_ok_count"):
        if int(summary.get(key, 0)) != 1:
            raise RuntimeError(f"Single-summary smoke missing {key}: {summary}")
    statuses = summary.get("statuses") or {}
    if not isinstance(statuses, dict) or int(statuses.get("valid", 0)) != 1:
        raise RuntimeError(f"Single-summary smoke should keep exactly one valid entry: {summary}")


def _build_frontier_runtime(image: modal.Image) -> modal.Image:
    return (
        image.apt_install(
            "ca-certificates",
            "curl",
            "git",
            "jq",
            "unzip",
            "zip",
            "build-essential",
            "pkg-config",
            "python3-dev",
            "pypy3",
            "openjdk-17-jdk",
        )
        .run_commands(
            "curl -fsSL https://deb.nodesource.com/setup_20.x | bash -",
            "apt-get install -y --no-install-recommends nodejs",
            "npm install -g npm@latest",
        )
        .run_commands(
            "set -eux; "
            "arch=\"$(uname -m)\"; "
            "case \"$arch\" in x86_64) goarch=amd64 ;; aarch64) goarch=arm64 ;; *) exit 1 ;; esac; "
            "url=\"https://github.com/criyle/go-judge/releases/download/v1.11.1/go-judge_linux_${goarch}.tar.gz\"; "
            "if ! curl -fsSL \"$url\" | tar -xz -C /usr/local/bin go-judge; then "
            "  url=$(curl -fsSL https://api.github.com/repos/criyle/go-judge/releases/latest "
            "    | jq -r --arg goarch \"$goarch\" '.assets[] | select(.name | test(\"linux.*\\($goarch).*tar.gz$\")) | .browser_download_url' "
            "    | head -n 1); "
            "  test -n \"$url\"; "
            "  curl -fsSL \"$url\" | tar -xz -C /usr/local/bin go-judge; "
            "fi; "
            "chmod +x /usr/local/bin/go-judge"
        )
        .run_commands(
            f"git clone --depth 1 --filter=blob:none --sparse https://github.com/FrontierCS/Frontier-CS.git {REMOTE_FRONTIER_DIR}",
            (
                f"cd {REMOTE_FRONTIER_DIR} && "
                "git sparse-checkout set "
                "pyproject.toml setup.py README.md src "
                "algorithmic/package.json algorithmic/package-lock.json "
                "algorithmic/server.js algorithmic/entrypoint.sh algorithmic/judge "
                "algorithmic/problems/0 algorithmic/README.md"
            ),
            (
                f"mkdir -p /app /app/problems /app/submissions /app/data /lib/testlib && "
                f"cp {REMOTE_FRONTIER_DIR}/algorithmic/package.json /app/ && "
                f"cp {REMOTE_FRONTIER_DIR}/algorithmic/package-lock.json /app/ && "
                f"cp {REMOTE_FRONTIER_DIR}/algorithmic/server.js /app/ && "
                f"cp {REMOTE_FRONTIER_DIR}/algorithmic/entrypoint.sh /app/ && "
                f"cp -r {REMOTE_FRONTIER_DIR}/algorithmic/judge/src /app/src && "
                f"cp -r {REMOTE_FRONTIER_DIR}/algorithmic/judge/include /app/include && "
                f"cp -r {REMOTE_FRONTIER_DIR}/algorithmic/judge/config /app/config && "
                f"cp -r {REMOTE_FRONTIER_DIR}/algorithmic/judge/include/. /lib/testlib/ && "
                f"cp -r {REMOTE_FRONTIER_DIR}/algorithmic/problems/0 /app/problems/0 && "
                "chmod +x /app/entrypoint.sh && sed -i 's/\\r$//' /app/entrypoint.sh"
            ),
            "cd /app && npm install --omit=dev --ignore-scripts",
        )
        .uv_pip_install("requests", "git+https://github.com/FrontierCS/Frontier-CS.git")
        .add_local_file(REPO_ROOT / "scripts" / "modal_local_gojudge.js", "/app/src/gojudge.js", copy=True)
    )


frontier_image = _build_frontier_runtime(
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12").entrypoint([])
)

train_image = (
    frontier_image.uv_pip_install(
        "vllm>=0.8.5,<=0.12.0",
        "accelerate",
        "codetiming",
        "datasets",
        "dill",
        "hydra-core",
        "numpy<2.0.0",
        "pandas",
        "peft",
        "pyarrow>=19.0.0",
        "pybind11",
        "pylatexenc",
        "ray[default]>=2.41.0",
        "torchdata",
        "tensordict>=0.8.0,<=0.10.0,!=0.9.0",
        "transformers>=4.55.0,<5.0.0",
        "wandb",
        "packaging>=20.0",
        "tensorboard",
        "sentencepiece",
        "tiktoken",
        "protobuf",
        "hf_transfer",
        "huggingface_hub[cli]",
        "math-verify",
        "latex2sympy2_extended",
        "fastapi",
        "uvicorn",
        "liger-kernel",
    )
    .uv_pip_install(FLASH_ATTN_TORCH29_CU12_WHEEL)
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "TORCH_CUDA_ARCH_LIST": "9.0 9.0a",
            "VLLM_NO_USAGE_STATS": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": REMOTE_REPO_DIR,
            "RAY_raylet_start_wait_time_s": "300",
        }
    )
    .add_local_dir(REPO_ROOT, REMOTE_REPO_DIR, ignore=_repo_ignore)
)


@app.function(image=frontier_image, timeout=30 * 60, cpu=16, memory=32768)
def verifier_smoke() -> dict[str, object]:
    judge = _start_judge()
    try:
        from frontier_cs import SingleEvaluator

        baseline = r"""
#include <bits/stdc++.h>
using namespace std;
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    if (!(cin >> n)) return 0;
    vector<array<int, 4>> ans(n);
    int cursor = 0, max_h = 1;
    for (int i = 0; i < n; ++i) {
        int k;
        cin >> k;
        int minx = INT_MAX, maxx = INT_MIN, miny = INT_MAX, maxy = INT_MIN;
        for (int j = 0; j < k; ++j) {
            int x, y;
            cin >> x >> y;
            minx = min(minx, x);
            maxx = max(maxx, x);
            miny = min(miny, y);
            maxy = max(maxy, y);
        }
        int w = maxx - minx + 1;
        int h = maxy - miny + 1;
        ans[i] = {cursor - minx, -miny, 0, 0};
        cursor += w;
        max_h = max(max_h, h);
    }
    cout << max(1, cursor) << ' ' << max_h << '\n';
    for (auto &a : ans) cout << a[0] << ' ' << a[1] << ' ' << a[2] << ' ' << a[3] << '\n';
    return 0;
}
"""
        evaluator = SingleEvaluator(
            register_cleanup=False,
            base_dir=Path(REMOTE_FRONTIER_DIR),
            judge_url="http://127.0.0.1:8081",
        )
        result = evaluator.evaluate("algorithmic", problem_id="0", code=baseline)
        payload = {
            "success": bool(getattr(result, "success", False)),
            "score": getattr(result, "score", None),
            "message": str(getattr(result, "message", "")),
        }
        print(json.dumps(payload, indent=2), flush=True)
        if not payload["success"]:
            _print_judge_log()
            raise RuntimeError(f"FrontierCS verifier smoke failed: {payload}")
        return payload
    finally:
        _stop_judge(judge)


@app.function(image=train_image, timeout=30 * 60, cpu=8, memory=32768)
def check_training_packages_smoke() -> dict[str, object]:
    _assert_training_packages_available()
    return {"ok": True}


@app.function(
    image=train_image,
    gpu=GPU_CONFIG,
    timeout=12 * 60 * 60,
    cpu=64,
    memory=262144,
    volumes={"/runs": runs_volume, "/cache": cache_volume},
)
def train_smoke() -> dict[str, object]:
    _run_streaming(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    _reset_output_dir()
    judge = _start_judge()
    try:
        _run_streaming(training_command(), cwd=REMOTE_REPO_DIR)
        summary = _summarize_output()
        print(json.dumps(summary, indent=2), flush=True)
        if int(summary.get("entry_count", 0)) <= 0:
            raise RuntimeError(f"Training smoke did not write any library entries: {summary}")
        return summary
    finally:
        runs_volume.commit()
        cache_volume.commit()
        _stop_judge(judge)


@app.function(
    image=train_image,
    gpu=HISTORY_GPU_CONFIG,
    timeout=12 * 60 * 60,
    cpu=48,
    memory=196608,
    volumes={"/runs": runs_volume, "/cache": cache_volume},
)
def train_history_smoke() -> dict[str, object]:
    _run_streaming(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    _reset_output_dir(REMOTE_HISTORY_OUTPUT_DIR)
    judge = _start_judge()
    try:
        _run_streaming(history_training_command(), cwd=REMOTE_REPO_DIR)
        summary = _summarize_output(REMOTE_HISTORY_OUTPUT_DIR)
        history_summary = _summarize_history_extraction(REMOTE_HISTORY_OUTPUT_DIR)
        merged_summary = {**summary, "history_extraction": history_summary}
        print(json.dumps(merged_summary, indent=2), flush=True)
        _validate_history_extraction(history_summary)
        return merged_summary
    finally:
        runs_volume.commit()
        cache_volume.commit()
        _stop_judge(judge)


@app.function(
    image=train_image,
    gpu=HISTORY_GPU_CONFIG,
    timeout=12 * 60 * 60,
    cpu=48,
    memory=196608,
    volumes={"/runs": runs_volume, "/cache": cache_volume},
)
def train_single_summary_smoke() -> dict[str, object]:
    _run_streaming(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    judge = _start_judge()
    try:
        _run_streaming(single_summary_training_command(), cwd=REMOTE_REPO_DIR)
        summary = _summarize_output(REMOTE_SINGLE_SUMMARY_OUTPUT_DIR)
        history_summary = _summarize_history_extraction(REMOTE_SINGLE_SUMMARY_OUTPUT_DIR)
        dump_summary = _dump_prompt_answer_artifacts(REMOTE_SINGLE_SUMMARY_OUTPUT_DIR)
        merged_summary = {**summary, "history_extraction": history_summary, "prompt_answer_dump": dump_summary}
        print(json.dumps(merged_summary, indent=2), flush=True)
        if int(history_summary.get("entry_count", 0)) <= 0:
            raise RuntimeError(f"Single-summary smoke did not write library entries: {history_summary}")
        return merged_summary
    finally:
        runs_volume.commit()
        cache_volume.commit()
        _stop_judge(judge)


@app.function(
    image=train_image,
    gpu=HISTORY_GPU_CONFIG,
    timeout=12 * 60 * 60,
    cpu=48,
    memory=196608,
    volumes={"/runs": runs_volume, "/cache": cache_volume},
)
def bootstrap_single_summary_smoke() -> dict[str, object]:
    _run_streaming(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    _reset_output_dir(REMOTE_SINGLE_SUMMARY_OUTPUT_DIR)
    judge = _start_judge()
    try:
        _run_streaming(single_summary_bootstrap_command(), cwd=REMOTE_REPO_DIR)
        summary = _summarize_output(REMOTE_SINGLE_SUMMARY_OUTPUT_DIR)
        history_summary = _summarize_history_extraction(REMOTE_SINGLE_SUMMARY_OUTPUT_DIR)
        merged_summary = {**summary, "history_extraction": history_summary}
        print(json.dumps(merged_summary, indent=2), flush=True)
        if int(history_summary.get("entry_count", 0)) <= 0:
            raise RuntimeError(f"Bootstrap smoke did not write any library entries: {merged_summary}")
        if int(history_summary.get("raw_summary_ok_count", 0)) <= 0:
            raise RuntimeError(f"Bootstrap smoke did not write a raw model summary: {merged_summary}")
        return merged_summary
    finally:
        runs_volume.commit()
        cache_volume.commit()
        _stop_judge(judge)


@app.function(
    image=train_image,
    gpu=HISTORY_GPU_CONFIG,
    timeout=12 * 60 * 60,
    cpu=48,
    memory=196608,
    volumes={"/runs": runs_volume, "/cache": cache_volume},
)
def train_qwen_exec_single_summary_smoke() -> dict[str, object]:
    _run_streaming(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    judge = _start_judge()
    try:
        _run_streaming(qwen_exec_single_summary_training_command(), cwd=REMOTE_REPO_DIR)
        summary = _summarize_output(REMOTE_QWEN_EXEC_SINGLE_SUMMARY_OUTPUT_DIR)
        history_summary = _summarize_history_extraction(REMOTE_QWEN_EXEC_SINGLE_SUMMARY_OUTPUT_DIR)
        dump_summary = _dump_prompt_answer_artifacts(REMOTE_QWEN_EXEC_SINGLE_SUMMARY_OUTPUT_DIR)
        merged_summary = {**summary, "history_extraction": history_summary, "prompt_answer_dump": dump_summary}
        print(json.dumps(merged_summary, indent=2), flush=True)
        if int(history_summary.get("entry_count", 0)) <= 0:
            raise RuntimeError(f"Qwen execution single-summary smoke did not write library entries: {history_summary}")
        return merged_summary
    finally:
        runs_volume.commit()
        cache_volume.commit()
        _stop_judge(judge)


@app.function(
    image=train_image,
    gpu=HISTORY_GPU_CONFIG,
    timeout=12 * 60 * 60,
    cpu=48,
    memory=196608,
    volumes={"/runs": runs_volume, "/cache": cache_volume},
)
def bootstrap_qwen_exec_single_summary_smoke() -> dict[str, object]:
    _run_streaming(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    _reset_output_dir(REMOTE_QWEN_EXEC_SINGLE_SUMMARY_OUTPUT_DIR)
    judge = _start_judge()
    try:
        _run_streaming(qwen_exec_single_summary_bootstrap_command(), cwd=REMOTE_REPO_DIR)
        summary = _summarize_output(REMOTE_QWEN_EXEC_SINGLE_SUMMARY_OUTPUT_DIR)
        history_summary = _summarize_history_extraction(REMOTE_QWEN_EXEC_SINGLE_SUMMARY_OUTPUT_DIR)
        merged_summary = {**summary, "history_extraction": history_summary}
        print(json.dumps(merged_summary, indent=2), flush=True)
        if int(history_summary.get("entry_count", 0)) <= 0:
            raise RuntimeError(f"Qwen execution bootstrap smoke did not write any library entries: {merged_summary}")
        if int(history_summary.get("raw_summary_ok_count", 0)) <= 0:
            raise RuntimeError(f"Qwen execution bootstrap smoke did not write a raw model summary: {merged_summary}")
        return merged_summary
    finally:
        runs_volume.commit()
        cache_volume.commit()
        _stop_judge(judge)


@app.function(
    image=train_image,
    gpu=HISTORY_GPU_CONFIG,
    timeout=12 * 60 * 60,
    cpu=48,
    memory=196608,
    volumes={"/runs": runs_volume, "/cache": cache_volume},
    secrets=[openrouter_secret],
)
def train_openrouter_gpt55_single_summary_smoke() -> dict[str, object]:
    _run_streaming(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    judge = _start_judge()
    try:
        _run_streaming(openrouter_gpt55_single_summary_training_command(), cwd=REMOTE_REPO_DIR)
        summary = _summarize_output(REMOTE_OPENROUTER_GPT55_SINGLE_SUMMARY_OUTPUT_DIR)
        history_summary = _summarize_history_extraction(REMOTE_OPENROUTER_GPT55_SINGLE_SUMMARY_OUTPUT_DIR)
        dump_summary = _dump_prompt_answer_artifacts(REMOTE_OPENROUTER_GPT55_SINGLE_SUMMARY_OUTPUT_DIR)
        merged_summary = {**summary, "history_extraction": history_summary, "prompt_answer_dump": dump_summary}
        print(json.dumps(merged_summary, indent=2), flush=True)
        if int(history_summary.get("entry_count", 0)) <= 0:
            raise RuntimeError(f"OpenRouter GPT-5.5 single-summary smoke did not write library entries: {history_summary}")
        return merged_summary
    finally:
        runs_volume.commit()
        cache_volume.commit()
        _stop_judge(judge)


@app.function(
    image=train_image,
    gpu=HISTORY_GPU_CONFIG,
    timeout=12 * 60 * 60,
    cpu=48,
    memory=196608,
    volumes={"/runs": runs_volume, "/cache": cache_volume},
    secrets=[openrouter_secret],
)
def train_openrouter_gpt55_seeded_single_summary_smoke() -> dict[str, object]:
    _run_streaming(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    _reset_output_dir(REMOTE_OPENROUTER_GPT55_SINGLE_SUMMARY_OUTPUT_DIR)
    judge = _start_judge()
    try:
        _run_streaming(openrouter_gpt55_single_summary_training_command(), cwd=REMOTE_REPO_DIR)
        summary = _summarize_output(REMOTE_OPENROUTER_GPT55_SINGLE_SUMMARY_OUTPUT_DIR)
        history_summary = _summarize_history_extraction(REMOTE_OPENROUTER_GPT55_SINGLE_SUMMARY_OUTPUT_DIR)
        dump_summary = _dump_prompt_answer_artifacts(REMOTE_OPENROUTER_GPT55_SINGLE_SUMMARY_OUTPUT_DIR)
        merged_summary = {**summary, "history_extraction": history_summary, "prompt_answer_dump": dump_summary}
        print(json.dumps(merged_summary, indent=2), flush=True)
        if int(history_summary.get("entry_count", 0)) <= 0:
            raise RuntimeError(f"OpenRouter GPT-5.5 seeded smoke did not write library entries: {history_summary}")
        return merged_summary
    finally:
        runs_volume.commit()
        cache_volume.commit()
        _stop_judge(judge)


@app.function(
    image=train_image,
    gpu=GPU_CONFIG,
    timeout=12 * 60 * 60,
    cpu=64,
    memory=262144,
    volumes={"/runs": runs_volume, "/cache": cache_volume},
    secrets=[openrouter_secret],
)
def train_openrouter_gpt55_4gpu_full_batch_smoke() -> dict[str, object]:
    _run_streaming(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    _reset_output_dir(REMOTE_OPENROUTER_GPT55_4GPU_FULL_BATCH_OUTPUT_DIR)
    judge = _start_judge()
    try:
        _run_streaming(openrouter_gpt55_4gpu_full_batch_training_command(), cwd=REMOTE_REPO_DIR)
        summary = _summarize_output(REMOTE_OPENROUTER_GPT55_4GPU_FULL_BATCH_OUTPUT_DIR)
        history_summary = _summarize_history_extraction(REMOTE_OPENROUTER_GPT55_4GPU_FULL_BATCH_OUTPUT_DIR)
        dump_summary = _dump_prompt_answer_artifacts(REMOTE_OPENROUTER_GPT55_4GPU_FULL_BATCH_OUTPUT_DIR)
        merged_summary = {**summary, "history_extraction": history_summary, "prompt_answer_dump": dump_summary}
        print(json.dumps(merged_summary, indent=2), flush=True)
        if int(history_summary.get("entry_count", 0)) <= 0:
            raise RuntimeError(f"OpenRouter GPT-5.5 4GPU full-batch smoke did not write entries: {history_summary}")
        return merged_summary
    finally:
        runs_volume.commit()
        cache_volume.commit()
        _stop_judge(judge)


@app.function(
    image=train_image,
    gpu=GPU_CONFIG,
    timeout=12 * 60 * 60,
    cpu=64,
    memory=262144,
    volumes={"/runs": runs_volume, "/cache": cache_volume},
    secrets=[evolvent_secret],
)
def train_evolvent_gpt55_4gpu_short_response_smoke() -> dict[str, object]:
    _run_streaming(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    _reset_output_dir(REMOTE_EVOLVENT_GPT55_4GPU_SHORT_RESPONSE_OUTPUT_DIR)
    judge = _start_judge()
    try:
        _run_streaming(evolvent_gpt55_4gpu_short_response_training_command(), cwd=REMOTE_REPO_DIR)
        summary = _summarize_output(REMOTE_EVOLVENT_GPT55_4GPU_SHORT_RESPONSE_OUTPUT_DIR)
        history_summary = _summarize_history_extraction(REMOTE_EVOLVENT_GPT55_4GPU_SHORT_RESPONSE_OUTPUT_DIR)
        dump_summary = _dump_prompt_answer_artifacts(REMOTE_EVOLVENT_GPT55_4GPU_SHORT_RESPONSE_OUTPUT_DIR)
        merged_summary = {**summary, "history_extraction": history_summary, "prompt_answer_dump": dump_summary}
        print(json.dumps(merged_summary, indent=2), flush=True)
        if int(history_summary.get("entry_count", 0)) <= 0:
            raise RuntimeError(f"Evolvent GPT-5.5 4GPU short-response smoke did not write entries: {history_summary}")
        return merged_summary
    finally:
        runs_volume.commit()
        cache_volume.commit()
        _stop_judge(judge)


@app.function(
    image=train_image,
    gpu=GPT_OSS_120B_GPU_CONFIG,
    timeout=24 * 60 * 60,
    cpu=64,
    memory=327680,
    volumes={"/runs": runs_volume, "/cache": cache_volume},
)
def train_gpt_oss_120b_5gpu_group64_smoke() -> dict[str, object]:
    _assert_training_packages_available()
    _run_streaming(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    _reset_output_dir(REMOTE_GPT_OSS_120B_5GPU_GROUP64_OUTPUT_DIR)
    execution_server = _start_gpt_oss_120b_server()
    judge = _start_judge()
    try:
        training_env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0,1,2,3"}
        _run_streaming(gpt_oss_120b_5gpu_group64_training_command(), cwd=REMOTE_REPO_DIR, env=training_env)
        summary = _summarize_output(REMOTE_GPT_OSS_120B_5GPU_GROUP64_OUTPUT_DIR)
        history_summary = _summarize_history_extraction(REMOTE_GPT_OSS_120B_5GPU_GROUP64_OUTPUT_DIR)
        dump_summary = _dump_prompt_answer_artifacts(REMOTE_GPT_OSS_120B_5GPU_GROUP64_OUTPUT_DIR)
        merged_summary = {**summary, "history_extraction": history_summary, "prompt_answer_dump": dump_summary}
        print(json.dumps(merged_summary, indent=2), flush=True)
        if int(history_summary.get("entry_count", 0)) <= 0:
            raise RuntimeError(f"GPT-OSS-120B 5GPU group64 smoke did not write entries: {history_summary}")
        return merged_summary
    finally:
        runs_volume.commit()
        cache_volume.commit()
        _stop_judge(judge)
        _stop_process(execution_server)


@app.function(
    image=train_image,
    gpu=GPT_OSS_120B_3GPU_CONFIG,
    timeout=24 * 60 * 60,
    cpu=64,
    memory=262144,
    volumes={"/runs": runs_volume, "/cache": cache_volume},
)
def train_gpt_oss_120b_3gpu_group16_smoke() -> dict[str, object]:
    _assert_training_packages_available()
    _run_streaming(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    _reset_output_dir(REMOTE_GPT_OSS_120B_3GPU_GROUP16_OUTPUT_DIR)
    execution_server = _start_gpt_oss_120b_server(cuda_visible_devices="2")
    judge = _start_judge()
    try:
        training_env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0,1"}
        _run_streaming(gpt_oss_120b_3gpu_group16_training_command(), cwd=REMOTE_REPO_DIR, env=training_env)
        summary = _summarize_output(REMOTE_GPT_OSS_120B_3GPU_GROUP16_OUTPUT_DIR)
        history_summary = _summarize_history_extraction(REMOTE_GPT_OSS_120B_3GPU_GROUP16_OUTPUT_DIR)
        dump_summary = _dump_prompt_answer_artifacts(REMOTE_GPT_OSS_120B_3GPU_GROUP16_OUTPUT_DIR)
        merged_summary = {**summary, "history_extraction": history_summary, "prompt_answer_dump": dump_summary}
        print(json.dumps(merged_summary, indent=2), flush=True)
        if int(history_summary.get("entry_count", 0)) <= 0:
            raise RuntimeError(f"GPT-OSS-120B 3GPU group16 smoke did not write entries: {history_summary}")
        return merged_summary
    finally:
        runs_volume.commit()
        cache_volume.commit()
        _stop_judge(judge)
        _stop_process(execution_server)


@app.function(
    image=train_image,
    gpu=GPT_OSS_120B_3GPU_CONFIG,
    timeout=24 * 60 * 60,
    cpu=64,
    memory=262144,
    volumes={"/runs": runs_volume, "/cache": cache_volume},
)
def train_gpt_oss_120b_3gpu_group16_h200_tuned_smoke() -> dict[str, object]:
    _assert_training_packages_available()
    _run_streaming(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    _reset_output_dir(REMOTE_GPT_OSS_120B_3GPU_GROUP16_H200_TUNED_OUTPUT_DIR)
    execution_server = _start_gpt_oss_120b_server(cuda_visible_devices="2")
    judge = _start_judge()
    try:
        training_env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0,1"}
        _run_streaming(
            gpt_oss_120b_3gpu_group16_h200_tuned_training_command(),
            cwd=REMOTE_REPO_DIR,
            env=training_env,
        )
        summary = _summarize_output(REMOTE_GPT_OSS_120B_3GPU_GROUP16_H200_TUNED_OUTPUT_DIR)
        history_summary = _summarize_history_extraction(
            REMOTE_GPT_OSS_120B_3GPU_GROUP16_H200_TUNED_OUTPUT_DIR
        )
        dump_summary = _dump_prompt_answer_artifacts(REMOTE_GPT_OSS_120B_3GPU_GROUP16_H200_TUNED_OUTPUT_DIR)
        merged_summary = {**summary, "history_extraction": history_summary, "prompt_answer_dump": dump_summary}
        print(json.dumps(merged_summary, indent=2), flush=True)
        if int(history_summary.get("entry_count", 0)) <= 0:
            raise RuntimeError(f"GPT-OSS-120B 3GPU group16 H200-tuned smoke did not write entries: {history_summary}")
        return merged_summary
    finally:
        runs_volume.commit()
        cache_volume.commit()
        _stop_judge(judge)
        _stop_process(execution_server)


@app.function(
    image=train_image,
    gpu=GPT_OSS_120B_3GPU_CONFIG,
    timeout=24 * 60 * 60,
    cpu=64,
    memory=262144,
    volumes={"/runs": runs_volume, "/cache": cache_volume},
)
def train_gpt_oss_120b_3gpu_batch8_group8_temp09_smoke() -> dict[str, object]:
    _assert_training_packages_available()
    _run_streaming(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    _reset_output_dir(REMOTE_GPT_OSS_120B_3GPU_BATCH8_GROUP8_TEMP09_OUTPUT_DIR)
    execution_server = _start_gpt_oss_120b_server(cuda_visible_devices="2")
    judge = _start_judge()
    try:
        training_env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0,1"}
        _run_streaming(
            gpt_oss_120b_3gpu_batch8_group8_temp09_training_command(),
            cwd=REMOTE_REPO_DIR,
            env=training_env,
        )
        summary = _summarize_output(REMOTE_GPT_OSS_120B_3GPU_BATCH8_GROUP8_TEMP09_OUTPUT_DIR)
        history_summary = _summarize_history_extraction(
            REMOTE_GPT_OSS_120B_3GPU_BATCH8_GROUP8_TEMP09_OUTPUT_DIR
        )
        dump_summary = _dump_prompt_answer_artifacts(REMOTE_GPT_OSS_120B_3GPU_BATCH8_GROUP8_TEMP09_OUTPUT_DIR)
        merged_summary = {**summary, "history_extraction": history_summary, "prompt_answer_dump": dump_summary}
        print(json.dumps(merged_summary, indent=2), flush=True)
        if int(history_summary.get("entry_count", 0)) <= 0:
            raise RuntimeError(f"GPT-OSS-120B 3GPU batch8 group8 temp09 run did not write entries: {history_summary}")
        return merged_summary
    finally:
        runs_volume.commit()
        cache_volume.commit()
        _stop_judge(judge)
        _stop_process(execution_server)


@app.function(
    image=train_image,
    gpu=GPT_OSS_120B_3GPU_CONFIG,
    timeout=24 * 60 * 60,
    cpu=64,
    memory=262144,
    volumes={"/runs": runs_volume, "/cache": cache_volume},
)
def train_gpt_oss_120b_3gpu_batch8_group16_concurrency16_smoke() -> dict[str, object]:
    _assert_training_packages_available()
    _run_streaming(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    _reset_output_dir(REMOTE_GPT_OSS_120B_3GPU_BATCH8_GROUP16_CONCURRENCY16_OUTPUT_DIR)
    execution_server = _start_gpt_oss_120b_server(cuda_visible_devices="2", max_num_seqs=16)
    judge = _start_judge(workers=16)
    try:
        training_env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0,1"}
        _run_streaming(
            gpt_oss_120b_3gpu_batch8_group16_concurrency16_training_command(),
            cwd=REMOTE_REPO_DIR,
            env=training_env,
        )
        summary = _summarize_output(REMOTE_GPT_OSS_120B_3GPU_BATCH8_GROUP16_CONCURRENCY16_OUTPUT_DIR)
        history_summary = _summarize_history_extraction(
            REMOTE_GPT_OSS_120B_3GPU_BATCH8_GROUP16_CONCURRENCY16_OUTPUT_DIR
        )
        dump_summary = _dump_prompt_answer_artifacts(
            REMOTE_GPT_OSS_120B_3GPU_BATCH8_GROUP16_CONCURRENCY16_OUTPUT_DIR
        )
        merged_summary = {**summary, "history_extraction": history_summary, "prompt_answer_dump": dump_summary}
        print(json.dumps(merged_summary, indent=2), flush=True)
        if int(history_summary.get("entry_count", 0)) <= 0:
            raise RuntimeError(
                f"GPT-OSS-120B 3GPU batch8 group16 concurrency16 run did not write entries: {history_summary}"
            )
        return merged_summary
    finally:
        runs_volume.commit()
        cache_volume.commit()
        _stop_judge(judge)
        _stop_process(execution_server)


@app.function(
    image=train_image,
    gpu=GPT_OSS_120B_3GPU_CONFIG,
    timeout=24 * 60 * 60,
    cpu=64,
    memory=262144,
    volumes={"/runs": runs_volume, "/cache": cache_volume},
)
def train_gpt_oss_120b_3gpu_batch8_group16_puct_fix_smoke() -> dict[str, object]:
    _assert_training_packages_available()
    _run_streaming(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    output_dir = REMOTE_GPT_OSS_120B_3GPU_BATCH8_GROUP16_PUCT_FIX_OUTPUT_DIR
    _reset_output_dir(output_dir)
    execution_server = _start_gpt_oss_120b_server(cuda_visible_devices="2", max_num_seqs=16)
    judge = _start_judge(workers=16)
    try:
        training_env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0,1"}
        _run_streaming(
            gpt_oss_120b_3gpu_batch8_group16_puct_fix_training_command(),
            cwd=REMOTE_REPO_DIR,
            env=training_env,
        )
        summary = _summarize_output(output_dir)
        history_summary = _summarize_history_extraction(output_dir)
        dump_summary = _dump_prompt_answer_artifacts(output_dir)
        puct_summary = _summarize_puct_group_accounting(output_dir)
        merged_summary = {
            **summary,
            "history_extraction": history_summary,
            "prompt_answer_dump": dump_summary,
            "puct_accounting": puct_summary,
        }
        print(json.dumps(merged_summary, indent=2), flush=True)
        if int(history_summary.get("entry_count", 0)) <= 0:
            raise RuntimeError(f"PUCT-fix acceptance run did not write entries: {history_summary}")
        _validate_puct_group_accounting(
            puct_summary,
            expected_groups=8,
            expected_rollout_n=16,
        )
        return merged_summary
    finally:
        runs_volume.commit()
        cache_volume.commit()
        _stop_judge(judge)
        _stop_process(execution_server)


@app.function(
    image=train_image,
    gpu=GPT_OSS_120B_3GPU_CONFIG,
    timeout=24 * 60 * 60,
    cpu=64,
    memory=262144,
    volumes={"/runs": runs_volume, "/cache": cache_volume},
)
def train_gpt_oss_120b_3gpu_batch8_group16_concurrency16_50step_smoke() -> dict[str, object]:
    _assert_training_packages_available()
    _run_streaming(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    _reset_output_dir(REMOTE_GPT_OSS_120B_3GPU_BATCH8_GROUP16_CONCURRENCY16_50STEP_OUTPUT_DIR)
    execution_server = _start_gpt_oss_120b_server(cuda_visible_devices="2", max_num_seqs=16)
    judge = _start_judge(workers=16)
    try:
        training_env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0,1"}
        _run_streaming(
            gpt_oss_120b_3gpu_batch8_group16_concurrency16_50step_training_command(),
            cwd=REMOTE_REPO_DIR,
            env=training_env,
        )
        summary = _summarize_output(REMOTE_GPT_OSS_120B_3GPU_BATCH8_GROUP16_CONCURRENCY16_50STEP_OUTPUT_DIR)
        history_summary = _summarize_history_extraction(
            REMOTE_GPT_OSS_120B_3GPU_BATCH8_GROUP16_CONCURRENCY16_50STEP_OUTPUT_DIR
        )
        dump_summary = _dump_prompt_answer_artifacts(
            REMOTE_GPT_OSS_120B_3GPU_BATCH8_GROUP16_CONCURRENCY16_50STEP_OUTPUT_DIR
        )
        merged_summary = {**summary, "history_extraction": history_summary, "prompt_answer_dump": dump_summary}
        print(json.dumps(merged_summary, indent=2), flush=True)
        if int(history_summary.get("entry_count", 0)) <= 0:
            raise RuntimeError(
                f"GPT-OSS-120B 3GPU batch8 group16 concurrency16 50-step run did not write entries: "
                f"{history_summary}"
            )
        return merged_summary
    finally:
        runs_volume.commit()
        cache_volume.commit()
        _stop_judge(judge)
        _stop_process(execution_server)


@app.function(
    image=train_image,
    gpu=GPT_OSS_120B_3GPU_CONFIG,
    timeout=24 * 60 * 60,
    cpu=64,
    memory=262144,
    volumes={"/runs": runs_volume, "/cache": cache_volume},
)
def train_gpt_oss_120b_3gpu_batch8_group16_prompt_refinement_50step_smoke() -> dict[str, object]:
    _assert_training_packages_available()
    _run_streaming(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    _reset_output_dir(REMOTE_GPT_OSS_120B_3GPU_BATCH8_GROUP16_PROMPT_REFINEMENT_50STEP_OUTPUT_DIR)
    execution_server = _start_gpt_oss_120b_server(cuda_visible_devices="2", max_num_seqs=16)
    judge = _start_judge(workers=16)
    try:
        training_env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0,1"}
        _run_streaming(
            gpt_oss_120b_3gpu_batch8_group16_prompt_refinement_50step_training_command(),
            cwd=REMOTE_REPO_DIR,
            env=training_env,
        )
        output_dir = REMOTE_GPT_OSS_120B_3GPU_BATCH8_GROUP16_PROMPT_REFINEMENT_50STEP_OUTPUT_DIR
        summary = _summarize_output(output_dir)
        history_summary = _summarize_history_extraction(output_dir)
        dump_summary = _dump_prompt_answer_artifacts(output_dir)
        merged_summary = {**summary, "history_extraction": history_summary, "prompt_answer_dump": dump_summary}
        print(json.dumps(merged_summary, indent=2), flush=True)
        if int(history_summary.get("entry_count", 0)) <= 0:
            raise RuntimeError(
                "GPT-OSS-120B 3GPU batch8 group16 prompt-refinement 50-step run "
                f"did not write entries: {history_summary}"
            )
        return merged_summary
    finally:
        runs_volume.commit()
        cache_volume.commit()
        _stop_judge(judge)
        _stop_process(execution_server)


@app.function(
    image=train_image,
    gpu=GPT_OSS_120B_SINGLE_GPU_CONFIG,
    timeout=4 * 60 * 60,
    cpu=32,
    memory=196608,
    volumes={"/runs": runs_volume, "/cache": cache_volume},
)
def bootstrap_gpt_oss_120b_seed_smoke() -> dict[str, object]:
    _assert_training_packages_available()
    _run_streaming(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    _reset_output_dir(REMOTE_GPT_OSS_120B_BOOTSTRAP_SEED_OUTPUT_DIR)
    execution_server = _start_gpt_oss_120b_server(cuda_visible_devices="0")
    judge = _start_judge()
    try:
        _run_streaming(gpt_oss_120b_bootstrap_seed_command(), cwd=REMOTE_REPO_DIR)
        summary = _summarize_output(REMOTE_GPT_OSS_120B_BOOTSTRAP_SEED_OUTPUT_DIR)
        history_summary = _summarize_history_extraction(REMOTE_GPT_OSS_120B_BOOTSTRAP_SEED_OUTPUT_DIR)
        dump_summary = _dump_prompt_answer_artifacts(REMOTE_GPT_OSS_120B_BOOTSTRAP_SEED_OUTPUT_DIR)
        merged_summary = {**summary, "history_extraction": history_summary, "prompt_answer_dump": dump_summary}
        print(json.dumps(merged_summary, indent=2), flush=True)
        if int(history_summary.get("entry_count", 0)) <= 0:
            raise RuntimeError(f"GPT-OSS-120B bootstrap seed did not write any library entries: {merged_summary}")
        if int(history_summary.get("raw_summary_ok_count", 0)) <= 0:
            raise RuntimeError(f"GPT-OSS-120B bootstrap seed did not write a raw model summary: {merged_summary}")
        statuses = set(summary.get("statuses") or [])
        if "valid" not in statuses:
            raise RuntimeError(f"GPT-OSS-120B bootstrap seed did not produce a valid candidate: {merged_summary}")
        return merged_summary
    finally:
        runs_volume.commit()
        cache_volume.commit()
        _stop_judge(judge)
        _stop_process(execution_server)


@app.function(
    image=train_image,
    gpu=HISTORY_GPU_CONFIG,
    timeout=12 * 60 * 60,
    cpu=48,
    memory=196608,
    volumes={"/runs": runs_volume, "/cache": cache_volume},
    secrets=[openrouter_secret],
)
def bootstrap_openrouter_gpt55_single_summary_smoke() -> dict[str, object]:
    _run_streaming(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    _reset_output_dir(REMOTE_OPENROUTER_GPT55_SINGLE_SUMMARY_OUTPUT_DIR)
    judge = _start_judge()
    try:
        _run_streaming(openrouter_gpt55_single_summary_bootstrap_command(), cwd=REMOTE_REPO_DIR)
        summary = _summarize_output(REMOTE_OPENROUTER_GPT55_SINGLE_SUMMARY_OUTPUT_DIR)
        history_summary = _summarize_history_extraction(REMOTE_OPENROUTER_GPT55_SINGLE_SUMMARY_OUTPUT_DIR)
        merged_summary = {**summary, "history_extraction": history_summary}
        print(json.dumps(merged_summary, indent=2), flush=True)
        if int(history_summary.get("entry_count", 0)) <= 0:
            raise RuntimeError(f"OpenRouter GPT-5.5 bootstrap smoke did not write any library entries: {merged_summary}")
        if int(history_summary.get("raw_summary_ok_count", 0)) <= 0:
            raise RuntimeError(f"OpenRouter GPT-5.5 bootstrap smoke did not write a raw model summary: {merged_summary}")
        return merged_summary
    finally:
        runs_volume.commit()
        cache_volume.commit()
        _stop_judge(judge)


@app.function(
    image=train_image,
    gpu=GPT_OSS_120B_SINGLE_GPU_CONFIG,
    timeout=2 * 60 * 60,
    cpu=16,
    memory=131072,
    volumes={"/runs": runs_volume, "/cache": cache_volume},
)
def gpt_oss_120b_xml_probe_smoke() -> dict[str, object]:
    _run_streaming(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    server = _start_gpt_oss_120b_server(cuda_visible_devices="0")
    try:
        result = _run_gpt_oss_xml_probe()
        runs_volume.commit()
        cache_volume.commit()
        return result
    finally:
        _stop_process(server)


@app.function(
    image=train_image,
    timeout=30 * 60,
    cpu=4,
    memory=8192,
    volumes={"/runs": runs_volume},
)
def prune_single_summary_smoke() -> dict[str, object]:
    prune_summary = _prune_to_single_summary(REMOTE_SINGLE_SUMMARY_OUTPUT_DIR)
    summary = _summarize_output(REMOTE_SINGLE_SUMMARY_OUTPUT_DIR)
    history_summary = _summarize_history_extraction(REMOTE_SINGLE_SUMMARY_OUTPUT_DIR)
    merged_summary = {**summary, "single_summary_prune": prune_summary, "history_extraction": history_summary}
    print(json.dumps(merged_summary, indent=2), flush=True)
    _validate_single_summary_extraction(history_summary)
    runs_volume.commit()
    return merged_summary


@app.local_entrypoint()
def main(action: str = "train"):
    if action == "verifier":
        print(verifier_smoke.remote())
    elif action == "check_training_packages":
        print(check_training_packages_smoke.remote())
    elif action == "train":
        print(train_smoke.remote())
    elif action == "train_history":
        print(train_history_smoke.remote())
    elif action == "bootstrap_single_summary":
        print(bootstrap_single_summary_smoke.remote())
    elif action == "train_single_summary":
        print(train_single_summary_smoke.remote())
    elif action == "bootstrap_single_summary_qwen_exec":
        print(bootstrap_qwen_exec_single_summary_smoke.remote())
    elif action == "train_single_summary_qwen_exec":
        print(train_qwen_exec_single_summary_smoke.remote())
    elif action == "bootstrap_single_summary_openrouter_gpt55":
        print(bootstrap_openrouter_gpt55_single_summary_smoke.remote())
    elif action == "train_single_summary_openrouter_gpt55":
        print(train_openrouter_gpt55_single_summary_smoke.remote())
    elif action == "train_single_summary_openrouter_gpt55_seeded":
        print(train_openrouter_gpt55_seeded_single_summary_smoke.remote())
    elif action == "train_openrouter_gpt55_4gpu_full_batch":
        print(train_openrouter_gpt55_4gpu_full_batch_smoke.remote())
    elif action == "train_evolvent_gpt55_4gpu_short_response":
        print(train_evolvent_gpt55_4gpu_short_response_smoke.remote())
    elif action == "train_gpt_oss_120b_5gpu_group64":
        print(train_gpt_oss_120b_5gpu_group64_smoke.remote())
    elif action == "train_gpt_oss_120b_3gpu_group16":
        print(train_gpt_oss_120b_3gpu_group16_smoke.remote())
    elif action == "train_gpt_oss_120b_3gpu_group16_h200_tuned":
        print(train_gpt_oss_120b_3gpu_group16_h200_tuned_smoke.spawn())
    elif action == "train_gpt_oss_120b_3gpu_batch8_group8_temp09":
        print(train_gpt_oss_120b_3gpu_batch8_group8_temp09_smoke.spawn())
    elif action == "train_gpt_oss_120b_3gpu_batch8_group16_concurrency16":
        print(train_gpt_oss_120b_3gpu_batch8_group16_concurrency16_smoke.spawn())
    elif action == "train_gpt_oss_120b_3gpu_batch8_group16_puct_fix":
        print(train_gpt_oss_120b_3gpu_batch8_group16_puct_fix_smoke.spawn())
    elif action == "train_gpt_oss_120b_3gpu_batch8_group16_concurrency16_50step":
        print(train_gpt_oss_120b_3gpu_batch8_group16_concurrency16_50step_smoke.spawn())
    elif action == "train_gpt_oss_120b_3gpu_batch8_group16_prompt_refinement_50step":
        print(train_gpt_oss_120b_3gpu_batch8_group16_prompt_refinement_50step_smoke.spawn())
    elif action == "bootstrap_gpt_oss_120b_seed":
        print(bootstrap_gpt_oss_120b_seed_smoke.remote())
    elif action == "gpt_oss_120b_xml_probe":
        print(gpt_oss_120b_xml_probe_smoke.remote())
    elif action == "prune_single_summary":
        print(prune_single_summary_smoke.remote())
    else:
        raise ValueError(
            "Unknown action "
            f"{action!r}; expected 'verifier', 'check_training_packages', 'train', 'train_history', "
            "'bootstrap_single_summary', "
            "'train_single_summary', 'bootstrap_single_summary_qwen_exec', "
            "'train_single_summary_qwen_exec', 'bootstrap_single_summary_openrouter_gpt55', "
            "'train_single_summary_openrouter_gpt55', 'train_single_summary_openrouter_gpt55_seeded', "
            "'train_openrouter_gpt55_4gpu_full_batch', 'train_evolvent_gpt55_4gpu_short_response', "
            "'train_gpt_oss_120b_5gpu_group64', 'train_gpt_oss_120b_3gpu_group16', "
            "'train_gpt_oss_120b_3gpu_group16_h200_tuned', "
            "'train_gpt_oss_120b_3gpu_batch8_group8_temp09', "
            "'train_gpt_oss_120b_3gpu_batch8_group16_concurrency16', "
            "'train_gpt_oss_120b_3gpu_batch8_group16_puct_fix', "
            "'train_gpt_oss_120b_3gpu_batch8_group16_concurrency16_50step', "
            "'train_gpt_oss_120b_3gpu_batch8_group16_prompt_refinement_50step', "
            "'bootstrap_gpt_oss_120b_seed', "
            "'gpt_oss_120b_xml_probe', "
            "or 'prune_single_summary'"
        )
