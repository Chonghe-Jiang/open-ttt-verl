from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from collections import Counter
from pathlib import Path

import modal
from fastapi import Request

APP_NAME = "guidance-ttt-trimul-smoke"
# The legacy VLIW secret is the default only so existing Modal workspaces can
# launch immediately. A dedicated secret can expose TRIMUL_JUDGE_TOKEN instead.
SECRET_NAME = os.environ.get("GUIDANCE_TTT_TRIMUL_SECRET", "guidance-ttt-vliw-secrets")
AUTH_TOKEN_ENV = "TRIMUL_JUDGE_TOKEN"
LEGACY_AUTH_TOKEN_ENV = "VLIW_JUDGE_TOKEN"
TRAIN_GPU_CONFIG = "H200:1"
EVALUATION_GPU_CONFIG = "H100!"
EVALUATION_MAX_CONTAINERS = max(
    1, int(os.environ.get("TRIMUL_JUDGE_MAX_CONTAINERS", "4"))
)
REMOTE_REPO_DIR = "/root/guidance"
DEFAULT_PROMPT_MODE = "code_delta"
REMOTE_CONFIG_PATHS = {
    "code_delta": (
        f"{REMOTE_REPO_DIR}/guidance_ttt/config/"
        "trimul_modal_h200_1gpu_evolvent_glm52_group2_1step.yaml"
    ),
    "summary_only": (
        f"{REMOTE_REPO_DIR}/guidance_ttt/config/"
        "trimul_modal_h200_1gpu_evolvent_glm52_group2_summary_only_1step.yaml"
    ),
}
REMOTE_OUTPUT_DIRS = {
    "code_delta": "/runs/guidance_ttt/trimul_modal_h200_1gpu_evolvent_glm52_group2_1step",
    "summary_only": (
        "/runs/guidance_ttt/trimul_modal_h200_1gpu_evolvent_glm52_group2_summary_only_1step"
    ),
}
SCRATCH_SEED_CONFIG_PATH = (
    f"{REMOTE_REPO_DIR}/guidance_ttt/config/trimul_modal_glm52_scratch_seed.yaml"
)
SCRATCH_SEED_OUTPUT_DIR = "/runs/guidance_ttt/trimul_modal_glm52_scratch_seed"
FROZEN_SCRATCH_SEED_RELATIVE_PATH = (
    "guidance_ttt/seeds/trimul/glm52_scratch_bootstrap_library.json"
)
# Preserve these aliases for callers and tests that predate prompt-mode selection.
REMOTE_CONFIG_PATH = REMOTE_CONFIG_PATHS[DEFAULT_PROMPT_MODE]
REMOTE_OUTPUT_DIR = REMOTE_OUTPUT_DIRS[DEFAULT_PROMPT_MODE]
REMOTE_PACKAGE_DIR = "/opt/guidance_ttt"
REMOTE_EVALUATOR_DIR = f"{REMOTE_PACKAGE_DIR}/tasks/assets/trimul_evaluator"
FLASH_ATTN_TORCH29_CU12_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
    "flash_attn-2.8.3%2Bcu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
)
REPO_ROOT = Path(__file__).resolve().parents[1]


def _repo_ignore(path: Path) -> bool:
    path = Path(path)
    try:
        rel = path.relative_to(REPO_ROOT) if path.is_absolute() else path
    except ValueError:
        return False
    if not rel.parts:
        return False
    ignored_roots = {
        ".apptainer_cache",
        ".apptainer_home",
        ".apptainer_tmp",
        ".git",
        ".hf_cache",
        ".pytest_cache",
        ".ray_tmp",
        ".ruff_cache",
        ".runtime",
        ".secrets",
        ".tmp",
        ".triton_cache",
        ".venv",
        "models",
        "outputs",
        "reference",
        "results",
        "tmp_modal_20step",
        "verl.egg-info",
    }
    return rel.parts[0] in ignored_roots or "__pycache__" in rel.parts


def _judge_subprocess_env() -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"EVOLVENT_API_KEY", AUTH_TOKEN_ENV, LEGACY_AUTH_TOKEN_ENV}
    }
    env.update({"NO_PROXY": "*", "no_proxy": "*", "PYTHONUNBUFFERED": "1"})
    return env


def _auth_token() -> str:
    return os.environ.get(AUTH_TOKEN_ENV, "") or os.environ.get(LEGACY_AUTH_TOKEN_ENV, "")


def _normalize_prompt_mode(prompt_mode: str) -> str:
    normalized = str(prompt_mode).strip().lower()
    if normalized not in REMOTE_CONFIG_PATHS:
        expected = ", ".join(sorted(REMOTE_CONFIG_PATHS))
        raise ValueError(f"Unsupported prompt mode {prompt_mode!r}; expected one of: {expected}")
    return normalized


def _paths_for_prompt_mode(prompt_mode: str) -> tuple[str, str]:
    normalized = _normalize_prompt_mode(prompt_mode)
    return REMOTE_CONFIG_PATHS[normalized], REMOTE_OUTPUT_DIRS[normalized]


app = modal.App(APP_NAME)
runs_volume = modal.Volume.from_name("guidance-ttt-runs", create_if_missing=True)
cache_volume = modal.Volume.from_name("guidance-ttt-cache", create_if_missing=True)
runtime_secret = modal.Secret.from_name(SECRET_NAME)


# This keeps the task-relevant discover TriMul environment. The original shared
# GPU-mode image also installed JAX, TinyGrad, cuPyNumeric, and CUTLASS DSL, but
# neither the task nor evaluator imports them; their mutable upstream wheels can
# fail hash validation and must not make the pinned TriMul judge unavailable.
# Triton is pinned explicitly to the prompt's 3.3.1 contract.
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
    .env({"PYTHONPATH": "/opt", "PYTHONUNBUFFERED": "1", "TORCH_CUDA_ARCH_LIST": "9.0 9.0a"})
    .add_local_dir(REPO_ROOT / "guidance_ttt", REMOTE_PACKAGE_DIR)
)

train_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("build-essential", "git", "python3-dev")
    .uv_pip_install(
        "vllm>=0.8.5,<=0.12.0",
        "accelerate",
        "codetiming",
        "datasets",
        "dill",
        "hydra-core",
        "numpy<2.0.0",
        "pandas",
        "peft",
        "pyarrow>=19.0.0,<21.0.0",
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
        "fsspec[http]==2023.9.2",
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

    expected_token = _auth_token()
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


@app.function(image=train_image, timeout=300, cpu=2, memory=8192)
def probe_train_runtime() -> dict[str, str]:
    import datasets
    import fsspec
    import pandas as pd
    import pyarrow

    with tempfile.TemporaryDirectory(prefix="trimul-runtime-probe-") as tmp_dir:
        parquet_path = Path(tmp_dir) / "slots.parquet"
        pd.DataFrame({"slot": [0]}).to_parquet(parquet_path, index=False)
        loaded = datasets.load_dataset("parquet", data_files=str(parquet_path))["train"]
    if len(loaded) != 1:
        raise RuntimeError(f"Runtime probe loaded {len(loaded)} rows, expected 1")
    return {
        "datasets": datasets.__version__,
        "fsspec": fsspec.__version__,
        "pyarrow": pyarrow.__version__,
        "rows": str(len(loaded)),
    }


@app.function(image=train_image, secrets=[runtime_secret], timeout=1500, cpu=2, memory=8192)
def probe_trimul_judge(judge_url: str) -> dict[str, object]:
    token = _auth_token()
    if not token:
        raise RuntimeError(f"Missing {AUTH_TOKEN_ENV} in the Modal runtime secret")
    seed_path = Path(REMOTE_REPO_DIR) / FROZEN_SCRATCH_SEED_RELATIVE_PATH
    seed_payload = json.loads(seed_path.read_text())
    seed_entries = [
        entry
        for entry in (seed_payload.get("entries") or {}).values()
        if isinstance(entry, dict) and int(entry.get("timestep") or 0) == 0
    ]
    if len(seed_entries) != 1 or not str(seed_entries[0].get("solution") or "").strip():
        raise RuntimeError(f"Frozen TriMul scratch seed is malformed: {seed_path}")
    seed_solution = str(seed_entries[0]["solution"])
    request = urllib.request.Request(
        judge_url,
        data=json.dumps({"solution": seed_solution, "runner_timeout_s": 520}).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=1160) as response:
        payload = json.loads(response.read())
    report = payload.get("report") if isinstance(payload, dict) else None
    if not isinstance(report, dict) or not report.get("all_correct"):
        raise RuntimeError(f"Frozen GLM-5.2 scratch seed failed official evaluator probe: {payload}")
    return {
        "all_correct": True,
        "score_us": float(report["score_us"]),
        "test_count": report.get("test_count"),
        "benchmark_count": report.get("benchmark_count"),
        "evaluation_gpu": payload.get("evaluation_gpu"),
        "runtime_versions": payload.get("runtime_versions"),
    }


def _run_streaming(command: list[str], *, env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    process = subprocess.Popen(command, cwd=REMOTE_REPO_DIR, env=env)
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}: {' '.join(command)}")


def _reset_output_dir(output_dir: str | None = None) -> None:
    output_dir = Path(output_dir or REMOTE_OUTPUT_DIR)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _smoke_summary(*, training_completed: bool, output_dir: str | None = None) -> dict[str, object]:
    library_path = Path(output_dir or REMOTE_OUTPUT_DIR) / "library.json"
    if not library_path.exists():
        return {"library_exists": False, "training_completed": training_completed}
    payload = json.loads(library_path.read_text())
    entries = [entry for entry in (payload.get("entries") or {}).values() if isinstance(entry, dict)]
    roots = [entry for entry in entries if int(entry.get("timestep") or 0) == 0]
    children = [entry for entry in entries if int(entry.get("timestep") or 0) == 1]
    groups = [group for group in (payload.get("groups") or {}).values() if isinstance(group, dict)]
    valid_children = [entry for entry in children if entry.get("verifier_status") == "valid"]
    parsed_children = [
        entry
        for entry in children
        if bool(str(entry.get("solution") or "").strip())
        and bool(str((entry.get("metadata") or {}).get("raw_model_summary") or "").strip())
    ]
    child_raw_scores = [
        float(entry["verifier_raw_score"])
        for entry in valid_children
        if entry.get("verifier_raw_score") is not None
    ]
    library_raw_scores = [
        float(entry["verifier_raw_score"])
        for entry in entries
        if entry.get("verifier_status") == "valid" and entry.get("verifier_raw_score") is not None
    ]
    models = Counter(
        str((entry.get("metadata") or {}).get("execution_model") or "missing") for entry in children
    )
    finish_reasons = Counter(
        str(
            (((entry.get("metadata") or {}).get("execution_response_metadata") or {}).get("finish_reason"))
            or "missing"
        )
        for entry in children
    )
    statuses = Counter(str(entry.get("verifier_status") or "missing") for entry in children)
    prompt_modes = Counter(
        str((entry.get("metadata") or {}).get("prompt_mode") or "missing") for entry in entries
    )
    root_prompt_modes = Counter(
        str((entry.get("metadata") or {}).get("prompt_mode") or "missing") for entry in roots
    )
    child_prompt_modes = Counter(
        str((entry.get("metadata") or {}).get("prompt_mode") or "missing") for entry in children
    )
    root_bootstrap_sources = Counter(
        str((entry.get("metadata") or {}).get("bootstrap_source") or "missing") for entry in roots
    )
    summary_semantics = Counter(
        str((entry.get("metadata") or {}).get("summary_semantics") or "missing") for entry in entries
    )
    summary_only_guidance_contexts = 0
    code_delta_guidance_contexts = 0
    execution_parent_contexts = 0
    for entry in children:
        metadata = entry.get("metadata") or {}
        guidance_prompt = metadata.get("guidance_prompt") or {}
        execution_prompt = metadata.get("execution_prompt") or {}
        guidance_user = str(guidance_prompt.get("user") or "")
        execution_user = str(execution_prompt.get("user") or "")
        if "<selected_summary>" in guidance_user and "<parent_code>" not in guidance_user:
            summary_only_guidance_contexts += 1
        if "<selected_candidate>" in guidance_user and "<parent_code>" in guidance_user:
            code_delta_guidance_contexts += 1
        if "<selected_parent>" in execution_user and "<parent_code>" in execution_user:
            execution_parent_contexts += 1
    return {
        "library_exists": True,
        "training_completed": training_completed,
        "root_entries": len(roots),
        "valid_roots": sum(entry.get("verifier_status") == "valid" for entry in roots),
        "child_entries": len(children),
        "parsed_children": len(parsed_children),
        "valid_children": len(valid_children),
        "best_runtime_us": min(library_raw_scores) if library_raw_scores else None,
        "best_child_runtime_us": min(child_raw_scores) if child_raw_scores else None,
        "execution_models": dict(models),
        "finish_reasons": dict(finish_reasons),
        "verifier_statuses": dict(statuses),
        "prompt_modes": dict(prompt_modes),
        "root_prompt_modes": dict(root_prompt_modes),
        "child_prompt_modes": dict(child_prompt_modes),
        "root_bootstrap_sources": dict(root_bootstrap_sources),
        "summary_semantics": dict(summary_semantics),
        "summary_only_guidance_contexts": summary_only_guidance_contexts,
        "code_delta_guidance_contexts": code_delta_guidance_contexts,
        "execution_parent_contexts": execution_parent_contexts,
        "groups": len(groups),
        "finalized_groups": sum(bool(group.get("finalized")) for group in groups),
        "puct_T": int(payload.get("puct_T", 0)),
        "best_node_id": payload.get("best_node_id"),
    }


def _scratch_seed_audit(*, output_dir: str = SCRATCH_SEED_OUTPUT_DIR) -> dict[str, object]:
    library_path = Path(output_dir) / "library.json"
    payload = json.loads(library_path.read_text())
    entries = [entry for entry in (payload.get("entries") or {}).values() if isinstance(entry, dict)]
    roots = [entry for entry in entries if int(entry.get("timestep") or 0) == 0]
    if len(roots) != 1:
        raise RuntimeError(f"Scratch generation produced {len(roots)} root entries; expected exactly one")
    entry = roots[0]
    metadata = entry.get("metadata") or {}
    prompt = metadata.get("execution_prompt") or {}
    prompt_user = str(prompt.get("user") or "")
    solution = str(entry.get("solution") or "")
    raw_summary = str(metadata.get("raw_model_summary") or "")
    errors: list[str] = []
    if entry.get("verifier_status") != "valid":
        errors.append(f"verifier_status={entry.get('verifier_status')!r}")
    if metadata.get("bootstrap_source") != "scratch":
        errors.append(f"bootstrap_source={metadata.get('bootstrap_source')!r}")
    if "<baseline_candidate>" in prompt_user or "<baseline_code>" in prompt_user:
        errors.append("scratch prompt contains a baseline candidate")
    if "bootstrap_baseline_verification" in metadata:
        errors.append("scratch entry contains baseline verification metadata")
    if "There is no parent candidate, prior solution" not in prompt_user:
        errors.append("scratch prompt is missing its no-prior-solution contract")
    if not solution.strip():
        errors.append("scratch entry has no solution")
    if not raw_summary.strip():
        errors.append("scratch entry has no raw model summary")
    discover_path = REPO_ROOT / "guidance_ttt/tasks/assets/trimul/baseline_solution.py"
    if discover_path.exists() and solution.encode() == discover_path.read_bytes():
        errors.append("scratch solution is byte-identical to the archived Discover result")
    if errors:
        raise RuntimeError("Invalid TriMul scratch seed: " + "; ".join(errors))
    return {
        "library_path": str(library_path),
        "entry_id": entry.get("id"),
        "bootstrap_source": metadata.get("bootstrap_source"),
        "bootstrap_attempts": metadata.get("bootstrap_attempts"),
        "execution_model": metadata.get("execution_model"),
        "verifier_status": entry.get("verifier_status"),
        "runtime_us": entry.get("verifier_raw_score"),
        "reward": entry.get("verifier_reward"),
        "solution_sha256": hashlib.sha256(solution.encode()).hexdigest(),
        "solution_chars": len(solution),
        "summary_chars": len(raw_summary),
        "prompt_has_parent": "<parent_code>" in prompt_user,
        "prompt_has_baseline": "<baseline_candidate>" in prompt_user,
    }


def _validate_smoke(
    summary: dict[str, object], *, expected_prompt_mode: str = DEFAULT_PROMPT_MODE
) -> None:
    expected_prompt_mode = _normalize_prompt_mode(expected_prompt_mode)
    errors: list[str] = []
    if not summary.get("training_completed"):
        errors.append("training command did not complete")
    if summary.get("root_entries") != 1 or summary.get("valid_roots") != 1:
        errors.append(
            f"bootstrap roots={summary.get('valid_roots')}/{summary.get('root_entries')}, expected 1/1"
        )
    if summary.get("child_entries") != 2:
        errors.append(f"child_entries={summary.get('child_entries')}, expected 2")
    if summary.get("groups") != 1 or summary.get("finalized_groups") != 1:
        errors.append(
            f"group accounting={summary.get('finalized_groups')}/{summary.get('groups')}, expected 1/1"
        )
    if summary.get("puct_T") != 1:
        errors.append(f"puct_T={summary.get('puct_T')}, expected one group-wise update")
    if int(summary.get("parsed_children") or 0) == 0:
        errors.append("no child contained both complete Python code and a raw model summary")
    if int(summary.get("valid_children") or 0) == 0:
        errors.append("no child passed the official H100 TriMul evaluator")
    finish_reasons = summary.get("finish_reasons") or {}
    if not isinstance(finish_reasons, dict) or sum(int(v) for v in finish_reasons.values()) != 2:
        errors.append(f"execution finish-reason accounting mismatch: {finish_reasons}")
    root_prompt_modes = summary.get("root_prompt_modes") or {}
    if root_prompt_modes != {"summary_only": 1}:
        errors.append(f"frozen-root prompt-mode accounting mismatch: {root_prompt_modes}")
    root_bootstrap_sources = summary.get("root_bootstrap_sources") or {}
    if root_bootstrap_sources != {"scratch": 1}:
        errors.append(f"frozen-root bootstrap-source accounting mismatch: {root_bootstrap_sources}")
    child_prompt_modes = summary.get("child_prompt_modes") or {}
    expected_child_count = int(summary.get("child_entries") or 0)
    if child_prompt_modes != {expected_prompt_mode: expected_child_count}:
        errors.append(
            f"child prompt-mode accounting mismatch: {child_prompt_modes}, expected {expected_prompt_mode}"
        )
    if summary.get("execution_parent_contexts") != 2:
        errors.append(
            f"execution parent-code contexts={summary.get('execution_parent_contexts')}, expected 2"
        )
    if expected_prompt_mode == "summary_only":
        if summary.get("summary_only_guidance_contexts") != 2:
            errors.append(
                "summary_only guidance contexts="
                f"{summary.get('summary_only_guidance_contexts')}, expected 2 without parent code"
            )
        if summary.get("code_delta_guidance_contexts") != 0:
            errors.append("summary_only guidance unexpectedly received a code_delta parent context")
    else:
        if summary.get("code_delta_guidance_contexts") != 2:
            errors.append(
                f"code_delta guidance contexts={summary.get('code_delta_guidance_contexts')}, expected 2"
            )
        if summary.get("summary_only_guidance_contexts") != 0:
            errors.append("code_delta guidance unexpectedly received a summary_only context")
    if errors:
        raise RuntimeError("TriMul Modal smoke validation failed: " + "; ".join(errors))


@app.function(
    image=train_image,
    secrets=[runtime_secret],
    timeout=3 * 60 * 60,
    cpu=8,
    memory=32768,
    volumes={"/runs": runs_volume},
)
def bootstrap_trimul_seed(
    judge_url: str, prompt_mode: str = DEFAULT_PROMPT_MODE
) -> dict[str, object]:
    missing = [name for name in ["EVOLVENT_API_KEY"] if not os.environ.get(name)]
    if not _auth_token():
        missing.append(f"{AUTH_TOKEN_ENV} (or {LEGACY_AUTH_TOKEN_ENV})")
    if missing:
        raise RuntimeError(f"Missing required Modal secret keys: {', '.join(missing)}")
    prompt_mode = _normalize_prompt_mode(prompt_mode)
    config_path, output_dir = _paths_for_prompt_mode(prompt_mode)
    _reset_output_dir(output_dir)
    env = {**os.environ, AUTH_TOKEN_ENV: _auth_token(), "TRIMUL_JUDGE_URL": judge_url}
    command = [
        sys.executable,
        "-m",
        "guidance_ttt.main_erdos",
        "--config",
        config_path,
        "--bootstrap-only",
    ]
    try:
        _run_streaming(command, env=env)
        summary = _smoke_summary(training_completed=False, output_dir=output_dir)
        if summary.get("root_entries") != 1 or summary.get("valid_roots") != 1:
            raise RuntimeError(f"Bootstrap did not produce exactly one valid root: {summary}")
        print("Bootstrap summary:", json.dumps(summary, indent=2), flush=True)
        return summary
    finally:
        runs_volume.commit()


@app.function(
    image=train_image,
    secrets=[runtime_secret],
    timeout=3 * 60 * 60,
    cpu=8,
    memory=32768,
    volumes={"/runs": runs_volume},
)
def generate_trimul_scratch_seed(judge_url: str) -> dict[str, object]:
    missing = [name for name in ["EVOLVENT_API_KEY"] if not os.environ.get(name)]
    if not _auth_token():
        missing.append(f"{AUTH_TOKEN_ENV} (or {LEGACY_AUTH_TOKEN_ENV})")
    if missing:
        raise RuntimeError(f"Missing required Modal secret keys: {', '.join(missing)}")
    _reset_output_dir(SCRATCH_SEED_OUTPUT_DIR)
    env = {**os.environ, AUTH_TOKEN_ENV: _auth_token(), "TRIMUL_JUDGE_URL": judge_url}
    command = [
        sys.executable,
        "-m",
        "guidance_ttt.main_erdos",
        "--config",
        SCRATCH_SEED_CONFIG_PATH,
        "--bootstrap-only",
    ]
    try:
        _run_streaming(command, env=env)
        audit = _scratch_seed_audit()
        result_path = Path(SCRATCH_SEED_OUTPUT_DIR) / "scratch_seed_result.json"
        result_path.write_text(json.dumps(audit, indent=2, sort_keys=True))
        print("Scratch seed audit:", json.dumps(audit, indent=2), flush=True)
        return audit
    finally:
        runs_volume.commit()


@app.function(
    image=train_image,
    gpu=TRAIN_GPU_CONFIG,
    secrets=[runtime_secret],
    timeout=24 * 60 * 60,
    cpu=32,
    memory=131072,
    volumes={"/runs": runs_volume, "/cache": cache_volume},
)
def train_trimul_smoke(
    judge_url: str, prompt_mode: str = DEFAULT_PROMPT_MODE
) -> dict[str, object]:
    missing = [name for name in ["EVOLVENT_API_KEY"] if not os.environ.get(name)]
    if not _auth_token():
        missing.append(f"{AUTH_TOKEN_ENV} (or {LEGACY_AUTH_TOKEN_ENV})")
    if missing:
        raise RuntimeError(f"Missing required Modal secret keys: {', '.join(missing)}")
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import datasets, fsspec, pyarrow, torch, transformers, vllm; "
                "print('runtime_versions', datasets.__version__, fsspec.__version__, "
                "pyarrow.__version__, torch.__version__, transformers.__version__, vllm.__version__)"
            ),
        ],
        check=True,
    )
    subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        check=True,
    )
    env = {
        **os.environ,
        AUTH_TOKEN_ENV: _auth_token(),
        "TRIMUL_JUDGE_URL": judge_url,
        "CUDA_VISIBLE_DEVICES": "0",
    }
    prompt_mode = _normalize_prompt_mode(prompt_mode)
    config_path, output_dir = _paths_for_prompt_mode(prompt_mode)
    command = [
        sys.executable,
        "-m",
        "guidance_ttt.main_erdos",
        "--config",
        config_path,
    ]
    try:
        bootstrap_summary = _smoke_summary(training_completed=False, output_dir=output_dir)
        if (
            bootstrap_summary.get("root_entries") != 1
            or bootstrap_summary.get("valid_roots") != 1
            or bootstrap_summary.get("child_entries") != 0
        ):
            raise RuntimeError(f"Training requires one clean valid bootstrap root: {bootstrap_summary}")
        _run_streaming(command, env=env)
        summary = _smoke_summary(training_completed=True, output_dir=output_dir)
        _validate_smoke(summary, expected_prompt_mode=prompt_mode)
        result_path = Path(output_dir) / "modal_smoke_result.json"
        result_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
        print("Final smoke summary:", json.dumps(summary, indent=2), flush=True)
        return summary
    finally:
        runs_volume.commit()
        cache_volume.commit()


@app.local_entrypoint()
def main(action: str = "smoke", prompt_mode: str = DEFAULT_PROMPT_MODE):
    prompt_mode = _normalize_prompt_mode(prompt_mode)
    judge_url = evaluate_trimul_candidate.get_web_url()
    if not judge_url:
        raise RuntimeError("Modal did not expose the TriMul H100 evaluator endpoint")
    if action == "judge_url":
        print(judge_url)
    elif action == "runtime_probe":
        print(probe_train_runtime.remote())
    elif action == "judge_probe":
        print(probe_trimul_judge.remote(judge_url))
    elif action == "bootstrap":
        print(bootstrap_trimul_seed.remote(judge_url, prompt_mode))
    elif action == "generate_scratch_seed":
        print(generate_trimul_scratch_seed.remote(judge_url))
    elif action == "train":
        print(train_trimul_smoke.remote(judge_url, prompt_mode))
    elif action == "smoke":
        print(probe_train_runtime.remote())
        print(bootstrap_trimul_seed.remote(judge_url, prompt_mode))
        print(train_trimul_smoke.remote(judge_url, prompt_mode))
    else:
        raise ValueError(
            f"Unknown action {action!r}; expected smoke, bootstrap, generate_scratch_seed, train, "
            "judge_probe, runtime_probe, or judge_url"
        )
