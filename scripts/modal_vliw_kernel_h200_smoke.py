from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections import Counter
from pathlib import Path

import modal
from fastapi import Request

APP_NAME = "guidance-ttt-vliw-kernel-smoke"
SECRET_NAME = "guidance-ttt-vliw-secrets"
GPU_CONFIG = "H200:2"
REMOTE_REPO_DIR = "/root/guidance"
REMOTE_CONFIG_PATH = (
    f"{REMOTE_REPO_DIR}/guidance_ttt/config/"
    "vliw_kernel_modal_h200_2gpu_glm52_batch8_group16_1step.yaml"
)
REMOTE_OUTPUT_DIR = "/runs/guidance_ttt/vliw_kernel_modal_h200_2gpu_glm52_batch8_group16_1step"
JUDGE_IMAGE = "seededge/edgebench.judge.vliw_kernel_optimization:5cdef0021634"
JUDGE_WORKSPACE = "/home/workspace/sebench_performance_takehome"
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
        ".git",
        ".hf_cache",
        ".pytest_cache",
        ".ray_tmp",
        ".runtime",
        ".secrets",
        ".tmp",
        ".triton_cache",
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
        if key not in {"EVOLVENT_API_KEY", "VLIW_JUDGE_TOKEN"}
    }
    env.update({"NO_PROXY": "*", "no_proxy": "*"})
    return env


app = modal.App(APP_NAME)
runs_volume = modal.Volume.from_name("guidance-ttt-runs", create_if_missing=True)
cache_volume = modal.Volume.from_name("guidance-ttt-cache", create_if_missing=True)
runtime_secret = modal.Secret.from_name(
    SECRET_NAME,
    required_keys=["EVOLVENT_API_KEY", "VLIW_JUDGE_TOKEN"],
)

judge_image = (
    modal.Image.from_registry(JUDGE_IMAGE)
    .entrypoint([])
    .pip_install("fastapi>=0.115", "pydantic>=2")
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
        # vLLM 0.12 currently resolves datasets 2.14.x, which still imports
        # PyExtensionType. PyArrow removed that compatibility alias in 21.
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
        # datasets 2.14.x misclassifies local filesystems with fsspec >=2023.10.
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
    secrets=[runtime_secret],
    timeout=700,
    cpu=4,
    memory=16384,
    max_containers=16,
    buffer_containers=2,
    scaledown_window=300,
    block_network=True,
)
@modal.concurrent(max_inputs=1)
@modal.fastapi_endpoint(method="POST", docs=False)
async def evaluate_vliw_candidate(request: Request):
    from fastapi import HTTPException

    expected_token = os.environ.get("VLIW_JUDGE_TOKEN", "")
    supplied_auth = str(request.headers.get("authorization") or "")
    if not expected_token or supplied_auth != f"Bearer {expected_token}":
        raise HTTPException(status_code=401, detail="unauthorized")
    payload = await request.json()
    solution = payload.get("solution") if isinstance(payload, dict) else None
    if not isinstance(solution, str) or not solution.strip():
        raise HTTPException(status_code=400, detail="solution must be a non-empty string")

    runner_timeout_s = max(1, min(int(payload.get("runner_timeout_s", 600)), 600))
    started_at = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="vliw-modal-judge-") as tmp_dir:
        workspace = Path(tmp_dir) / "workspace"
        shutil.copytree(JUDGE_WORKSPACE, workspace)
        (workspace / "solution.py").write_text(solution)
        report_path = Path(tmp_dir) / "report.json"
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "runner.py",
                    "--solution",
                    "solution.py",
                    "--cases",
                    "test_cases/hidden_cases.json",
                    "--output",
                    str(report_path),
                ],
                cwd=workspace,
                text=True,
                capture_output=True,
                timeout=runner_timeout_s,
            check=False,
            env=_judge_subprocess_env(),
        )
        except subprocess.TimeoutExpired as exc:
            return {
                "report": {
                    "all_correct": False,
                    "score_cycles": None,
                    "best_cycles": None,
                    "passed_thresholds": [],
                    "results": [],
                    "error": f"evaluation timed out after {runner_timeout_s}s",
                },
                "runner_returncode": None,
                "stdout": str(exc.stdout or "")[-4000:],
                "stderr": str(exc.stderr or "")[-4000:],
                "elapsed_s": time.monotonic() - started_at,
                "provider": "modal_http",
                "judge_image": JUDGE_IMAGE,
            }
        if report_path.exists():
            try:
                report = json.loads(report_path.read_text())
            except json.JSONDecodeError as exc:
                report = {
                    "all_correct": False,
                    "score_cycles": None,
                    "results": [],
                    "error": f"invalid report JSON: {exc}",
                }
        else:
            report = {
                "all_correct": False,
                "score_cycles": None,
                "results": [],
                "error": "runner did not produce report.json",
            }
        return {
            "report": report,
            "runner_returncode": completed.returncode,
            "stdout": (completed.stdout or "")[-4000:],
            "stderr": (completed.stderr or "")[-4000:],
            "elapsed_s": time.monotonic() - started_at,
            "provider": "modal_http",
            "judge_image": JUDGE_IMAGE,
        }


@app.function(
    image=judge_image,
    secrets=[runtime_secret],
    timeout=900,
    cpu=1,
    memory=4096,
)
def probe_vliw_judge(judge_url: str) -> dict[str, object]:
    """Exercise the deployed HTTP judge with the official starter before renting GPUs."""
    token = os.environ.get("VLIW_JUDGE_TOKEN", "")
    if not token:
        raise RuntimeError("Missing VLIW_JUDGE_TOKEN in the Modal runtime secret")
    baseline_path = Path(JUDGE_WORKSPACE) / "solution.py"
    if not baseline_path.exists():
        raise RuntimeError(f"Official judge image has no starter at {baseline_path}")
    request = urllib.request.Request(
        judge_url,
        data=json.dumps({"solution": baseline_path.read_text()}).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=750) as response:
        payload = json.loads(response.read())
    report = payload.get("report") if isinstance(payload, dict) else None
    if not isinstance(report, dict) or not report.get("all_correct"):
        raise RuntimeError(f"Official starter failed Modal judge probe: {payload}")
    cycles = report.get("score_cycles")
    if not isinstance(cycles, (int, float)) or cycles <= 0:
        raise RuntimeError(f"Modal judge probe returned invalid score_cycles: {payload}")
    return {
        "all_correct": True,
        "score_cycles": float(cycles),
        "case_count": len(report.get("results") or []),
        "provider": payload.get("provider"),
        "judge_image": payload.get("judge_image"),
    }


@app.function(image=train_image, timeout=300, cpu=2, memory=8192)
def probe_train_runtime() -> dict[str, str]:
    """Check the exact datasets/PyArrow/fsspec stack without allocating a GPU."""
    import datasets
    import fsspec
    import pandas as pd
    import pyarrow

    with tempfile.TemporaryDirectory(prefix="vliw-runtime-probe-") as tmp_dir:
        parquet_path = Path(tmp_dir) / "slots.parquet"
        pd.DataFrame({"slot": [0, 1]}).to_parquet(parquet_path, index=False)
        loaded = datasets.load_dataset("parquet", data_files=str(parquet_path))["train"]
    if len(loaded) != 2:
        raise RuntimeError(f"Runtime probe loaded {len(loaded)} rows, expected 2")
    return {
        "datasets": datasets.__version__,
        "fsspec": fsspec.__version__,
        "pyarrow": pyarrow.__version__,
        "rows": str(len(loaded)),
    }


def _run_streaming(command: list[str], *, env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    process = subprocess.Popen(command, cwd=REMOTE_REPO_DIR, env=env)
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}: {' '.join(command)}")


def _reset_output_dir() -> None:
    output_dir = Path(REMOTE_OUTPUT_DIR)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _smoke_summary(*, training_completed: bool) -> dict[str, object]:
    library_path = Path(REMOTE_OUTPUT_DIR) / "library.json"
    if not library_path.exists():
        return {"library_exists": False, "training_completed": training_completed}
    payload = json.loads(library_path.read_text())
    entries = [entry for entry in (payload.get("entries") or {}).values() if isinstance(entry, dict)]
    roots = [entry for entry in entries if int(entry.get("timestep") or 0) == 0]
    children = [entry for entry in entries if int(entry.get("timestep") or 0) == 1]
    groups = [group for group in (payload.get("groups") or {}).values() if isinstance(group, dict)]
    valid_children = [entry for entry in children if entry.get("verifier_status") == "valid"]
    positive_reward_children = [
        entry for entry in valid_children if float(entry.get("verifier_reward") or 0.0) > 0.0
    ]
    complete_valid_children = [
        entry
        for entry in positive_reward_children
        if bool(str(entry.get("solution") or "").strip())
        and bool(str((entry.get("metadata") or {}).get("raw_model_summary") or "").strip())
    ]
    raw_scores = [
        float(entry["verifier_raw_score"])
        for entry in valid_children
        if entry.get("verifier_raw_score") is not None
    ]
    models = Counter(
        str((entry.get("metadata") or {}).get("execution_model") or "missing") for entry in children
    )
    finish_reasons = Counter(
        str(
            (((entry.get("metadata") or {}).get("execution_response_metadata") or {}).get(
                "finish_reason"
            ))
            or "missing"
        )
        for entry in children
    )
    parsed_children = sum(
        bool(str(entry.get("solution") or "").strip())
        and bool(str((entry.get("metadata") or {}).get("raw_model_summary") or "").strip())
        for entry in children
    )
    result = {
        "library_exists": True,
        "training_completed": training_completed,
        "root_entries": len(roots),
        "valid_roots": sum(entry.get("verifier_status") == "valid" for entry in roots),
        "positive_reward_roots": sum(
            entry.get("verifier_status") == "valid"
            and float(entry.get("verifier_reward") or 0.0) > 0.0
            for entry in roots
        ),
        "child_entries": len(children),
        "valid_children": len(valid_children),
        "positive_reward_children": len(positive_reward_children),
        "complete_valid_children": len(complete_valid_children),
        "parsed_children": parsed_children,
        "best_cycles": min(raw_scores) if raw_scores else None,
        "execution_models": dict(models),
        "finish_reasons": dict(finish_reasons),
        "groups": len(groups),
        "finalized_groups": sum(bool(group.get("finalized")) for group in groups),
        "puct_T": int(payload.get("puct_T", 0)),
        "best_node_id": payload.get("best_node_id"),
    }
    return result


def _validate_smoke(summary: dict[str, object]) -> None:
    errors: list[str] = []
    if not summary.get("training_completed"):
        errors.append("training command did not complete")
    if summary.get("root_entries") != 1:
        errors.append(f"root_entries={summary.get('root_entries')}, expected 1")
    if summary.get("child_entries") != 128:
        errors.append(f"child_entries={summary.get('child_entries')}, expected 128")
    if summary.get("groups") != 8 or summary.get("finalized_groups") != 8:
        errors.append(
            f"group accounting={summary.get('finalized_groups')}/{summary.get('groups')}, expected 8/8"
        )
    if summary.get("puct_T") != 8:
        errors.append(f"puct_T={summary.get('puct_T')}, expected 8 group-wise updates")
    if int(summary.get("parsed_children") or 0) == 0:
        errors.append("no child contained both solution code and a raw summary")
    if int(summary.get("valid_children") or 0) == 0:
        errors.append("no child passed the official EdgeBench judge")
    if int(summary.get("positive_reward_children") or 0) == 0:
        errors.append("no valid child received a positive dense training reward")
    if int(summary.get("complete_valid_children") or 0) == 0:
        errors.append("no positive-reward child contained both solution code and a raw summary")
    models = summary.get("execution_models") or {}
    if not isinstance(models, dict) or int(models.get("glm-5.2", 0)) != 128:
        errors.append(f"execution model provenance mismatch: {models}")
    finish_reasons = summary.get("finish_reasons") or {}
    finish_reason_count = (
        sum(int(value) for value in finish_reasons.values())
        if isinstance(finish_reasons, dict)
        else 0
    )
    if finish_reason_count != 128:
        errors.append(f"execution finish-reason accounting mismatch: {finish_reasons}")
    elif int(finish_reasons.get("missing", 0)):
        errors.append(f"execution responses missing finish reasons: {finish_reasons}")
    if errors:
        raise RuntimeError("VLIW Modal smoke validation failed: " + "; ".join(errors))


@app.function(
    image=train_image,
    secrets=[runtime_secret],
    timeout=2 * 60 * 60,
    cpu=8,
    memory=32768,
    volumes={"/runs": runs_volume},
)
def bootstrap_vliw_seed(judge_url: str) -> dict[str, object]:
    """Build and verify the seed on Modal CPU before allocating training GPUs."""
    required_env = ["EVOLVENT_API_KEY", "VLIW_JUDGE_TOKEN"]
    missing = [name for name in required_env if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"Missing required Modal secret keys: {', '.join(missing)}")
    _reset_output_dir()
    env = {**os.environ, "VLIW_JUDGE_URL": judge_url}
    base_command = [
        sys.executable,
        "-m",
        "guidance_ttt.main_erdos",
        "--config",
        REMOTE_CONFIG_PATH,
    ]
    try:
        _run_streaming([*base_command, "--bootstrap-only"], env=env)
        summary = _smoke_summary(training_completed=False)
        if summary.get("root_entries") != 1 or summary.get("valid_roots") != 1:
            raise RuntimeError(f"Bootstrap did not produce exactly one valid root: {summary}")
        if summary.get("positive_reward_roots") != 1:
            raise RuntimeError(f"Bootstrap root did not receive a positive dense reward: {summary}")
        print("Bootstrap summary:", json.dumps(summary, indent=2), flush=True)
        return summary
    finally:
        runs_volume.commit()


@app.function(
    image=train_image,
    gpu=GPU_CONFIG,
    secrets=[runtime_secret],
    timeout=24 * 60 * 60,
    cpu=64,
    memory=262144,
    volumes={"/runs": runs_volume, "/cache": cache_volume},
)
def train_vliw_smoke(judge_url: str) -> dict[str, object]:
    required_env = ["EVOLVENT_API_KEY", "VLIW_JUDGE_TOKEN"]
    missing = [name for name in required_env if not os.environ.get(name)]
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
    env = {**os.environ, "VLIW_JUDGE_URL": judge_url, "CUDA_VISIBLE_DEVICES": "0,1"}
    base_command = [
        sys.executable,
        "-m",
        "guidance_ttt.main_erdos",
        "--config",
        REMOTE_CONFIG_PATH,
    ]
    try:
        bootstrap_summary = _smoke_summary(training_completed=False)
        if (
            bootstrap_summary.get("root_entries") != 1
            or bootstrap_summary.get("valid_roots") != 1
            or bootstrap_summary.get("child_entries") != 0
        ):
            raise RuntimeError(
                "Training requires one committed valid bootstrap root and no children: "
                f"{bootstrap_summary}"
            )
        subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import datasets, sys; "
                    "dataset = datasets.load_dataset('parquet', data_files=sys.argv[1])['train']; "
                    "print('slot_dataset_rows', len(dataset))"
                ),
                str(Path(REMOTE_OUTPUT_DIR) / "ttt_slots.parquet"),
            ],
            check=True,
            env=env,
        )
        _run_streaming(base_command, env=env)
        summary = _smoke_summary(training_completed=True)
        _validate_smoke(summary)
        result_path = Path(REMOTE_OUTPUT_DIR) / "modal_smoke_result.json"
        result_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
        print("Final smoke summary:", json.dumps(summary, indent=2), flush=True)
        return summary
    finally:
        runs_volume.commit()
        cache_volume.commit()


@app.local_entrypoint()
def main(action: str = "smoke"):
    judge_url = evaluate_vliw_candidate.get_web_url()
    if not judge_url:
        raise RuntimeError("Modal did not expose the VLIW judge web endpoint")
    if action == "judge_url":
        print(judge_url)
    elif action == "judge_probe":
        print(probe_vliw_judge.remote(judge_url))
    elif action == "runtime_probe":
        print(probe_train_runtime.remote())
    elif action == "bootstrap":
        print(bootstrap_vliw_seed.remote(judge_url))
    elif action == "train":
        print(train_vliw_smoke.remote(judge_url))
    elif action == "smoke":
        print(bootstrap_vliw_seed.remote(judge_url))
        print(train_vliw_smoke.remote(judge_url))
    else:
        raise ValueError(
            f"Unknown action {action!r}; expected 'smoke', 'bootstrap', 'train', "
            "'judge_probe', 'runtime_probe', or 'judge_url'"
        )
