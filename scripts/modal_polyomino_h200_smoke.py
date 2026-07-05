from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

import modal


APP_NAME = "guidance-ttt-polyomino-h200-smoke"
GPU_CONFIG = "H200:4"
HISTORY_GPU_CONFIG = "H200:2"
REMOTE_REPO_DIR = "/root/guidance"
REMOTE_FRONTIER_DIR = "/opt/Frontier-CS"
REMOTE_CONFIG_PATH = f"{REMOTE_REPO_DIR}/guidance_ttt/config/polyomino_modal_h200_4gpu_smoke.yaml"
REMOTE_OUTPUT_DIR = "/runs/guidance_ttt/polyomino_modal_h200_4gpu_smoke"
REMOTE_HISTORY_CONFIG_PATH = f"{REMOTE_REPO_DIR}/guidance_ttt/config/polyomino_modal_h200_2gpu_history_smoke.yaml"
REMOTE_HISTORY_OUTPUT_DIR = "/runs/guidance_ttt/polyomino_modal_h200_2gpu_history_smoke"
JUDGE_LOG_PATH = "/tmp/frontier_judge.log"

REPO_ROOT = Path(__file__).resolve().parents[1]

app = modal.App(APP_NAME)
runs_volume = modal.Volume.from_name("guidance-ttt-runs", create_if_missing=True)
cache_volume = modal.Volume.from_name("guidance-ttt-cache", create_if_missing=True)


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


def _print_judge_log(max_bytes: int = 20000) -> None:
    log_path = Path(JUDGE_LOG_PATH)
    if not log_path.exists():
        print(f"Judge log does not exist: {JUDGE_LOG_PATH}", flush=True)
        return
    payload = log_path.read_bytes()[-max_bytes:].decode(errors="replace")
    print(f"----- {JUDGE_LOG_PATH} tail -----\n{payload}\n----- end judge log -----", flush=True)


def _start_judge() -> subprocess.Popen:
    env = {
        **__import__("os").environ,
        "PORT": "8081",
        "GJ_ADDR": "http://127.0.0.1:5050",
        "JUDGE_WORKERS": "8",
        "GJ_PARALLELISM": "8",
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


def _validate_history_extraction(summary: dict[str, object]) -> None:
    entry_count = int(summary.get("entry_count", 0))
    if entry_count <= 0:
        raise RuntimeError(f"History smoke did not write library entries: {summary}")
    for key in ("guidance_ok_count", "raw_summary_ok_count", "canonical_summary_ok_count", "execution_text_ok_count"):
        if int(summary.get(key, 0)) <= 0:
            raise RuntimeError(f"History smoke missing {key}: {summary}")


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
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "TORCH_CUDA_ARCH_LIST": "9.0 9.0a",
            "VLLM_NO_USAGE_STATS": "1",
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


@app.local_entrypoint()
def main(action: str = "train"):
    if action == "verifier":
        print(verifier_smoke.remote())
    elif action == "train":
        print(train_smoke.remote())
    elif action == "train_history":
        print(train_history_smoke.remote())
    else:
        raise ValueError(f"Unknown action {action!r}; expected 'verifier', 'train', or 'train_history'")
