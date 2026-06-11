from __future__ import annotations

import builtins
import contextlib
import io
import multiprocessing as mp
import os
import queue
import re
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from erdos_slime.ttt_discover.erdos_env import evaluate_erdos_solution, verify_c5_solution
from erdos_slime.ttt_discover.state import DiscoveryState


@dataclass
class SandboxResult:
    output: Any | None
    stdout: str
    error: str | None


def extract_python_code(response: str) -> str | None:
    match = re.search(r"```python\s+([\s\S]*?)\s*```", response)
    if match:
        return match.group(1).strip()
    stripped = response.strip()
    return stripped if stripped.startswith("def run") else None


def _install_sandbox_guards() -> None:
    original_open = builtins.open

    def guarded_open(file, mode="r", *args, **kwargs):
        if any(flag in mode for flag in ("w", "a", "+", "x")):
            raise PermissionError("File writes are disabled in TTT sandbox")
        return original_open(file, mode, *args, **kwargs)

    builtins.open = guarded_open

    def blocked(*args, **kwargs):
        raise PermissionError("Filesystem mutation is disabled in TTT sandbox")

    for name in ("remove", "unlink", "rename", "replace", "rmdir", "mkdir", "makedirs", "chmod"):
        if hasattr(os, name):
            setattr(os, name, blocked)


def _worker(code: str, construction: list[Any] | None, timeout_s: int, result_queue: mp.Queue) -> None:
    try:
        _install_sandbox_guards()
        globals_dict = {
            "__builtins__": builtins.__dict__,
            "np": np,
            "numpy": np,
            "evaluate_erdos_solution": evaluate_erdos_solution,
            "verify_c5_solution": verify_c5_solution,
            "initial_h_values": np.asarray(construction, dtype=np.float64) if construction is not None else None,
        }
        locals_dict: dict[str, Any] = {}
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exec(code, globals_dict, locals_dict)
            run_fn = locals_dict.get("run") or globals_dict.get("run")
            if run_fn is None:
                raise ValueError("Program must define run(seed=42, budget_s=..., **kwargs)")
            output = run_fn(seed=42, budget_s=int(timeout_s))
        result_queue.put(SandboxResult(output=output, stdout=stdout.getvalue(), error=None))
    except Exception:
        result_queue.put(SandboxResult(output=None, stdout="", error=traceback.format_exc()))


def evaluate_python_code(
    code: str,
    *,
    state: DiscoveryState,
    timeout_s: int,
    work_dir: str | Path | None = None,
) -> SandboxResult:
    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue(maxsize=1)
    with tempfile.TemporaryDirectory(dir=str(work_dir) if work_dir else None):
        process = ctx.Process(target=_worker, args=(code, state.construction, int(timeout_s), result_queue))
        process.start()
        process.join(timeout_s)
        if process.is_alive():
            process.terminate()
            process.join(1)
            if process.is_alive():
                process.kill()
            return SandboxResult(output=None, stdout="", error=f"Timed out after {timeout_s}s")
        try:
            return result_queue.get_nowait()
        except queue.Empty:
            return SandboxResult(output=None, stdout="", error=f"Process exited with code {process.exitcode}")
