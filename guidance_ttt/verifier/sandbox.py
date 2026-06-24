from __future__ import annotations

import builtins
import multiprocessing as mp
import queue
import re
import traceback
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SandboxResult:
    output: Any | None
    error: str | None


def extract_python_code(text: str) -> str | None:
    solution_match = re.search(r"<solution>\s*([\s\S]*?)\s*</solution>", text)
    if solution_match:
        return extract_python_code(solution_match.group(1))
    matches = list(re.finditer(r"```python\s*([\s\S]*?)\s*```", text))
    if matches:
        return matches[-1].group(1).strip()
    channel_matches = list(re.finditer(r"(?:assistant)?commentary\s+to=python\s+code([\s\S]*)", text))
    if channel_matches:
        code = channel_matches[-1].group(1).strip()
        summary_start = code.find("<summary>")
        if summary_start != -1:
            code = code[:summary_start].strip()
        return code
    stripped = text.strip()
    if stripped.startswith("def run"):
        return stripped
    return None


def _worker(code: str, result_queue: mp.Queue) -> None:
    try:
        globals_dict = {
            "__builtins__": builtins.__dict__,
            "np": np,
            "numpy": np,
        }
        exec(code, globals_dict)
        run_fn = globals_dict.get("run")
        if run_fn is None:
            raise ValueError("Program must define run(seed=42, budget_s=..., **kwargs)")
        result_queue.put(SandboxResult(output=run_fn(), error=None))
    except Exception:
        result_queue.put(SandboxResult(output=None, error=traceback.format_exc()))


def evaluate_python_code(code: str, *, timeout_s: int) -> SandboxResult:
    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue(maxsize=1)
    process = ctx.Process(target=_worker, args=(code, result_queue))
    process.start()
    process.join(timeout_s)
    if process.is_alive():
        process.terminate()
        process.join(1)
        if process.is_alive():
            process.kill()
        return SandboxResult(output=None, error=f"Timed out after {timeout_s}s")
    try:
        return result_queue.get_nowait()
    except queue.Empty:
        return SandboxResult(output=None, error=f"Process exited with code {process.exitcode}")
