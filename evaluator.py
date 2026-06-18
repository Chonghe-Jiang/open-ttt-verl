from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

try:
    from openevolve.evaluation_result import EvaluationResult
except ModuleNotFoundError:
    @dataclass
    class EvaluationResult:
        metrics: dict[str, float]
        artifacts: dict[str, Union[str, bytes]] = field(default_factory=dict)

from erdos_evolve.verifier import verify_erdos_program_text


def _artifact_suggestion(status: str) -> str:
    if status == "timeout":
        return "Reduce loops or honor budget_s so run() completes before the evaluator timeout."
    if status == "valid":
        return "Try lowering C5 while preserving run(seed=42, budget_s=..., **kwargs)."
    return "Return finite 1D h_values in [0, 1], sum n_points/2, and recompute c5_bound exactly."


def _result_to_evaluation(result, runtime_s: float) -> EvaluationResult:
    raw_score = float(result.raw_score) if result.raw_score is not None else 0.0
    reward = float(result.reward)
    metrics = {
        "combined_score": reward,
        "c5_reward": reward,
        "c5_score": raw_score,
        "raw_c5": raw_score,
        "c5_quality": 1.0 / (1.0 + raw_score) if result.valid else 0.0,
        "valid": 1.0 if result.valid else 0.0,
        "runtime_s": float(runtime_s),
        "n_points": 256.0,
    }
    artifacts = {
        "status": result.status,
        "message": result.message,
        "raw_score": "" if result.raw_score is None else f"{result.raw_score:.12f}",
        "suggestion": _artifact_suggestion(result.status),
    }
    if "code" in result.artifacts:
        artifacts["code_excerpt"] = str(result.artifacts["code"])[:4000]
    return EvaluationResult(metrics=metrics, artifacts=artifacts)


def evaluate(program_path) -> EvaluationResult:
    start = time.time()
    path = Path(program_path)
    try:
        code = path.read_text()
    except Exception as exc:
        return EvaluationResult(
            metrics={
                "combined_score": 0.0,
                "c5_reward": 0.0,
                "c5_score": 0.0,
                "raw_c5": 0.0,
                "c5_quality": 0.0,
                "valid": 0.0,
                "runtime_s": 0.0,
                "n_points": 256.0,
            },
            artifacts={
                "status": "read_error",
                "message": str(exc),
                "suggestion": "Ensure OpenEvolve passes a readable Python program path to evaluator.py.",
            },
        )
    result = verify_erdos_program_text(code, timeout_s=10, n_points=256)
    return _result_to_evaluation(result, time.time() - start)


def evaluate_stage1(program_path) -> EvaluationResult:
    start = time.time()
    path = Path(program_path)
    try:
        code = path.read_text()
    except Exception as exc:
        return EvaluationResult(
            metrics={
                "combined_score": 0.0,
                "c5_reward": 0.0,
                "c5_score": 0.0,
                "raw_c5": 0.0,
                "c5_quality": 0.0,
                "valid": 0.0,
                "runtime_s": 0.0,
                "n_points": 64.0,
            },
            artifacts={"status": "read_error", "message": str(exc), "suggestion": "Readable Python file required."},
        )
    result = verify_erdos_program_text(code, timeout_s=5, n_points=64)
    return _result_to_evaluation(result, time.time() - start)


def evaluate_stage2(program_path) -> EvaluationResult:
    return evaluate(program_path)
