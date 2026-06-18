from __future__ import annotations

from typing import Any

import numpy as np

from guidance_ttt.state import VerificationResult
from guidance_ttt.verifier.sandbox import evaluate_python_code, extract_python_code


class ErdosEvaluationError(ValueError):
    pass


def verify_c5_solution(h_values: Any, c5_achieved: float, n_points: int) -> float:
    try:
        h_array = np.asarray(h_values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ErdosEvaluationError(f"Cannot convert h_values to numpy array: {exc}") from exc
    if h_array.ndim != 1:
        raise ErdosEvaluationError(f"h_values must be 1D array, got shape {h_array.shape}")
    if h_array.shape[0] != int(n_points):
        raise ErdosEvaluationError(f"Expected h shape ({n_points},), got {h_array.shape}")
    if not np.all(np.isfinite(h_array)):
        raise ErdosEvaluationError("h_values contain NaN or inf values")
    if np.any(h_array < 0) or np.any(h_array > 1):
        raise ErdosEvaluationError(f"h(x) is not in [0, 1]. Range: [{h_array.min()}, {h_array.max()}]")
    target_sum = int(n_points) / 2.0
    current_sum = float(np.sum(h_array))
    if current_sum == 0:
        raise ErdosEvaluationError("h_values sum to zero")
    if not np.isclose(current_sum, target_sum, atol=1e-8):
        raise ErdosEvaluationError(f"sum(h) must equal n_points / 2. Got {current_sum:.12g}, expected {target_sum:.12g}")
    dx = 2.0 / int(n_points)
    computed_c5 = float(np.max(np.correlate(h_array, 1.0 - h_array, mode="full") * dx))
    if not np.isfinite(computed_c5):
        raise ErdosEvaluationError(f"Computed C5 is not finite: {computed_c5}")
    if not np.isclose(computed_c5, float(c5_achieved), atol=1e-4):
        raise ErdosEvaluationError(f"C5 mismatch: reported {c5_achieved:.6f}, computed {computed_c5:.6f}")
    return computed_c5


def verify_erdos_solution_text(text: str, *, timeout_s: int) -> VerificationResult:
    code = extract_python_code(text)
    if code is None:
        return VerificationResult(
            reward=0.0,
            raw_score=None,
            valid=False,
            status="parse_error",
            message="No Python code block found",
            artifacts={},
        )
    sandbox = evaluate_python_code(code, timeout_s=timeout_s)
    if sandbox.error is not None:
        status = "timeout" if sandbox.error.startswith("Timed out") else "invalid"
        return VerificationResult(
            reward=0.0,
            raw_score=None,
            valid=False,
            status=status,
            message=sandbox.error,
            artifacts={"code": code},
        )
    try:
        h_values, c5_bound, n_points = sandbox.output
        raw_score = verify_c5_solution(h_values, c5_bound, n_points)
    except Exception as exc:
        return VerificationResult(
            reward=0.0,
            raw_score=None,
            valid=False,
            status="invalid",
            message=str(exc),
            artifacts={"code": code},
        )
    reward = 1.0 / (1e-8 + raw_score)
    return VerificationResult(
        reward=reward,
        raw_score=raw_score,
        valid=True,
        status="valid",
        message=f"C5 bound: {raw_score:.6f}",
        artifacts={"code": code},
    )
