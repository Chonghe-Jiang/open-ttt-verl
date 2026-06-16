from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from erdos_evolve.sandbox import evaluate_python_code


class ErdosEvaluationError(ValueError):
    pass


@dataclass
class VerificationResult:
    reward: float
    raw_score: float | None
    valid: bool
    status: str
    message: str
    artifacts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_c5(h_values: Any, n_points: int) -> float:
    h_array = np.asarray(h_values, dtype=np.float64)
    dx = 2.0 / int(n_points)
    return float(np.max(np.correlate(h_array, 1.0 - h_array, mode="full") * dx))


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
        h_array = h_array * (target_sum / current_sum)
        if np.any(h_array < 0) or np.any(h_array > 1):
            raise ErdosEvaluationError("After normalization, h(x) is not in [0, 1]")
    computed_c5 = compute_c5(h_array, int(n_points))
    if not np.isfinite(computed_c5):
        raise ErdosEvaluationError(f"Computed C5 is not finite: {computed_c5}")
    if not np.isclose(computed_c5, float(c5_achieved), atol=1e-4):
        raise ErdosEvaluationError(f"C5 mismatch: reported {c5_achieved:.6f}, computed {computed_c5:.6f}")
    return computed_c5


def verify_program_result(output: Any) -> VerificationResult:
    try:
        h_values, c5_bound, n_points = output
        raw_score = verify_c5_solution(h_values, c5_bound, n_points)
    except Exception as exc:
        return VerificationResult(
            reward=0.0,
            raw_score=None,
            valid=False,
            status="invalid",
            message=str(exc),
            artifacts={},
        )
    reward = 1.0 / (1e-8 + raw_score)
    return VerificationResult(
        reward=reward,
        raw_score=raw_score,
        valid=True,
        status="valid",
        message=f"C5 bound: {raw_score:.6f}",
        artifacts={},
    )


def verify_erdos_program_text(
    text: str,
    *,
    timeout_s: int,
    seed: int = 42,
    budget_s: int = 1,
    n_points: int = 256,
) -> VerificationResult:
    sandbox = evaluate_python_code(
        text,
        timeout_s=timeout_s,
        seed=seed,
        budget_s=budget_s,
        n_points=n_points,
    )
    if sandbox.error is not None:
        status = "timeout" if sandbox.error.startswith("Timed out") else "invalid"
        return VerificationResult(
            reward=0.0,
            raw_score=None,
            valid=False,
            status=status,
            message=sandbox.error,
            artifacts={"code": text},
        )
    result = verify_program_result(sandbox.output)
    result.artifacts["code"] = text
    return result

