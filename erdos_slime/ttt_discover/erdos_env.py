from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from erdos_slime.ttt_discover.state import DiscoveryState


class ErdosEvaluationError(ValueError):
    pass


@dataclass
class ErdosScore:
    reward: float
    raw_score: float
    state: DiscoveryState
    message: str


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
            raise ErdosEvaluationError(
                f"After normalization, h(x) is not in [0, 1]. Range: [{h_array.min()}, {h_array.max()}]"
            )

    dx = 2.0 / int(n_points)
    computed_c5 = float(np.max(np.correlate(h_array, 1.0 - h_array, mode="full") * dx))
    if not np.isfinite(computed_c5):
        raise ErdosEvaluationError(f"Computed C5 is not finite: {computed_c5}")
    if not np.isclose(computed_c5, float(c5_achieved), atol=1e-4):
        raise ErdosEvaluationError(f"C5 mismatch: reported {c5_achieved:.6f}, computed {computed_c5:.6f}")
    return computed_c5


def evaluate_erdos_solution(result: tuple[Any, float, int]) -> float:
    h_values, c5_bound, n_points = result
    return verify_c5_solution(h_values, c5_bound, n_points)


def score_erdos_result(
    result: tuple[Any, float, int],
    *,
    code: str,
    timestep: int,
    stdout: str = "",
) -> ErdosScore:
    h_values, c5_bound, n_points = result
    raw_score = evaluate_erdos_solution((h_values, c5_bound, n_points))
    h_array = np.asarray(h_values, dtype=np.float64)
    state = DiscoveryState(
        timestep=timestep,
        value=-raw_score,
        raw_score=raw_score,
        code=code,
        construction=h_array.tolist(),
        observation=stdout,
    )
    reward = 1.0 / (1e-8 + raw_score)
    return ErdosScore(reward=reward, raw_score=raw_score, state=state, message=f"C5 bound: {raw_score:.6f}")


def create_random_initial_state(seed: int | None = None, n_points_min: int = 40, n_points_max: int = 100) -> DiscoveryState:
    rng = np.random.default_rng(seed)
    n_points = int(rng.integers(n_points_min, n_points_max))
    construction = np.ones(n_points, dtype=np.float64) * 0.5
    perturbation = rng.uniform(-0.4, 0.4, n_points)
    perturbation = perturbation - np.mean(perturbation)
    construction = np.clip(construction + perturbation, 0.0, 1.0)
    construction = construction * ((n_points / 2.0) / np.sum(construction))
    c5_bound = verify_c5_solution(
        construction,
        float(np.max(np.correlate(construction, 1.0 - construction, mode="full") * (2.0 / n_points))),
        n_points,
    )
    return DiscoveryState(
        timestep=-1,
        value=-c5_bound,
        raw_score=c5_bound,
        code="",
        construction=construction.tolist(),
    )


def build_erdos_prompt(
    state: DiscoveryState,
    *,
    budget_s: int,
    cpus: int,
    target_c5: float = 0.3808,
) -> str:
    raw_score = state.raw_score if state.raw_score is not None else -float(state.value)
    code_context = (
        "Here is the last code we ran:\n```python\n" + state.code + "\n```"
        if state.code.strip()
        else "No previous code available."
    )
    construction_context = ""
    if state.construction:
        construction_context = (
            f"\nYou may start from the current construction through the pre-imported "
            f"`initial_h_values` global variable (n={len(state.construction)} samples).\n"
        )

    return f"""You are an expert in harmonic analysis, numerical optimization, and mathematical discovery.
Your task is to find an improved upper bound for the Erdos minimum overlap problem constant C5.

## Problem

Find a step function h: [0, 2] -> [0, 1] that minimizes:

C5 = max_k integral h(x)(1 - h(x+k)) dx

Constraints:
1. h(x) is in [0, 1] for all x
2. integral_0^2 h(x) dx = 1

Represent h as n_points samples over [0, 2]. The evaluator computes:
max(np.correlate(h, 1-h, mode="full") * (2.0 / n_points)).

## Budget & Resources
- Time budget: {budget_s}s
- CPUs: {cpus}

## Rules
- Define `run(seed=42, budget_s={budget_s}, **kwargs)` returning `(h_values, c5_bound, n_points)`.
- Use numpy, scipy, cvxpy, and math only.
- No filesystem or network IO.
- `evaluate_erdos_solution()` and `initial_h_values` are pre-imported.
- Return the best solution found before timeout.

Lower C5 values are better. Current C5 bound: {raw_score:.6f}
Target: {target_c5:.6f}
{construction_context}
{code_context}

Write Python code in one final ```python code block.
"""
