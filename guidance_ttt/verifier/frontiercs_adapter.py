from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class FrontierCSEnvironmentError(RuntimeError):
    """Raised when the external FrontierCS/go-judge environment is unavailable."""


@dataclass(frozen=True)
class FrontierCSResult:
    valid: bool
    score: float | None
    message: str
    artifacts: dict[str, Any] = field(default_factory=dict)


def evaluate_cpp_solution(
    code: str,
    *,
    problem_id: str,
    config: dict[str, Any] | None = None,
) -> FrontierCSResult:
    config = dict(config or {})
    base_dir_value = config.pop("base_dir", None)
    base_dir = Path(str(base_dir_value)).expanduser().resolve() if base_dir_value else None
    try:
        from frontier_cs import SingleEvaluator
    except Exception as exc:
        raise FrontierCSEnvironmentError(
            "frontier_cs is not installed or importable. Install FrontierCS outside the guidance repo "
            "and ensure the Python environment can import frontier_cs."
        ) from exc

    try:
        evaluator = SingleEvaluator(register_cleanup=False, base_dir=base_dir)
        result = evaluator.evaluate("algorithmic", problem_id=str(problem_id), code=code)
    except Exception as exc:
        raise FrontierCSEnvironmentError(
            "FrontierCS evaluation failed. Check that Docker/go-judge is running and that the "
            f"FrontierCS environment is configured correctly: {exc}"
        ) from exc

    success = bool(getattr(result, "success", False))
    raw_score = getattr(result, "score", None)
    score = None if raw_score is None else float(raw_score)
    message = str(getattr(result, "message", "") or ("accepted" if success else "Evaluation failed"))
    artifacts = {
        "problem_id": str(problem_id),
        "frontiercs_config": config,
    }
    if base_dir is not None:
        artifacts["frontiercs_base_dir"] = str(base_dir)
    for attr in ("stdout", "stderr", "details"):
        if hasattr(result, attr):
            artifacts[attr] = getattr(result, attr)
    return FrontierCSResult(valid=success, score=score, message=message, artifacts=artifacts)
