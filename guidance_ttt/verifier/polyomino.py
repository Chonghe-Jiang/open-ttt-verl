from __future__ import annotations

import re
from typing import Any

from guidance_ttt.state import VerificationResult
from guidance_ttt.verifier.frontiercs_adapter import FrontierCSEnvironmentError, evaluate_cpp_solution


def extract_cpp_solution_code(text: str) -> str | None:
    solution_match = re.search(r"<solution>\s*([\s\S]*?)\s*</solution>", text)
    if solution_match:
        return extract_cpp_solution_code(solution_match.group(1))
    matches = list(re.finditer(r"```(?:cpp|c\+\+|C\+\+)\s*([\s\S]*?)\s*```", text))
    if matches:
        code = matches[-1].group(1).strip()
        return code or None
    return None


def verify_polyomino_solution_text(
    text: str,
    *,
    problem_id: str = "0",
    config: dict[str, Any] | None = None,
) -> VerificationResult:
    code = extract_cpp_solution_code(text)
    if code is None:
        return VerificationResult(
            reward=0.0,
            raw_score=None,
            valid=False,
            status="parse_error",
            message="No C++17 code block found in <solution>",
            artifacts={},
        )
    try:
        result = evaluate_cpp_solution(code, problem_id=problem_id, config=config or {})
    except FrontierCSEnvironmentError as exc:
        return VerificationResult(
            reward=0.0,
            raw_score=None,
            valid=False,
            status="environment_error",
            message=str(exc),
            artifacts={"code": code},
        )
    artifacts = {"code": code}
    artifacts.update(result.artifacts or {})
    if not result.valid:
        return VerificationResult(
            reward=0.0,
            raw_score=None,
            valid=False,
            status="invalid",
            message=result.message,
            artifacts=artifacts,
        )
    score = 0.0 if result.score is None else float(result.score)
    return VerificationResult(
        reward=score,
        raw_score=score,
        valid=True,
        status="valid",
        message=result.message or f"FrontierCS score: {score:.2f}",
        artifacts=artifacts,
    )
