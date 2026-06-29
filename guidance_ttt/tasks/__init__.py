"""Task definitions for Guidance-TTT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

from guidance_ttt.state import LibraryEntry, LibraryNode, VerificationResult
from guidance_ttt.tasks.erdos import ERDOS_PROBLEM_PROMPT, create_root_node as create_erdos_root_node
from guidance_ttt.tasks.polyomino import POLYOMINO_PROBLEM_PROMPT, create_root_node as create_polyomino_root_node
from guidance_ttt.verifier.erdos import verify_erdos_solution_text
from guidance_ttt.verifier.polyomino import extract_cpp_solution_code, verify_polyomino_solution_text
from guidance_ttt.verifier.sandbox import extract_python_code

ScoreDirection = Literal["min", "max"]


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    problem_prompt: str
    solution_language: str
    execution_solution_contract: str
    score_direction: ScoreDirection
    raw_score_label: str
    create_root_node: Callable[..., LibraryNode]
    verifier: Callable[..., VerificationResult]
    solution_extractor: Callable[[str], str | None]
    guidance_objective: Callable[[float | None], str]

    def verify_execution_text(
        self,
        text: str,
        *,
        timeout_s: int,
        config: dict[str, Any] | None = None,
    ) -> VerificationResult:
        return self.verifier(text, timeout_s=timeout_s, config=config or {})

    def best_target(
        self,
        selected_node: LibraryNode,
        selected_entry: LibraryEntry | None,
        global_best_entries: list[LibraryEntry],
    ) -> float | None:
        candidates = [
            entry.verifier_raw_score
            for entry in [selected_entry, *global_best_entries]
            if entry is not None and entry.verifier_status == "valid" and entry.verifier_raw_score is not None
        ]
        if not candidates:
            return selected_node.raw_score
        if self.score_direction == "max":
            return max(float(score) for score in candidates)
        return min(float(score) for score in candidates)


def _verify_erdos(text: str, *, timeout_s: int, config: dict[str, Any] | None = None) -> VerificationResult:
    return verify_erdos_solution_text(text, timeout_s=timeout_s)


def _verify_polyomino(text: str, *, timeout_s: int, config: dict[str, Any] | None = None) -> VerificationResult:
    frontiercs_config = dict(config or {})
    problem_id = str(frontiercs_config.pop("problem_id", "0"))
    return verify_polyomino_solution_text(text, problem_id=problem_id, config=frontiercs_config)


def _erdos_objective(target: float | None) -> str:
    return (
        "Your task is to provide the next **evolutionary guidance** to beat the current visible "
        f"target raw score ({target}). Lower raw C5 is better."
    )


def _polyomino_objective(target: float | None) -> str:
    return (
        "Your task is to provide the next **evolutionary guidance** to beat the current visible "
        f"FrontierCS score target ({target}). Higher FrontierCS score is better."
    )


def _extract_python_solution(text: str) -> str | None:
    return extract_python_code(text)


_TASKS: dict[str, TaskSpec] = {
    "erdos_min_overlap": TaskSpec(
        task_id="erdos_min_overlap",
        problem_prompt=ERDOS_PROBLEM_PROMPT,
        solution_language="python",
        execution_solution_contract=(
            "The <solution> block must contain one complete executable Python candidate in a ```python fenced block."
        ),
        score_direction="min",
        raw_score_label="Raw C5",
        create_root_node=create_erdos_root_node,
        verifier=_verify_erdos,
        solution_extractor=_extract_python_solution,
        guidance_objective=_erdos_objective,
    ),
    "polyomino_packing": TaskSpec(
        task_id="polyomino_packing",
        problem_prompt=POLYOMINO_PROBLEM_PROMPT,
        solution_language="cpp",
        execution_solution_contract=(
            "The <solution> block must contain one complete C++17 program in a ```cpp fenced block. "
            "It must read the Polyomino Packing instance from stdin and write the placement to stdout."
        ),
        score_direction="max",
        raw_score_label="FrontierCS score",
        create_root_node=create_polyomino_root_node,
        verifier=_verify_polyomino,
        solution_extractor=extract_cpp_solution_code,
        guidance_objective=_polyomino_objective,
    ),
}


def get_task_spec(task_id: str | None) -> TaskSpec:
    normalized = task_id or "erdos_min_overlap"
    try:
        return _TASKS[normalized]
    except KeyError as exc:
        known = ", ".join(sorted(_TASKS))
        raise KeyError(f"Unknown Guidance-TTT task: {normalized}. Known tasks: {known}") from exc


def task_ids() -> list[str]:
    return sorted(_TASKS)
