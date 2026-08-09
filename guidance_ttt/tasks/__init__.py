"""Task definitions for Guidance-TTT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

from guidance_ttt.state import LibraryNode, VerificationResult
from guidance_ttt.tasks.erdos import ERDOS_PROBLEM_PROMPT
from guidance_ttt.tasks.erdos import create_root_node as create_erdos_root_node
from guidance_ttt.tasks.polyomino import POLYOMINO_PROBLEM_PROMPT
from guidance_ttt.tasks.polyomino import create_root_node as create_polyomino_root_node
from guidance_ttt.tasks.trimul import TRIMUL_PROBLEM_PROMPT
from guidance_ttt.tasks.trimul import (
    create_root_node as create_trimul_root_node,
)
from guidance_ttt.tasks.vliw_kernel import (
    VLIW_BASELINE_SOLUTION,
    VLIW_BASELINE_SUMMARY,
    VLIW_KERNEL_PROBLEM_PROMPT,
)
from guidance_ttt.tasks.vliw_kernel import (
    create_root_node as create_vliw_root_node,
)
from guidance_ttt.verifier.erdos import verify_erdos_solution_text
from guidance_ttt.verifier.polyomino import extract_cpp_solution_code, verify_polyomino_solution_text
from guidance_ttt.verifier.sandbox import extract_python_code
from guidance_ttt.verifier.trimul import extract_trimul_solution_code, verify_trimul_solution_text
from guidance_ttt.verifier.vliw_kernel import extract_vliw_solution_code, verify_vliw_solution_text

ScoreDirection = Literal["min", "max"]


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    problem_prompt: str
    solution_language: str
    execution_solution_contract: str
    guidance_mechanism_constraint: str
    score_direction: ScoreDirection
    raw_score_label: str
    create_root_node: Callable[..., LibraryNode]
    verifier: Callable[..., VerificationResult]
    solution_extractor: Callable[[str], str | None]
    guidance_objective: Callable[[float | None], str]
    scratch_bootstrap_constraint: str | None = None
    bootstrap_solution: str | None = None
    bootstrap_summary: str | None = None

    def verify_execution_text(
        self,
        text: str,
        *,
        timeout_s: int,
        config: dict[str, Any] | None = None,
    ) -> VerificationResult:
        return self.verifier(text, timeout_s=timeout_s, config=config or {})

def _verify_erdos(text: str, *, timeout_s: int, config: dict[str, Any] | None = None) -> VerificationResult:
    return verify_erdos_solution_text(text, timeout_s=timeout_s)


def _verify_polyomino(text: str, *, timeout_s: int, config: dict[str, Any] | None = None) -> VerificationResult:
    frontiercs_config = dict(config or {})
    problem_id = str(frontiercs_config.pop("problem_id", "0"))
    return verify_polyomino_solution_text(text, problem_id=problem_id, config=frontiercs_config)


def _verify_vliw(text: str, *, timeout_s: int, config: dict[str, Any] | None = None) -> VerificationResult:
    return verify_vliw_solution_text(text, timeout_s=timeout_s, config=config or {})


def _verify_trimul(text: str, *, timeout_s: int, config: dict[str, Any] | None = None) -> VerificationResult:
    return verify_trimul_solution_text(text, timeout_s=timeout_s, config=config or {})


def _erdos_objective(target: float | None) -> str:
    _ = target
    return (
        "Your task is to provide the next **evolutionary guidance** to reach a higher reward score "
        "by lowering raw C5. Lower raw C5 is better."
    )


def _polyomino_objective(target: float | None) -> str:
    _ = target
    return (
        "Your task is to provide the next **evolutionary guidance** to reach a higher FrontierCS "
        "score. Higher FrontierCS score is better."
    )


def _vliw_objective(target: float | None) -> str:
    _ = target
    return (
        "Your task is to provide the next **evolutionary guidance** that preserves correctness while reducing "
        "the selected candidate's simulator cycles. Lower raw cycles and higher normalized EdgeBench reward "
        "are better."
    )


def _trimul_objective(target: float | None) -> str:
    _ = target
    return (
        "Your task is to provide the next **evolutionary guidance** that preserves numerical correctness while "
        "reducing the selected TriMul candidate's geometric-mean H100 runtime. Lower runtime in microseconds and "
        "higher inverse-runtime reward are better."
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
        guidance_mechanism_constraint=(
            "Every proposed mechanism must be implementable inside one self-contained Python candidate using only "
            "information and resources available in the task's permitted runtime. Do not rely on offline training "
            "data, hidden benchmark access, external models or APIs, learned weights that are not supplied, or "
            "unavailable precomputation."
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
        guidance_mechanism_constraint=(
            "Every proposed mechanism must be implementable inside one self-contained C++17 program using only "
            "the current input instance. Do not rely on offline training data, benchmark access, external models, "
            "APIs, learned weights, or unavailable precomputation."
        ),
        score_direction="max",
        raw_score_label="FrontierCS score",
        create_root_node=create_polyomino_root_node,
        verifier=_verify_polyomino,
        solution_extractor=extract_cpp_solution_code,
        guidance_objective=_polyomino_objective,
    ),
    "vliw_kernel_optimization": TaskSpec(
        task_id="vliw_kernel_optimization",
        problem_prompt=VLIW_KERNEL_PROBLEM_PROMPT,
        solution_language="python",
        execution_solution_contract=(
            "The <solution> block must contain one complete replacement solution.py in a ```python fenced block. "
            "It must preserve the required KernelBuilder interface and may modify no other file."
        ),
        guidance_mechanism_constraint=(
            "Every proposal must be implementable entirely in solution.py against the supplied public VLIW/SIMD "
            "ISA. Guidance may specify compiler-level mechanisms such as dependency-aware bundle packing, SIMD "
            "batching, software pipelining, unrolling, scratch allocation, and memory/compute overlap, but must not "
            "write Python code or concrete instruction arrays. Do not rely on external models, APIs, network access, "
            "hidden cases, seed-specific constants, or changes to protected benchmark files."
        ),
        score_direction="min",
        raw_score_label="Simulator cycles",
        create_root_node=create_vliw_root_node,
        verifier=_verify_vliw,
        solution_extractor=extract_vliw_solution_code,
        guidance_objective=_vliw_objective,
        bootstrap_solution=VLIW_BASELINE_SOLUTION,
        bootstrap_summary=VLIW_BASELINE_SUMMARY,
    ),
    "trimul": TaskSpec(
        task_id="trimul",
        problem_prompt=TRIMUL_PROBLEM_PROMPT,
        solution_language="python",
        execution_solution_contract=(
            "The <solution> block must contain one complete replacement submission.py in a ```python fenced "
            "block. It must define custom_kernel(data), contain at least one @triton.jit kernel, support every "
            "official shape and mask/distribution case, and return a float32 output tensor."
        ),
        guidance_mechanism_constraint=(
            "Every proposal must be implementable in one self-contained submission.py using PyTorch and Triton "
            "3.3.1 on an H100. Guidance may propose fusion boundaries, tensor layouts, precision choices, tiling, "
            "persistent scheduling, contraction decomposition, launch reduction, or shape-specialized dispatch, "
            "but must remain concrete enough for the executor to implement. Scope each attempt to one primary "
            "optimization mechanism, or one tightly coupled change set, and preserve every unrelated stage of the "
            "verified parent. Do not bundle independent speculative rewrites across multiple pipeline stages. Do "
            "not rely on offline training, external models or APIs, hidden-case access, generated binary artifacts, "
            "or hardware other than the stated H100 environment."
        ),
        score_direction="min",
        raw_score_label="H100 geometric-mean runtime (us)",
        create_root_node=create_trimul_root_node,
        verifier=_verify_trimul,
        solution_extractor=extract_trimul_solution_code,
        guidance_objective=_trimul_objective,
        scratch_bootstrap_constraint=(
            "Prioritize correctness and Triton compilability over speed for this initial seed. Implement the public "
            "PyTorch reference equations directly for input normalization, all projections, mask application, left "
            "and right sigmoid gating, torch.einsum contraction, output normalization, and final projection. The "
            "contraction must be written exactly as torch.einsum(\"bikd,bjkd->bijd\", left, right), must return "
            "[B,N,N,H], and must not be followed by a compensating reshape. Call torch.nn.functional.layer_norm "
            "with the supplied weights and bias for both normalizations rather than implementing LayerNorm "
            "manually. Use exactly one simple, functionally required Triton kernel only for multiplying the "
            "normalized contraction by sigmoid(out_gate). Make both tensors contiguous, flatten them, launch a "
            "one-dimensional grid over "
            "their shared numel, and use flat unit-stride offsets with a final bounds mask; do not express 4D strides "
            "inside this seed kernel. Do not write a custom contraction kernel, a custom mask kernel, extra Triton "
            "kernels, placeholder operations, dead kernels, unused speculative expressions, or code that your own "
            "comments say should be replaced. Before returning, audit that every emitted statement compiles under "
            "Triton 3.3.1 and that all public shapes, binary masks, normal/Cauchy inputs, and float32 output are "
            "supported."
        ),
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
