from guidance_ttt.prompts import (
    build_execution_prompt,
    build_guidance_prompt,
    build_summary_prompt,
)
from guidance_ttt.state import LibraryEntry, LibraryNode


def _node() -> LibraryNode:
    return LibraryNode(
        id="node-1",
        problem_id="erdos",
        timestep=3,
        entry_id="entry-1",
        value=2.5,
        raw_score=0.4,
        visits=2,
        parent_id="root",
        children=[],
        metadata={},
    )


def _entry() -> LibraryEntry:
    return LibraryEntry(
        id="entry-1",
        parent_id="root",
        problem_id="erdos",
        timestep=3,
        guidance="preserve symmetry, tune local search",
        execution_thinking="used projected gradient",
        solution="def run(seed=42, budget_s=1, **kwargs):\n    return ([0.5, 0.5], 0.5, 2)",
        verifier_reward=2.5,
        verifier_raw_score=0.4,
        verifier_status="valid",
        verifier_message="C5 bound: 0.400000",
        summary="projected gradient improved stability",
        reusable_idea="project after each perturbation",
        failure_mode=None,
        metadata={},
    )


def test_guidance_prompt_attaches_selected_library_node_but_not_full_solution():
    prompt = build_guidance_prompt(
        problem_prompt="Find better C5",
        selected_node=_node(),
        selected_entry=_entry(),
        global_best_entries=[],
        local_failure_entries=[],
    )

    assert "Find better C5" in prompt.user
    assert "<selected_library_node>" in prompt.user
    assert "projected gradient improved stability" in prompt.user
    assert "preserve symmetry" in prompt.user
    assert "def run(seed=42" not in prompt.user
    assert "<guidance>" in prompt.user


def test_execution_prompt_attaches_same_library_node_full_solution_and_guidance():
    prompt = build_execution_prompt(
        problem_prompt="Find better C5",
        selected_node=_node(),
        selected_entry=_entry(),
        guidance="Try deterministic coordinate descent.",
    )

    assert "Find better C5" in prompt.user
    assert "Try deterministic coordinate descent." in prompt.user
    assert "Previous full solution code" in prompt.user
    assert "def run(seed=42" in prompt.user
    assert "<execution_thinking>" in prompt.user


def test_summary_prompt_contains_guidance_execution_solution_and_reward():
    prompt = build_summary_prompt(
        selected_entry=_entry(),
        guidance="Try deterministic coordinate descent.",
        execution_thinking="I changed the optimizer.",
        solution="def run(): return None",
        reward=0.0,
        raw_score=None,
        status="invalid",
        message="shape mismatch",
    )

    assert "Try deterministic coordinate descent." in prompt.user
    assert "I changed the optimizer." in prompt.user
    assert "def run()" in prompt.user
    assert "shape mismatch" in prompt.user
    assert "<summary>" in prompt.user

