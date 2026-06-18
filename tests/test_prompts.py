import re

from guidance_ttt.prompts import (
    build_execution_prompt,
    build_guidance_prompt,
    extract_guidance_or_format_error,
    extract_tag,
    extract_tag_or_none,
    extract_text_outside_tag,
)
from guidance_ttt.state import LibraryEntry, LibraryNode
from guidance_ttt.verifier.erdos import verify_erdos_solution_text


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
    assert "You may think first" in prompt.system
    assert "final submitted guidance" in prompt.system
    assert "Thinking is allowed" in prompt.user
    assert "only text outside any <think>...</think> block" in prompt.user
    assert "target to beat" in prompt.user
    assert "constant h[i] = 0.5 baseline" in prompt.user
    assert "What to change to beat raw score 0.4" in prompt.user
    assert "controlled improvement" in prompt.user
    assert "Use the best valid h profile as the initialization" in prompt.user
    assert "Use n_points in {9, 11, 13, 15, 17, 21, 25}" in prompt.user
    assert "mirror/complement symmetry" in prompt.user
    assert "same verifier normalization" in prompt.user
    assert "Avoid vague RL-reward advice" in prompt.user


def test_guidance_prompt_attaches_global_best_and_local_failure_history():
    global_best = _entry()
    global_best.id = "best-entry"
    global_best.summary = "best history used mirror minimax"
    global_best.guidance = "preserve best mirror profile"
    global_best.reusable_idea = "reuse best projection repair"
    local_failure = _entry()
    local_failure.id = "failure-entry"
    local_failure.summary = "failed because sum drifted"
    local_failure.guidance = "bad alternating pattern"
    local_failure.verifier_status = "invalid"
    local_failure.failure_mode = "invalid"
    local_failure.verifier_message = "sum(h) must equal n_points / 2"

    prompt = build_guidance_prompt(
        problem_prompt="Find better C5",
        selected_node=_node(),
        selected_entry=_entry(),
        global_best_entries=[global_best],
        local_failure_entries=[local_failure],
    )

    assert "<global_best>" in prompt.user
    assert "best-entry" in prompt.user
    assert "best history used mirror minimax" in prompt.user
    assert "reuse best projection repair" in prompt.user
    assert "<local_failures>" in prompt.user
    assert "failure-entry" in prompt.user
    assert "failed because sum drifted" in prompt.user
    assert "sum(h) must equal n_points / 2" in prompt.user
    assert "Failure mode: invalid" in prompt.user


def test_guidance_prompt_targets_controlled_improvement_from_best_valid_entry():
    global_best = _entry()
    global_best.id = "best-entry"
    global_best.verifier_raw_score = 0.3821438682282878
    global_best.verifier_status = "valid"
    global_best.summary = "best valid mirror profile"
    global_best.guidance = "preserve current best"

    prompt = build_guidance_prompt(
        problem_prompt="Find better C5",
        selected_node=_node(),
        selected_entry=_entry(),
        global_best_entries=[global_best],
        local_failure_entries=[],
    )

    assert "Preserve the current best valid construction with raw_score = 0.3821438682282878" in prompt.user
    assert "Do not restart from the constant baseline" in prompt.user
    assert "Use the best valid h profile as the initialization" in prompt.user
    assert "Search only small deterministic" in prompt.user
    assert "Optimize only the independent half of the variables" in prompt.user
    assert "Include known information directly in the guidance" in prompt.user
    assert "best valid solution excerpt" in prompt.user
    assert "Use n_points in {9, 11, 13, 15, 17, 21, 25}" in prompt.user
    assert "Target: strictly improve over 0.3821438682282878, not merely beat 0.5" in prompt.user
    assert "Avoid GPU tensors, large correlation matrices, or unbounded minimax solvers" not in prompt.user
    assert "Do not ask the execution model to reinvent minimax from scratch" not in prompt.user


def test_execution_prompt_attaches_same_library_node_solution_excerpt_and_guidance():
    prompt = build_execution_prompt(
        problem_prompt="Find better C5",
        selected_node=_node(),
        selected_entry=_entry(),
        guidance="Try deterministic coordinate descent.",
    )

    assert "Find better C5" in prompt.user
    assert "Try deterministic coordinate descent." in prompt.user
    assert "Previous solution code excerpt" in prompt.user
    assert "def run(seed=42" in prompt.user
    assert "<execution_thinking>" in prompt.user
    assert "<summary>" in prompt.user
    assert "Risk / possible failure mode" in prompt.user
    assert "Do not claim verifier" in prompt.user
    assert "success because the verifier has not run yet" in prompt.user
    assert "Target: produce a valid candidate with raw C5 lower than 0.4" in prompt.user
    assert "permits fractional, non-binary h values" in prompt.user
    assert "compute the actual c5_bound" in prompt.user
    assert "SLSQP" in prompt.user
    assert "deterministic projected local search" in prompt.user
    assert "project or adjust h" in prompt.user
    assert "Use n_points in {9, 11, 13, 15, 17, 21, 25}" in prompt.user
    assert "strict improvement over the current best raw C5" in prompt.user
    assert "box-constrained projection" in prompt.user
    assert "sum(h) == n_points / 2 to verifier tolerance" in prompt.user
    assert "project_to_box_sum" in prompt.user
    assert "return [float(x) for x in h], float(c5_bound), int(n_points)" in prompt.user
    assert "Avoid GPU tensors, large correlation matrices, or unbounded minimax solvers" not in prompt.user
    assert "invent a fresh minimax construction" not in prompt.user


def test_execution_prompt_attaches_global_best_valid_solution_excerpt():
    global_best = _entry()
    global_best.id = "best-entry"
    global_best.verifier_raw_score = 0.3821438682282878
    global_best.solution = "def run(seed=42, budget_s=1, **kwargs):\n    return ([0.4, 0.6], 0.3821438682282878, 2)"

    prompt = build_execution_prompt(
        problem_prompt="Find better C5",
        selected_node=_node(),
        selected_entry=None,
        global_best_entries=[global_best],
        guidance="Preserve current best and perturb locally.",
    )

    assert "<global_best_valid_solution>" in prompt.user
    assert "Entry id: best-entry" in prompt.user
    assert "Raw score: 0.3821438682282878" in prompt.user
    assert "def run(seed=42" in prompt.user
    assert "return ([0.4, 0.6], 0.3821438682282878, 2)" in prompt.user
    assert "Initialize from this global best valid solution when available" in prompt.user


def test_execution_prompt_known_good_minimax_template_is_verifier_valid():
    prompt = build_execution_prompt(
        problem_prompt="Find better C5",
        selected_node=_node(),
        selected_entry=_entry(),
        guidance="Use the minimax template.",
    )
    match = re.search(r"<known_good_minimax_template>\s*```python\n([\s\S]*?)\n```\s*</known_good_minimax_template>", prompt.user)

    assert match is not None
    verification = verify_erdos_solution_text(f"```python\n{match.group(1)}\n```", timeout_s=20)

    assert verification.valid is True
    assert verification.raw_score < 0.382


def test_execution_text_contains_parseable_summary_tag():
    execution_text = """<execution_thinking>
I changed the optimizer.
</execution_thinking>

```python
def run(): return None
```

<summary>
Outcome hypothesis: test
Reusable idea: coordinate descent
Risk / possible failure mode: shape mismatch
What future guidance should preserve: symmetry
What future guidance should change: step schedule
</summary>"""

    assert extract_tag(execution_text, "summary").startswith("Outcome hypothesis")


def test_guidance_extraction_requires_explicit_guidance_tag():
    text = "<guidance>\nHypothesis: try coordinate descent\n</guidance>"

    guidance, ok = extract_guidance_or_format_error(text)

    assert ok is True
    assert guidance == "Hypothesis: try coordinate descent"


def test_guidance_extraction_falls_back_to_text_outside_think_tags():
    text = """<think>
long hidden reasoning
</think>

Hypothesis: use projected coordinate descent.
Plan:
1. keep the mass constraint
2. perturb paired coordinates
"""

    guidance, ok = extract_guidance_or_format_error(text)

    assert ok is False
    assert guidance.startswith("Hypothesis: use projected coordinate descent.")
    assert "<think>" not in guidance
    assert "long hidden reasoning" not in guidance
    assert extract_tag_or_none(text, "guidance") is None


def test_guidance_extraction_returns_format_error_when_only_think_text_exists():
    text = "<think>\nlong hidden reasoning\n</think>"

    guidance, ok = extract_guidance_or_format_error(text)

    assert ok is False
    assert "formatting failure" in guidance
    assert "long hidden reasoning" not in guidance


def test_extract_text_outside_tag_drops_unclosed_think_block():
    text = "submitted guidance\n<think>\nunfinished hidden reasoning"

    assert extract_text_outside_tag(text, "think") == "submitted guidance"
