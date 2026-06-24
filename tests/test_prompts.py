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
        summary=(
            "Execution Interpretation\n"
            "Projected gradient improved stability.\n\n"
            "Implemented Algorithm\n"
            "Use n_points=19, project_to_box_sum, coordinate step 2e-4, "
            "random walk step size 1e-3 for 2000 steps, and optional SLSQP maxiter=300.\n\n"
            "Next Guidance Delta\n"
            "Keep the projection repair and reduce only paired coordinates that raise C5."
        ),
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
    assert "Projected gradient improved stability" in prompt.user
    assert "preserve symmetry" in prompt.user
    assert "def run(seed=42" not in prompt.user
    assert "<guidance>" in prompt.user
    assert "You are the Guidance Model" in prompt.system
    assert "evolutionary guidance" in prompt.system
    assert "Do not write final code" in prompt.system
    assert "escape local optima" in prompt.system
    assert "# Objective" in prompt.user
    assert "next **evolutionary guidance**" in prompt.user
    assert "# Evolutionary Guidelines" in prompt.user
    assert "Analyze History, Do Not Repeat It" in prompt.user
    assert "High-Level Mutations, No Low-Level Details" in prompt.user
    assert "Strict Separation of Thought and Action" in prompt.user
    assert "<think>" in prompt.user
    assert "</think>" in prompt.user
    assert "The `<think>` block" in prompt.user
    assert "The `<guidance>` block" in prompt.user
    assert "Evolutionary Mutation" in prompt.user
    assert "Directional Search Strategy" in prompt.user
    assert "Progress Target" in prompt.user
    assert "Do not write code or micromanage hyperparameters" in prompt.user
    assert "coordinate step 2e-4" in prompt.user
    assert "random walk step size 1e-3 for 2000 steps" in prompt.user
    assert "SLSQP maxiter=300" in prompt.user
    assert "current best valid raw score (0.4)" in prompt.user
    assert "successful mutation from 0.4 towards 0.4 or lower" in prompt.user


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

    assert "current best valid raw score (0.3821438682282878)" in prompt.user
    assert "Lower raw C5 is better" in prompt.user
    assert "why the current profile plateaued" in prompt.user
    assert "conceptual algorithmic shifts" in prompt.user
    assert "structural relaxations" in prompt.user
    assert "novel search topologies" in prompt.user
    assert "introducing a new mathematical constraint" in prompt.user
    assert "hybridizing optimization frameworks" in prompt.user
    assert "Do not write code or micromanage hyperparameters" in prompt.user
    assert "successful mutation from 0.4 towards 0.3821438682282878 or lower" in prompt.user
    assert "The preferred submitted guidance" not in prompt.user
    assert "Preserve the current best valid construction" not in prompt.user
    assert "SLSQP again" not in prompt.user
    assert "Avoid GPU tensors, large correlation matrices, or unbounded minimax solvers" not in prompt.user
    assert "Do not ask the execution model to reinvent minimax from scratch" not in prompt.user


def test_guidance_prompt_prefers_verified_profile_artifacts_over_summary_head_tail():
    best = _entry()
    best.id = "best-entry"
    best.verifier_raw_score = 0.3812435631313583
    best.verifier_reward = 2.6229950364447134
    best.summary = (
        "Empirical Outcome\n"
        "Verified returned profile: n_points=19, c5_bound=0.3812435631313583, "
        "h length=19, head=[1.0, 0.9872194239853203], tail=[0.9951851401967569, 0.9999965067437544]"
    )
    best.metadata = {
        "verification_artifacts": {
            "n_points": 19,
            "c5_bound": 0.3812435631313583,
            "h_values": [
                1.0,
                0.9872194239853203,
                0.6538357213148515,
                0.2649608397726771,
                0.791971283002877,
                0.22245751838093442,
                0.43596,
                0.31250,
                0.01329,
                0.10448,
                0.04501,
                0.31191,
                0.43588,
                0.22262,
                0.79183,
                0.2681266810637105,
                0.6427089545574667,
                0.9951851401967569,
                0.9999965067437544,
            ],
        }
    }

    prompt = build_guidance_prompt(
        problem_prompt="Find better C5",
        selected_node=_node(),
        selected_entry=best,
        global_best_entries=[best],
        local_failure_entries=[],
    )

    assert "Verified returned profile artifacts (authoritative initialization)" in prompt.user
    assert "raw C5=0.3812435631313583" in prompt.user
    assert "h=[1.0, 0.9872194239853203" in prompt.user
    assert "0.9999965067437544]" in prompt.user
    assert "why the current profile plateaued" in prompt.user


def test_guidance_prompt_root_defaults_to_known_good_target_and_profile_facts():
    root_node = LibraryNode(
        id="root",
        problem_id="erdos",
        timestep=0,
        entry_id=None,
        value=2.0,
        raw_score=0.5,
        visits=0,
        parent_id=None,
        children=[],
        metadata={},
    )
    code_entry = _entry()
    code_entry.summary = "Execution Interpretation\n" + ("long inherited context. " * 80) + """

Implemented Algorithm
```python
import numpy as np
n_points = 19
h0 = np.array([0.9999934729084969, 0.9999971729084968, 0.6512429646755881,
              0.2573433218863083, 0.7922939787622869])
result = minimize(c5_score, h0, method="SLSQP", options={"maxiter": 300, "ftol": 1e-13})
```
Next Guidance Delta
Try pairwise mass transfer around the first five coordinates.
"""

    prompt = build_guidance_prompt(
        problem_prompt="Find better C5",
        selected_node=root_node,
        selected_entry=None,
        global_best_entries=[],
        local_failure_entries=[code_entry],
    )

    assert "current best valid raw score (0.3810181186942784)" in prompt.user
    assert "successful mutation from 0.5 towards 0.3810181186942784 or lower" in prompt.user
    assert "Extracted attached-solution facts" in prompt.user
    assert "profile h=[0.9999934729084969" in prompt.user
    assert "0.7922939787622869" in prompt.user
    assert "SLSQP maxiter=300" in prompt.user
    assert "SLSQP ftol=1e-13" in prompt.user


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
    contract = prompt.user.split("has not run yet.", 1)[1]
    assert contract.find("<execution_thinking>") < contract.find("```python\ndef run(seed=42")
    assert contract.find("```python\ndef run(seed=42") < contract.find("<summary>")
    assert "Return exactly these three blocks in this order" in prompt.user
    assert "Do not claim verifier" in prompt.user
    assert "success because the verifier has not run yet" in prompt.user
    assert "Execution Interpretation" in prompt.user
    assert "Implemented Algorithm" in prompt.user
    assert "New Ideas Introduced" in prompt.user
    assert "Empirical Outcome" in prompt.user
    assert "Failure / Bottleneck Analysis" in prompt.user
    assert "Next Guidance Delta" in prompt.user
    assert "Target: produce a valid candidate with raw C5 lower than 0.4" in prompt.user
    assert "permits fractional, non-binary h values" in prompt.user
    assert "compute the actual c5_bound" in prompt.user
    assert "copying it or rerunning the exact same SLSQP setup is not an improvement" in prompt.user
    assert "Use at least one concrete non-copy search change from the guidance" in prompt.user
    assert "If guidance mentions active-lag search" in prompt.user
    assert "SLSQP" in prompt.user
    assert "deterministic projected local search" in prompt.user
    assert "project or adjust h" in prompt.user
    assert "keep n_points=63" in prompt.user
    assert "known-good 63-point lineage already showed step-to-step progress" in prompt.user
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


def test_execution_prompt_without_visible_best_targets_known_good_template():
    root_node = LibraryNode(
        id="root",
        problem_id="erdos",
        timestep=0,
        entry_id=None,
        value=2.0,
        raw_score=0.5,
        visits=0,
        parent_id=None,
        children=[],
        metadata={},
    )

    prompt = build_execution_prompt(
        problem_prompt="Find better C5",
        selected_node=root_node,
        selected_entry=None,
        guidance="Use the known-good 63-point template, then perturb it.",
    )

    assert "Target: produce a valid candidate with raw C5 lower than 0.3810181186942784" in prompt.user
    assert "copying it or rerunning the exact same SLSQP setup is not an improvement" in prompt.user


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
Execution Interpretation
test

Implemented Algorithm
coordinate descent

New Ideas Introduced
paired perturbations

Empirical Outcome
pending

Failure / Bottleneck Analysis
shape mismatch

Next Guidance Delta
preserve symmetry and adjust step schedule
</summary>"""

    assert extract_tag(execution_text, "summary").startswith("Execution Interpretation")


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
