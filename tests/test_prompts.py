from guidance_ttt.prompts import (
    build_execution_prompt,
    build_guidance_prompt,
    extract_guidance_or_format_error,
    extract_tag,
    extract_tag_or_none,
    extract_text_outside_tag,
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


def test_prompt_context_prefers_raw_model_summary_plus_score_over_canonical_summary():
    entry = _entry()
    entry.summary = """Execution Interpretation
canonical thinking should not be attached

Implemented Algorithm
```python
def canonical_solution_code_should_not_be_attached():
    pass
```

Empirical Outcome
Raw C5: 0.4
"""
    entry.metadata = {
        "raw_model_summary": "Raw model summary: projected coordinate descent with mass repair.",
    }

    prompt = build_guidance_prompt(
        problem_prompt="Find better C5",
        selected_node=_node(),
        selected_entry=entry,
        global_best_entries=[],
        local_failure_entries=[],
        raw_score_label="Raw C5",
    )

    assert "Raw model summary: projected coordinate descent with mass repair." in prompt.user
    assert "Raw C5: 0.4" in prompt.user
    assert "Reward: 2.5" in prompt.user
    assert "Verifier status: valid" in prompt.user
    assert "Verifier message: C5 bound: 0.400000" in prompt.user
    assert "canonical thinking should not be attached" not in prompt.user
    assert "canonical_solution_code_should_not_be_attached" not in prompt.user


def test_prompt_context_can_recover_raw_summary_from_execution_text_metadata():
    entry = _entry()
    entry.summary = "Canonical summary should not be attached when execution_text is available."
    entry.metadata = {
        "execution_text": """<execution_thinking>
think
</execution_thinking>
<solution>
```python
def run(): return None
```
</solution>
<summary>
Recovered raw summary from the execution response.
</summary>""",
    }

    prompt = build_execution_prompt(
        problem_prompt="Find better C5",
        selected_node=_node(),
        selected_entry=entry,
        guidance="Try deterministic coordinate descent.",
        raw_score_label="Raw C5",
    )

    assert "Recovered raw summary from the execution response." in prompt.user
    assert "Raw C5: 0.4" in prompt.user
    assert "Reward: 2.5" in prompt.user
    assert "Canonical summary should not be attached" not in prompt.user
    assert "def run(): return None" not in prompt.user


def test_prompt_context_rejects_raw_summary_when_model_omits_closing_summary_tag():
    entry = _entry()
    entry.summary = "Canonical summary should not be attached when execution_text has raw summary content."
    entry.metadata = {
        "execution_text": """<execution_thinking>
think
</execution_thinking>
<solution>
```cpp
int main() { return 0; }
```
</solution>
<summary>
Recovered raw summary from a model response that ended without the closing tag.
It should remain raw model text, not the canonical summary.""",
    }

    prompt = build_execution_prompt(
        problem_prompt="Pack polyominoes.",
        selected_node=_node(),
        selected_entry=entry,
        guidance="Try shelf packing.",
        raw_score_label="FrontierCS score",
    )

    assert "No previous summary is attached." in prompt.user
    assert "Recovered raw summary from a model response that ended without the closing tag." not in prompt.user
    assert "It should remain raw model text, not the canonical summary." not in prompt.user
    assert "FrontierCS score: 0.4" in prompt.user
    assert "Canonical summary should not be attached" not in prompt.user
    assert "int main()" not in prompt.user


def test_prompt_context_never_falls_back_to_canonical_summary_when_raw_summary_is_missing():
    entry = _entry()
    entry.summary = """Execution Interpretation
canonical thinking should not be attached

Implemented Algorithm
```python
def canonical_solution_code_should_not_be_attached():
    pass
```
"""
    entry.metadata = {}

    prompt = build_execution_prompt(
        problem_prompt="Find better C5",
        selected_node=_node(),
        selected_entry=entry,
        guidance="Try deterministic coordinate descent.",
        raw_score_label="Raw C5",
    )

    assert "No previous summary is attached." in prompt.user
    assert "Raw C5: 0.4" in prompt.user
    assert "Reward: 2.5" in prompt.user
    assert "Verifier status: valid" in prompt.user
    assert "Verifier message: C5 bound: 0.400000" in prompt.user
    assert "canonical thinking should not be attached" not in prompt.user
    assert "canonical_solution_code_should_not_be_attached" not in prompt.user


def test_guidance_prompt_attaches_selected_raw_summary_but_not_library_details():
    entry = _entry()
    entry.metadata = {
        "raw_model_summary": """  Raw summary with intentional leading spaces.

Implemented Algorithm
```python
def run(seed=42, budget_s=1, **kwargs):
    h_values = [0.1, 0.2, 0.7]
    return (h_values, 0.4, 3)
```

Next Guidance Delta
Preserve this exact raw text."""
    }
    prompt = build_guidance_prompt(
        problem_prompt="Find better C5",
        selected_node=_node(),
        selected_entry=entry,
        global_best_entries=[],
        local_failure_entries=[],
    )

    assert "Find better C5" in prompt.user
    assert "The next sections describe the current search state for this problem" in prompt.user
    assert prompt.user.index("</problem>") < prompt.user.index("The next sections describe")
    assert prompt.user.index("The next sections describe") < prompt.user.index("<selected_summary>")
    assert "<selected_summary>" in prompt.user
    assert "  Raw summary with intentional leading spaces." in prompt.user
    assert "```python\ndef run(seed=42, budget_s=1, **kwargs):" in prompt.user
    assert "h_values = [0.1, 0.2, 0.7]" in prompt.user
    assert "Preserve this exact raw text." in prompt.user
    assert "[code omitted" not in prompt.user
    assert "Extracted attached-code facts" not in prompt.user
    assert "[truncated]" not in prompt.user
    assert "Entry id:" not in prompt.user
    assert "Reward: 2.5" in prompt.user
    assert "Raw score:" not in prompt.user
    assert "Score: 0.4" in prompt.user
    assert "Verifier status: valid" in prompt.user
    assert "Verifier message: C5 bound: 0.400000" in prompt.user
    assert "Previous guidance:" not in prompt.user
    assert "Reusable idea:" not in prompt.user
    assert "preserve symmetry" not in prompt.user
    assert "<guidance>" in prompt.user
    assert "You are the Guidance Model" in prompt.system
    assert "evolutionary guidance" in prompt.system
    assert "Do not write final code" in prompt.system
    assert "escape local optima" in prompt.system
    assert "# Objective" in prompt.user
    assert "next **evolutionary guidance**" in prompt.user
    assert "# Evolutionary Guidelines" in prompt.user
    assert "1. Analyze the search history." in prompt.user
    assert "what has already been tried, what worked, what failed" in prompt.user
    assert "2. Stay at the algorithmic-strategy level." in prompt.user
    assert "Propose high-level algorithmic directions and ideas." in prompt.user
    assert "Do not write code, implementation details, or parameter schedules." in prompt.user
    assert "3. Produce exactly the required XML structure." in prompt.user
    assert "Provide internal reasoning and return exactly one `<guidance>` block" in prompt.user
    assert "Analyze History, Do Not Repeat It" not in prompt.user
    assert "High-Level Mutations, No Low-Level Details" not in prompt.user
    assert "Strict Separation of Thought and Action" not in prompt.user
    assert "<think>" not in prompt.user
    assert "</think>" not in prompt.user
    assert "The following notes explain what each block should contain" not in prompt.user
    assert "Please do internal reasoning and provide your response exactly in the following format" in prompt.user
    assert "Think step by step internally" not in prompt.user
    assert "<guidance>\nProvide the final evolutionary guidance for the next execution attempt" in prompt.user
    assert "Describe the main algorithmic direction and keep the guidance conceptual and actionable" in prompt.user
    assert "Use this space entirely for internal reflection" not in prompt.user
    assert "This must contain only your final, actionable evolutionary trajectory" not in prompt.user
    format_index = prompt.user.index("Please do internal reasoning and provide your response exactly in the following format")
    guidance_index = prompt.user.index("<guidance>\nProvide the final evolutionary guidance for the next execution attempt")
    assert format_index < guidance_index
    assert "Evolutionary Mutation" not in prompt.user
    assert "Directional Search Strategy" not in prompt.user
    assert "Progress Target" not in prompt.user
    assert "Do not write code, implementation details, or parameter schedules." in prompt.user
    assert "current visible target raw score (0.4)" in prompt.user
    assert "successful mutation from 0.4 towards 0.4 or lower" not in prompt.user


def test_guidance_prompt_omits_global_best_and_keeps_local_failure_history():
    global_best = _entry()
    global_best.id = "best-entry"
    global_best.summary = "best history used mirror minimax"
    global_best.guidance = "preserve best mirror profile"
    global_best.reusable_idea = "reuse best projection repair"
    local_failure = _entry()
    local_failure.id = "failure-entry"
    local_failure.metadata = {"raw_model_summary": "failed because sum drifted"}
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

    assert "<global_best>" not in prompt.user
    assert "best history used mirror minimax" not in prompt.user
    assert "<local_failures>" in prompt.user
    assert "failed because sum drifted" in prompt.user
    assert "best-entry" not in prompt.user
    assert "failure-entry" not in prompt.user
    assert "reuse best projection repair" not in prompt.user
    assert "sum(h) must equal n_points / 2" in prompt.user
    assert "Failure mode: invalid" not in prompt.user


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

    assert "current visible target score (0.3821438682282878)" in prompt.user
    assert "according to the task-specific objective and score direction" in prompt.user
    assert "Lower raw C5 is better" not in prompt.user
    assert "what bottleneck the next attempt should address" in prompt.user
    assert "high-level algorithmic directions" in prompt.user
    assert "Propose high-level algorithmic directions and ideas." in prompt.user
    assert "Do not write code, implementation details, or parameter schedules." in prompt.user
    assert "The following notes explain what each block should contain" not in prompt.user
    assert "Provide the final evolutionary guidance for the next execution attempt" in prompt.user
    assert "The preferred submitted guidance" not in prompt.user
    assert "Preserve the current best valid construction" not in prompt.user
    assert "SLSQP again" not in prompt.user
    assert "Avoid GPU tensors, large correlation matrices, or unbounded minimax solvers" not in prompt.user
    assert "Do not ask the execution model to reinvent minimax from scratch" not in prompt.user


def test_guidance_prompt_accepts_task_specific_objective_text():
    prompt = build_guidance_prompt(
        problem_prompt="Pack the Polyominoes",
        selected_node=LibraryNode(
            id="poly-root",
            problem_id="polyomino_packing",
            timestep=0,
            entry_id=None,
            value=0.0,
            raw_score=0.0,
            visits=0,
            parent_id=None,
            children=[],
            metadata={},
        ),
        selected_entry=None,
        global_best_entries=[],
        local_failure_entries=[],
        objective_text="Beat the current visible FrontierCS score target (0.0). Higher FrontierCS score is better.",
    )

    assert "Pack the Polyominoes" in prompt.user
    assert "Beat the current visible FrontierCS score target (0.0)" in prompt.user
    assert "Higher FrontierCS score is better" in prompt.user
    assert "Lower raw C5 is better" not in prompt.user


def test_guidance_prompt_uses_only_summary_not_verified_profile_artifacts():
    best = _entry()
    best.id = "best-entry"
    best.verifier_raw_score = 0.3812435631313583
    best.verifier_reward = 2.6229950364447134
    best.metadata = {
        "raw_model_summary": (
            "Empirical Outcome\n"
            "Verified returned profile: n_points=19, c5_bound=0.3812435631313583, "
            "h length=19, head=[1.0, 0.9872194239853203], tail=[0.9951851401967569, 0.9999965067437544]"
        ),
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

    assert "Verified returned profile artifacts (authoritative initialization)" not in prompt.user
    assert "raw C5=0.3812435631313583" not in prompt.user
    assert "Empirical Outcome" in prompt.user
    assert "Verified returned profile: n_points=19, c5_bound=0.3812435631313583" in prompt.user
    assert "Score: 0.3812435631313583" in prompt.user
    assert "Reward: 2.6229950364447134" in prompt.user
    assert "The selected summary is also the current global best visible summary." not in prompt.user
    assert "<global_best>" not in prompt.user
    assert "what bottleneck the next attempt should address" in prompt.user


def test_guidance_prompt_root_uses_selected_raw_score_without_initial_construction_facts():
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
        metadata={
            "initialization": "random_perturbed_constant",
            "n_points": 4,
            "h_values": [0.2, 0.4, 0.6, 0.8],
            "c5_bound": 0.5,
        },
    )
    code_entry = _entry()
    code_entry.metadata = {
        "raw_model_summary": "Execution Interpretation\n" + ("long inherited context. " * 80) + """

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
    }

    prompt = build_guidance_prompt(
        problem_prompt="Find better C5",
        selected_node=root_node,
        selected_entry=None,
        global_best_entries=[],
        local_failure_entries=[code_entry],
    )

    assert "current visible target raw score (0.5)" in prompt.user
    assert "The following notes explain what each block should contain" not in prompt.user
    assert "Provide the final evolutionary guidance for the next execution attempt" in prompt.user
    assert "Current initial construction (reference state to improve)" not in prompt.user
    assert "initialization=random_perturbed_constant" not in prompt.user
    assert "h=[0.2, 0.4, 0.6, 0.8]" not in prompt.user
    assert "Extracted attached-code facts" not in prompt.user
    assert "[code omitted" not in prompt.user
    assert "```python" in prompt.user
    assert "result = minimize(c5_score, h0, method=\"SLSQP\", options={\"maxiter\": 300, \"ftol\": 1e-13})" in prompt.user
    assert "Try pairwise mass transfer around the first five coordinates." in prompt.user


def test_execution_prompt_is_thin_wrapper_around_problem_guidance_and_raw_summaries():
    selected_entry = _entry()
    selected_entry.metadata = {
        "raw_model_summary": """Selected raw summary.

```python
def selected_candidate():
    return "keep this code visible"
```
"""
    }
    prompt = build_execution_prompt(
        problem_prompt="Find better C5",
        selected_node=_node(),
        selected_entry=selected_entry,
        guidance="Try deterministic coordinate descent.",
    )

    assert "Find better C5" in prompt.user
    assert "Try deterministic coordinate descent." in prompt.user
    assert "The next sections describe the current search state for this problem" in prompt.user
    assert "run-local context when implementing the guided candidate" not in prompt.user
    assert prompt.user.index("</problem>") < prompt.user.index("The next sections describe")
    assert prompt.user.index("The next sections describe") < prompt.user.index("<selected_summary>")
    assert "<selected_summary>" in prompt.user
    assert "Selected raw summary." in prompt.user
    assert "def selected_candidate()" in prompt.user
    assert "Node id:" not in prompt.user
    assert "Timestep:" not in prompt.user
    assert "Value:" not in prompt.user
    assert "Raw score:" not in prompt.user
    assert "Entry id:" not in prompt.user
    assert "Verifier status: valid" in prompt.user
    assert "Previous guidance:" not in prompt.user
    assert "Reusable idea:" not in prompt.user
    assert "Previous solution code excerpt" not in prompt.user
    assert "return ([0.5, 0.5], 0.5, 2)" not in prompt.user
    assert "<execution_thinking>" in prompt.user
    assert "<solution>" in prompt.user
    assert "</solution>" in prompt.user
    assert "<summary>" in prompt.user
    assert "Use the problem statement as the authoritative task specification" in prompt.user
    assert "Use the attached library context as historical evidence" in prompt.user
    assert "Score direction: min." in prompt.user
    assert "Implement one concrete solution that follows the guidance" in prompt.user
    assert "Return exactly these three blocks" not in prompt.user
    assert "Your response must contain exactly three top-level XML blocks" in prompt.user
    assert "Required output format:" in prompt.user
    contract = prompt.user.split("Required output format:", 1)[1]
    assert contract.find("<execution_thinking>") < contract.find("<solution>")
    assert contract.find("<solution>") < contract.find("```python")
    assert contract.find("```python") < contract.find("</solution>")
    assert contract.find("</solution>") < contract.find("<summary>")
    assert "A short explanation of how the guidance was converted into the submitted algorithm" in prompt.user
    assert "Do not include code." in prompt.user
    assert "A concise natural-language summary of the candidate" in prompt.user
    assert "guidance-driven change from the prior idea" in prompt.user
    assert "If a suggested guidance component was not actually implemented" in prompt.user
    assert "representation choices, search operators, feasibility checks" in prompt.user
    assert "objective handling, restart or exploration strategy, normalization, projection" in prompt.user
    assert "Do not include source code, code fences, copied constants" in prompt.user
    assert "Any response that does not follow this exact three-block structure should be treated as invalid" in prompt.user
    assert "You must output all three XML blocks exactly as shown below" in prompt.user
    assert "The <solution> block is mandatory and must contain a fenced ```python code block" in prompt.user
    assert "The <summary>...</summary> block is mandatory and must be closed" in prompt.user
    assert "Do not output only execution_thinking, only a summary, or plain natural language" in prompt.user
    assert "Do not omit angle brackets from XML tags" in prompt.user
    assert "Execution Interpretation" not in contract
    assert "Implemented Algorithm" not in contract
    assert "New Ideas Introduced" not in contract
    assert "Empirical Outcome" not in contract
    assert "Failure / Bottleneck Analysis" not in contract
    assert "Next Guidance Delta" not in contract
    assert "Target: produce a valid candidate with raw C5" not in prompt.user
    assert "copying it or rerunning the exact same SLSQP setup is not an improvement" not in prompt.user
    assert "Use at least one concrete non-copy search change from the guidance" not in prompt.user
    assert "If guidance mentions active-lag search" not in prompt.user
    assert "deterministic projected local search" not in prompt.user
    assert "project or adjust h" not in prompt.user
    assert "Preserve n_points when reusing a previous valid profile" not in prompt.user
    assert "strict improvement over the current best raw C5" not in prompt.user
    assert "box-constrained projection" not in prompt.user
    assert "sum(h) == n_points / 2 to verifier tolerance" not in prompt.user
    assert "project_to_box_sum" not in prompt.user
    assert "return [float(x) for x in h], float(c5_bound), int(n_points)" not in prompt.user
    assert "Avoid GPU tensors, large correlation matrices, or unbounded minimax solvers" not in prompt.user
    assert "invent a fresh minimax construction" not in prompt.user


def test_execution_prompt_accepts_cpp_solution_contract():
    prompt = build_execution_prompt(
        problem_prompt="Pack polyominoes from stdin.",
        selected_node=LibraryNode(
            id="poly-root",
            problem_id="polyomino_packing",
            timestep=0,
            entry_id=None,
            value=0.0,
            raw_score=0.0,
            visits=0,
            parent_id=None,
            children=[],
            metadata={},
        ),
        selected_entry=None,
        guidance="Use skyline placement with rotations.",
        solution_language="cpp",
        solution_contract=(
            "The <solution> block must contain one complete C++17 program in a ```cpp fenced block."
        ),
        score_direction="max",
    )

    assert "Pack polyominoes from stdin." in prompt.user
    assert "Use skyline placement with rotations." in prompt.user
    assert "complete C++17 program" in prompt.user
    assert "Score direction: max." in prompt.user
    assert "```cpp" in prompt.user
    assert "```python" not in prompt.user
    assert "Turn guidance into one concrete runnable C++17 candidate" in prompt.system


def test_execution_prompt_omits_global_best_valid_summary():
    global_best = _entry()
    global_best.id = "best-entry"
    global_best.verifier_raw_score = 0.3821438682282878
    global_best.summary = """Best valid raw summary.

```python
def best_candidate():
    return ([0.4, 0.6], 0.3821438682282878, 2)
```
"""
    global_best.solution = "def run(seed=42, budget_s=1, **kwargs):\n    return ([0.4, 0.6], 0.3821438682282878, 2)"
    global_best.metadata = {
        "verification_artifacts": {
            "n_points": 2,
            "c5_bound": 0.3821438682282878,
            "h_values": [0.4, 0.6],
        }
    }

    prompt = build_execution_prompt(
        problem_prompt="Find better C5",
        selected_node=_node(),
        selected_entry=None,
        global_best_entries=[global_best],
        guidance="Preserve current best and perturb locally.",
    )

    assert "<global_best_valid_summary>" not in prompt.user
    assert "Best valid raw summary." not in prompt.user
    assert "def best_candidate()" not in prompt.user
    assert "Entry id: best-entry" not in prompt.user
    assert "Score: 0.3821438682282878" not in prompt.user
    assert "Reward: 2.5" not in prompt.user
    assert "Verified returned profile artifacts" not in prompt.user
    assert "h=[0.4, 0.6]" not in prompt.user
    assert "def run(seed=42, budget_s=1, **kwargs)" not in prompt.user
    assert "Initialize from verified profile artifacts when available" not in prompt.user
    assert "Use the problem statement as the authoritative task specification" in prompt.user


def test_execution_prompt_without_visible_best_omits_root_metadata_and_task_specific_target():
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
        metadata={
            "initialization": "random_perturbed_constant",
            "n_points": 4,
            "h_values": [0.2, 0.4, 0.6, 0.8],
            "c5_bound": 0.5,
        },
    )

    prompt = build_execution_prompt(
        problem_prompt="Find better C5",
        selected_node=root_node,
        selected_entry=None,
        guidance="Perturb the current initial construction.",
    )

    assert "Target: produce a valid candidate with raw C5 lower than 0.5" not in prompt.user
    assert "Current initial construction (reference state to improve)" not in prompt.user
    assert "initialization=random_perturbed_constant" not in prompt.user
    assert "h=[0.2, 0.4, 0.6, 0.8]" not in prompt.user
    assert "copying it or rerunning the exact same SLSQP setup is not an improvement" not in prompt.user


def test_prompts_do_not_embed_removed_known_good_seed():
    guidance_prompt = build_guidance_prompt(
        problem_prompt="Find better C5",
        selected_node=_node(),
        selected_entry=None,
        global_best_entries=[],
        local_failure_entries=[],
    )
    execution_prompt = build_execution_prompt(
        problem_prompt="Find better C5",
        selected_node=_node(),
        selected_entry=None,
        guidance="Try a new projected perturbation.",
    )

    combined = "\n".join(
        [
            guidance_prompt.system,
            guidance_prompt.user,
            execution_prompt.system,
            execution_prompt.user,
        ]
    )
    assert "0.3810181186942784" not in combined
    assert "<known_good_minimax_template>" not in combined
    assert "known-good" not in combined


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


def test_guidance_extraction_unwraps_redundant_guidance_tags():
    text = """<think>
</think>

<guidance>
<guidance>
Hypothesis: try skyline packing.
</guidance>
</guidance>"""

    guidance, ok = extract_guidance_or_format_error(text)

    assert ok is True
    assert guidance == "Hypothesis: try skyline packing."


def test_guidance_extraction_rejects_empty_nested_guidance_tags():
    text = """<think>
</think>

<guidance>
<guidance>
</guidance>
</guidance>"""

    guidance, ok = extract_guidance_or_format_error(text)

    assert ok is False
    assert "formatting failure" in guidance
    assert "</guidance>" not in guidance


def test_guidance_extraction_rejects_text_outside_think_without_guidance_tag():
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
    assert "formatting failure" in guidance
    assert "Hypothesis: use projected coordinate descent." not in guidance
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


def test_execution_prompt_unwraps_guidance_before_wrapping_once():
    prompt = build_execution_prompt(
        problem_prompt="Pack the pieces.",
        selected_node=_node(),
        selected_entry=_entry(),
        guidance="<guidance>\nUse skyline packing.\n</guidance>",
        solution_language="cpp",
        solution_contract="Return C++.",
    )

    assert prompt.user.count("<guidance>") == 1
    assert prompt.user.count("</guidance>") == 1
    assert "<guidance>\nUse skyline packing.\n</guidance>" in prompt.user
