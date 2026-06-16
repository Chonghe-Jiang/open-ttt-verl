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
    assert "<summary>" in prompt.user
    assert "Risk / possible failure mode" in prompt.user
    assert "Do not claim verifier success" in prompt.user


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
