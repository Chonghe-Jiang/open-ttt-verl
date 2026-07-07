import json
from pathlib import Path

import pytest

from guidance_ttt.bootstrap import build_bootstrap_execution_prompt, bootstrap_library_entries
from guidance_ttt.library import GuidanceLibrary
from guidance_ttt.state import LLMResponse, make_root_node
from guidance_ttt.tasks import get_task_spec


class MissingSummaryCloseClient:
    async def complete(self, request):
        return LLMResponse(
            text="""<execution_thinking>
Use a constant baseline.
</execution_thinking>

<solution>
```python
def run(seed=42, budget_s=1, **kwargs):
    return ([0.5, 0.5], 0.5, 2)
```
</solution>

<summary>
This summary is intentionally missing its closing tag.
""",
            model=request.model,
            finish_reason="stop",
            usage={},
            metadata={"mock": True},
        )


def test_bootstrap_prompt_has_no_guidance_or_library_context():
    prompt = build_bootstrap_execution_prompt(task_spec=get_task_spec("polyomino_packing"))

    assert "<selected_summary>" not in prompt.user
    assert "<guidance>" not in prompt.user
    assert "<local_failures>" not in prompt.user
    assert "initial bootstrap candidate" in prompt.user
    assert "exactly three top-level XML blocks" in prompt.user
    assert "<execution_thinking>" in prompt.user
    assert "<solution>" in prompt.user
    assert "```cpp" in prompt.user
    assert "complete self-contained C++17 program" in prompt.user
    assert "<summary>" in prompt.user
    assert "future guidance could improve" in prompt.user


def test_bootstrap_entry_attaches_to_root_and_becomes_selected_summary(tmp_path):
    library_path = tmp_path / "library.json"
    root = make_root_node(problem_id="erdos", raw_score=0.5, reward=2.0)
    library = GuidanceLibrary(library_path, initial_nodes=[root], rollout_n=1)

    result = bootstrap_library_entries(
        library_path,
        task_config={"id": "erdos_min_overlap"},
        execution_llm_config={"provider": "mock", "model": "mock-exec"},
        verifier_timeout_s=20,
        max_attempts=1,
    )
    snapshot = library.snapshot()
    root_store = snapshot["nodes"][root.id]
    entry_id = root_store["entry_id"]
    context = library.context_for_node(root)

    assert result["created_count"] == 1
    assert entry_id in snapshot["entries"]
    assert snapshot["entries"][entry_id]["problem_id"] == "erdos"
    assert root_store["metadata"]["bootstrap"] is True
    assert root_store["value"] == snapshot["entries"][entry_id]["verifier_reward"]
    assert context["selected_entry"].id == entry_id
    assert context["selected_entry"].metadata["bootstrap"] is True
    assert context["selected_entry"].metadata["raw_model_summary"]


def test_bootstrap_rejects_missing_summary_close_without_writing_entry(tmp_path):
    library_path = tmp_path / "library.json"
    root = make_root_node(problem_id="erdos_min_overlap", raw_score=0.5, reward=2.0)
    GuidanceLibrary(library_path, initial_nodes=[root], rollout_n=1)

    with pytest.raises(RuntimeError, match="summary"):
        bootstrap_library_entries(
            library_path,
            task_config={"id": "erdos_min_overlap"},
            execution_llm_config={"provider": "mock", "model": "mock-exec"},
            verifier_timeout_s=20,
            max_attempts=1,
            execution_client=MissingSummaryCloseClient(),
        )

    data = json.loads(Path(library_path).read_text())
    assert data["entries"] == {}
    assert data["nodes"][root.id]["entry_id"] is None
