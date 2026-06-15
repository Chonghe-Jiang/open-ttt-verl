from __future__ import annotations

from dataclasses import dataclass

from guidance_ttt.state import LibraryEntry, LibraryNode


@dataclass
class Prompt:
    system: str
    user: str


def _entry_summary(entry: LibraryEntry | None, *, include_solution: bool) -> str:
    if entry is None:
        return "No previous library entry is attached."
    parts = [
        f"Entry id: {entry.id}",
        f"Summary: {entry.summary}",
        f"Previous guidance: {entry.guidance}",
        f"Previous execution thinking: {entry.execution_thinking}",
        f"Reward: {entry.verifier_reward}",
        f"Raw score: {entry.verifier_raw_score}",
        f"Verifier status: {entry.verifier_status}",
        f"Verifier message: {entry.verifier_message}",
        f"Reusable idea: {entry.reusable_idea}",
    ]
    if entry.failure_mode:
        parts.append(f"Failure mode: {entry.failure_mode}")
    if include_solution:
        parts.append("Previous full solution code:\n```python\n" + entry.solution.strip() + "\n```")
    return "\n".join(parts)


def build_guidance_prompt(
    *,
    problem_prompt: str,
    selected_node: LibraryNode,
    selected_entry: LibraryEntry | None,
    global_best_entries: list[LibraryEntry],
    local_failure_entries: list[LibraryEntry],
) -> Prompt:
    best_text = "\n\n".join(_entry_summary(entry, include_solution=False) for entry in global_best_entries)
    failure_text = "\n\n".join(_entry_summary(entry, include_solution=False) for entry in local_failure_entries)
    user = f"""<problem>
{problem_prompt}
</problem>

<selected_library_node>
Node id: {selected_node.id}
Timestep: {selected_node.timestep}
Value: {selected_node.value}
Raw score: {selected_node.raw_score}
Visits: {selected_node.visits}
{_entry_summary(selected_entry, include_solution=False)}
</selected_library_node>

<global_best>
{best_text or "No global best entry yet."}
</global_best>

<local_failures>
{failure_text or "No local failure entries yet."}
</local_failures>

Return exactly:
<guidance>
Hypothesis: ...
Plan:
1. ...
2. ...
What to preserve: ...
What to change: ...
Expected verifier signal: ...
</guidance>
"""
    return Prompt(
        system="You are the trainable guidance model. Produce ideas, not final code.",
        user=user,
    )


def build_execution_prompt(
    *,
    problem_prompt: str,
    selected_node: LibraryNode,
    selected_entry: LibraryEntry | None,
    guidance: str,
) -> Prompt:
    user = f"""<problem>
{problem_prompt}
</problem>

<selected_library_node>
Node id: {selected_node.id}
Timestep: {selected_node.timestep}
Value: {selected_node.value}
Raw score: {selected_node.raw_score}
{_entry_summary(selected_entry, include_solution=True)}
</selected_library_node>

<guidance>
{guidance}
</guidance>

Return:
<execution_thinking>
brief reasoning
</execution_thinking>

```python
# final runnable solution
```
"""
    return Prompt(
        system="You are the execution model. Turn guidance into one concrete candidate solution.",
        user=user,
    )


def build_summary_prompt(
    *,
    selected_entry: LibraryEntry | None,
    guidance: str,
    execution_thinking: str,
    solution: str,
    reward: float,
    raw_score: float | None,
    status: str,
    message: str,
) -> Prompt:
    user = f"""Selected parent summary:
{_entry_summary(selected_entry, include_solution=False)}

Guidance:
{guidance}

Execution thinking:
{execution_thinking}

Solution/code:
```python
{solution}
```

Verifier reward: {reward}
Verifier raw score: {raw_score}
Verifier status: {status}
Verifier message: {message}

Return exactly:
<summary>
Outcome: ...
Reusable idea: ...
Failure mode: ...
What future guidance should preserve: ...
What future guidance should change: ...
</summary>
"""
    return Prompt(
        system="You summarize attempts for a TTT library. Use only provided evidence.",
        user=user,
    )


def extract_tag(text: str, tag: str) -> str:
    start = text.find(f"<{tag}>")
    end = text.find(f"</{tag}>")
    if start == -1 or end == -1 or end <= start:
        return text.strip()
    return text[start + len(tag) + 2 : end].strip()
