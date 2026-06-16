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

The preferred submitted guidance is the text inside the XML block below. Thinking is allowed,
but if the XML block is missing, only text outside any <think>...</think> block will be used
as the submitted guidance.

Return exactly one block and nothing else:
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
        system=(
            "You are the trainable guidance model. Produce high level ideas, not final code. "
            "You may think first, but the final submitted guidance should be in one "
            "<guidance>...</guidance> block."
        ),
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

Return all three sections in this exact order. Do not claim verifier success because the verifier has not run yet.

Return:
<execution_thinking>
brief reasoning
</execution_thinking>

```python
# final runnable solution
```

<summary>
Outcome hypothesis: ...
Reusable idea: ...
Risk / possible failure mode: ...
What future guidance should preserve: ...
What future guidance should change: ...
</summary>
"""
    return Prompt(
        system=(
            "You are the execution model. Turn guidance into one concrete candidate solution, "
            "and include a concise library summary of the attempt intent."
        ),
        user=user,
    )


def extract_tag(text: str, tag: str) -> str:
    start = text.find(f"<{tag}>")
    end = text.find(f"</{tag}>")
    if start == -1 or end == -1 or end <= start:
        return text.strip()
    return text[start + len(tag) + 2 : end].strip()


def extract_tag_or_none(text: str, tag: str) -> str | None:
    start = text.find(f"<{tag}>")
    end = text.find(f"</{tag}>")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start + len(tag) + 2 : end].strip()


def extract_text_outside_tag(text: str, tag: str) -> str:
    lower_text = text.lower()
    open_tag = f"<{tag.lower()}>"
    close_tag = f"</{tag.lower()}>"
    parts: list[str] = []
    cursor = 0
    while True:
        start = lower_text.find(open_tag, cursor)
        if start == -1:
            parts.append(text[cursor:])
            break
        parts.append(text[cursor:start])
        end = lower_text.find(close_tag, start + len(open_tag))
        if end == -1:
            break
        cursor = end + len(close_tag)
    return "".join(parts).strip()


def extract_guidance_or_format_error(text: str) -> tuple[str, bool]:
    guidance = extract_tag_or_none(text, "guidance")
    if guidance:
        return guidance, True
    outside_think = extract_text_outside_tag(text, "think")
    if outside_think:
        return outside_think, False
    return (
        "Hypothesis: The guidance model did not emit a valid <guidance> block.\n"
        "Plan:\n"
        "1. Treat this attempt as a formatting failure because no text was found outside the thinking block.\n"
        "2. Retry with an explicit tagged guidance response on the next rollout.\n"
        "What to preserve: The selected library context and Erdos verifier constraints.\n"
        "What to change: Emit exactly one tagged guidance block after any thinking.\n"
        "Expected verifier signal: formatting_error",
        False,
    )
