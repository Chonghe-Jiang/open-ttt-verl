from __future__ import annotations

from dataclasses import dataclass

from guidance_ttt.state import LibraryEntry, LibraryNode


@dataclass
class Prompt:
    system: str
    user: str


def _raw_model_summary_for_prompt(entry: LibraryEntry | None) -> str | None:
    if entry is None:
        return None
    metadata = entry.metadata or {}
    raw_model_summary = metadata.get("raw_model_summary")
    if isinstance(raw_model_summary, str) and raw_model_summary != "":
        return raw_model_summary
    execution_text = metadata.get("execution_text")
    if isinstance(execution_text, str) and execution_text:
        parsed_summary = extract_tag_or_none(execution_text, "summary")
        if parsed_summary:
            return parsed_summary
    return None


def _score_for_prompt(entry: LibraryEntry, *, raw_score_label: str) -> str:
    raw_score = "None" if entry.verifier_raw_score is None else repr(float(entry.verifier_raw_score))
    return "\n".join(
        [
            f"Verifier status: {entry.verifier_status}",
            f"{raw_score_label}: {raw_score}",
            f"Reward: {float(entry.verifier_reward)!r}",
            f"Verifier message: {entry.verifier_message}",
        ]
    )


def _raw_summary_for_prompt(entry: LibraryEntry | None, *, fallback: str, raw_score_label: str = "Score") -> str:
    if entry is None:
        return fallback
    raw_summary = _raw_model_summary_for_prompt(entry)
    if raw_summary is None:
        return f"{fallback}\n\n{_score_for_prompt(entry, raw_score_label=raw_score_label)}"
    return f"{raw_summary}\n\n{_score_for_prompt(entry, raw_score_label=raw_score_label)}"


def build_guidance_prompt(
    *,
    problem_prompt: str,
    selected_node: LibraryNode,
    selected_entry: LibraryEntry | None,
    global_best_entries: list[LibraryEntry],
    local_failure_entries: list[LibraryEntry],
    objective_text: str | None = None,
    raw_score_label: str = "Score",
) -> Prompt:
    failure_text = "\n\n".join(
        _raw_summary_for_prompt(
            entry,
            fallback="No previous summary is attached.",
            raw_score_label=raw_score_label,
        )
        for entry in local_failure_entries
    )
    best_valid = _best_valid_entry(global_best_entries, selected_entry)
    best_valid_raw_score = (
        best_valid.verifier_raw_score
        if best_valid is not None and best_valid.verifier_raw_score is not None
        else selected_node.raw_score
    )
    best_valid_target = best_valid_raw_score if best_valid_raw_score is not None else selected_node.raw_score
    objective = objective_text or (
        "Your task is to provide the next **evolutionary guidance** to beat the current visible "
        f"target raw score ({best_valid_target}). Lower raw C5 is better."
    )
    user = f"""<problem>
{problem_prompt}
</problem>

The next sections describe the current search state for this problem. Use them
as run-local context when deciding the next step.

<selected_summary>
{_raw_summary_for_prompt(selected_entry, fallback="No previous summary is attached.", raw_score_label=raw_score_label)}
</selected_summary>

<local_failures>
{failure_text or "No local failure summaries yet."}
</local_failures>

# Objective
{objective}

# Evolutionary Guidelines
1. **Analyze History, Do Not Repeat It:** Identify why the current profile plateaued based on `<selected_summary>` and `<local_failures>`.
2. **High-Level Mutations, No Low-Level Details:** Propose conceptual algorithmic shifts, structural relaxations, or novel search topologies (e.g., introducing a new mathematical constraint or hybridizing optimization frameworks). Do not write code or micromanage hyperparameters.
3. **Strict Separation of Thought and Action:** You must separate your cognitive process from the final directional output using the exact XML tags provided below.

Provide your response exactly in the following format:

<think>
</think>

<guidance>
</guidance>

The following notes explain what each block should contain:
<think>
Use this space entirely for internal reflection. Diagnose historical bottlenecks from the logs, extract lessons from local failures, and debate which conceptual shift is most likely to yield a breakthrough.
</think>
<guidance>
This must contain only your final, actionable evolutionary trajectory.
</guidance>
"""
    return Prompt(
        system=(
            "You are the Guidance Model, acting as a strategic navigator for an open-ended scientific "
            "discovery process.\n\n"
            "Your primary objective is to provide **evolutionary guidance**. Do not write final code "
            "or focus on low-level implementation details. Instead, your task is to propose high-level "
            "directional shifts, conceptual mutations, and novel pathways to explore the search space.\n\n"
            "Focus on how the current ideas can *evolve* to escape local optima and discover "
            "fundamentally new mechanisms."
        ),
        user=user,
    )


def _best_valid_entry(
    *entry_groups: list[LibraryEntry] | LibraryEntry | None,
    score_direction: str = "min",
) -> LibraryEntry | None:
    entries: list[LibraryEntry] = []
    for group in entry_groups:
        if group is None:
            continue
        if isinstance(group, list):
            entries.extend(group)
        else:
            entries.append(group)
    valid_entries = [
        entry
        for entry in entries
        if entry.verifier_status == "valid" and entry.verifier_raw_score is not None
    ]
    if not valid_entries:
        return None
    key = lambda entry: float(entry.verifier_raw_score)
    if score_direction == "max":
        return max(valid_entries, key=key)
    return min(valid_entries, key=key)


def build_execution_prompt(
    *,
    problem_prompt: str,
    selected_node: LibraryNode,
    selected_entry: LibraryEntry | None,
    global_best_entries: list[LibraryEntry] | None = None,
    guidance: str,
    solution_language: str = "python",
    solution_contract: str | None = None,
    score_direction: str = "min",
    raw_score_label: str = "Score",
) -> Prompt:
    _ = global_best_entries
    prompt_guidance = _unwrap_redundant_tag(guidance, "guidance") or guidance.strip()
    fenced_language = "cpp" if solution_language.lower() in {"cpp", "c++", "cxx"} else "python"
    language_name = "C++17" if fenced_language == "cpp" else "Python"
    placeholder = (
        "// complete executable solution required by the problem"
        if fenced_language == "cpp"
        else "# complete executable solution required by the problem"
    )
    contract = solution_contract or (
        "The <solution> block must contain one complete executable Python candidate in a ```python fenced block."
    )
    user = f"""<problem>
{problem_prompt}
</problem>

The next sections describe the current search state for this problem.

<selected_summary>
{_raw_summary_for_prompt(selected_entry, fallback="No previous summary is attached.", raw_score_label=raw_score_label)}
</selected_summary>

<guidance>
{prompt_guidance}
</guidance>

Use the problem statement as the authoritative task specification.
Use the attached library context as historical evidence, not as code to copy blindly.
Score direction: {score_direction}.
Implement one concrete solution that follows the guidance while satisfying the problem specification.
{contract}

Return exactly these three blocks:
If you omit any required XML block, the attempt will be treated as invalid.

<execution_thinking>
Briefly explain how the guidance was translated into the submitted solution.
Do not place the summary inside <execution_thinking>.
</execution_thinking>

<solution>
```{fenced_language}
{placeholder}
```
</solution>

<summary>
The <summary>...</summary> block is required.
Summarize the solution and its guidance-driven diff from the previous idea in natural language. Explain how the candidate was generated, including the search, refinement, or optimization strategy used, and what specific changes were made based on the guidance. If implementation details are central to the solution, such as parameter tuning, threshold choices, normalization, perturbation design, or constraint handling, highlight them and explain why they matter. Do not include code, hard-coded arrays, copied profile values, or raw candidate parameters.
</summary>
"""
    return Prompt(
        system=(
            f"You are the execution model. Turn guidance into one concrete runnable {language_name} "
            "candidate. Output execution thinking first, then the code block, then the summary."
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
    open_tag = f"<{tag}>"
    close_tag = f"</{tag}>"
    start = text.find(open_tag)
    if start == -1:
        return None
    cursor = start + len(open_tag)
    depth = 1
    while depth > 0:
        next_open = text.find(open_tag, cursor)
        next_close = text.find(close_tag, cursor)
        if next_close == -1:
            return None
        if next_open != -1 and next_open < next_close:
            depth += 1
            cursor = next_open + len(open_tag)
            continue
        depth -= 1
        if depth == 0:
            return text[start + len(open_tag) : next_close].strip()
        cursor = next_close + len(close_tag)
    return None


def _unwrap_redundant_tag(text: str, tag: str) -> str:
    value = text.strip()
    while value:
        inner = extract_tag_or_none(value, tag)
        if inner is None:
            break
        outside = extract_text_outside_tag(value, tag)
        if outside:
            break
        if inner.strip() == value:
            break
        value = inner.strip()
    return value


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
    if guidance is not None:
        unwrapped_guidance = _unwrap_redundant_tag(guidance, "guidance")
        if unwrapped_guidance:
            return unwrapped_guidance, True
        return _guidance_format_error()
    return _guidance_format_error()


def _guidance_format_error() -> tuple[str, bool]:
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
