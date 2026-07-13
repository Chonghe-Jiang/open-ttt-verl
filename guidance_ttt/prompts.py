from __future__ import annotations

from dataclasses import dataclass

from guidance_ttt.state import LibraryEntry, LibraryNode


PROMPT_MODE_SUMMARY_ONLY = "summary_only"
PROMPT_MODE_CODE_DELTA = "code_delta"
SUPPORTED_PROMPT_MODES = frozenset({PROMPT_MODE_SUMMARY_ONLY, PROMPT_MODE_CODE_DELTA})


@dataclass
class Prompt:
    system: str
    user: str


def normalize_prompt_mode(prompt_mode: str | None) -> str:
    normalized = str(prompt_mode or PROMPT_MODE_SUMMARY_ONLY).strip().lower()
    if normalized not in SUPPORTED_PROMPT_MODES:
        supported = ", ".join(sorted(SUPPORTED_PROMPT_MODES))
        raise ValueError(f"Unsupported prompt mode {prompt_mode!r}; expected one of: {supported}")
    return normalized


def summary_semantics_for_entry(entry: LibraryEntry) -> str:
    metadata = entry.metadata or {}
    semantics = metadata.get("summary_semantics")
    if isinstance(semantics, str) and semantics.strip():
        return semantics.strip()
    if bool(metadata.get("bootstrap")) or int(entry.timestep) == 0:
        return "baseline"
    return "delta_from_parent"


def validate_entry_prompt_mode(entry: LibraryEntry, prompt_mode: str | None) -> None:
    requested_mode = normalize_prompt_mode(prompt_mode)
    metadata = entry.metadata or {}
    recorded_mode = metadata.get("prompt_mode")
    if bool(metadata.get("bootstrap")) or int(entry.timestep) == 0:
        return
    effective_mode = normalize_prompt_mode(recorded_mode)
    if effective_mode != requested_mode:
        raise RuntimeError(
            f"Library entry {entry.id!r} uses prompt_mode={effective_mode!r}, "
            f"but this run requests prompt_mode={requested_mode!r}. Start a fresh output directory "
            "or resume with the matching prompt mode."
        )


def _raw_model_summary_for_prompt(entry: LibraryEntry | None) -> str | None:
    if entry is None:
        return None
    metadata = entry.metadata or {}
    raw_model_summary = metadata.get("raw_model_summary")
    if isinstance(raw_model_summary, str) and raw_model_summary != "":
        return raw_model_summary
    execution_text = metadata.get("execution_text")
    if isinstance(execution_text, str) and execution_text:
        parsed_summary = extract_terminal_tag_or_none(execution_text, "summary")
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


def _parent_code_for_prompt(
    entry: LibraryEntry | None,
    *,
    solution_language: str,
    raw_score_label: str,
) -> str:
    if entry is None or not entry.solution.strip():
        raise ValueError("Execution prompt requires a selected library entry with non-empty solution code")
    fenced_language = "cpp" if solution_language.lower() in {"cpp", "c++", "cxx"} else "python"
    return (
        f"{_score_for_prompt(entry, raw_score_label=raw_score_label)}\n\n"
        f"<parent_code>\n```{fenced_language}\n{entry.solution.strip()}\n```\n</parent_code>"
    )


def _selected_candidate_for_guidance(
    entry: LibraryEntry | None,
    *,
    solution_language: str,
    raw_score_label: str,
) -> str:
    if entry is None or not entry.solution.strip():
        raise ValueError("Code-delta guidance prompt requires a selected library entry with non-empty solution code")
    fenced_language = "cpp" if solution_language.lower() in {"cpp", "c++", "cxx"} else "python"
    raw_summary = _raw_model_summary_for_prompt(entry) or "No model summary was emitted for this candidate."
    summary_semantics = summary_semantics_for_entry(entry)
    return f"""<selected_candidate>
<parent_code>
```{fenced_language}
{entry.solution.strip()}
```
</parent_code>

<change_summary>
Summary type: {summary_semantics}
{raw_summary}
</change_summary>

<score>
{_score_for_prompt(entry, raw_score_label=raw_score_label)}
</score>
</selected_candidate>"""


def build_guidance_prompt(
    *,
    problem_prompt: str,
    selected_node: LibraryNode,
    selected_entry: LibraryEntry | None,
    global_best_entries: list[LibraryEntry],
    local_failure_entries: list[LibraryEntry],
    objective_text: str | None = None,
    mechanism_constraint: str | None = None,
    raw_score_label: str = "Score",
    solution_language: str = "python",
    prompt_mode: str = PROMPT_MODE_SUMMARY_ONLY,
) -> Prompt:
    _ = selected_node, global_best_entries, local_failure_entries
    prompt_mode = normalize_prompt_mode(prompt_mode)
    objective = objective_text or (
        "Your task is to provide the next **evolutionary guidance** to reach a higher score."
    )
    executable_mechanism_constraint = mechanism_constraint or (
        "Every proposed mechanism must be implementable within the task's required self-contained solution using "
        "only information and resources available at execution time. Do not rely on offline training data, hidden "
        "benchmark access, external models or APIs, learned weights that are not supplied, or unavailable "
        "precomputation."
    )
    if prompt_mode == PROMPT_MODE_CODE_DELTA:
        selected_context = _selected_candidate_for_guidance(
            selected_entry,
            solution_language=solution_language,
            raw_score_label=raw_score_label,
        )
        context_intro = """The next section contains the selected candidate's complete code, its incremental
summary, and its verified score. Treat the code as the authoritative description
of the current algorithm. The change summary describes only how this candidate
changed from its own parent; it is not a complete description of the code."""
        history_instruction = (
            "Use `<parent_code>`, `<change_summary>`, and `<score>` to identify what the candidate currently "
            "implements, what its previous refinement changed, and what bottleneck the next attempt should address."
        )
    else:
        selected_context = (
            "<selected_summary>\n"
            f"{_raw_summary_for_prompt(selected_entry, fallback='No previous summary is attached.', raw_score_label=raw_score_label)}\n"
            "</selected_summary>"
        )
        context_intro = (
            "The next sections describe the current search state for this problem. Use them\n"
            "as run-local context when deciding the next step."
        )
        history_instruction = (
            "Use `<selected_summary>` to identify what has already been tried, what worked, and what bottleneck "
            "the next attempt should address."
        )

    user = f"""<problem>
{problem_prompt}
</problem>

{context_intro}

{selected_context}

# Objective
{objective}

# Evolutionary Guidelines
1. Analyze the search history.
   {history_instruction}

2. Stay at the algorithmic-strategy level.
   Propose high-level algorithmic directions, strategic refinements, or changes
   to important algorithmic components. The next attempt does not need to replace
   the current algorithm entirely: if the overall approach is promising, it is
   equally valuable to improve specific strategies, mechanisms, or design choices
   that may address the identified bottleneck. Do not write code, low-level
   implementation details, or parameter schedules.

3. Propose only executable mechanisms.
   {executable_mechanism_constraint}

4. Produce exactly the required XML structure.
   Provide internal reasoning and return exactly one `<guidance>` block and no other custom XML blocks, commentary, code, or markdown.

Please do internal reasoning and provide your response exactly in the following format:

<guidance>
Provide the final evolutionary guidance for the next execution attempt. Describe the main algorithmic direction and keep the guidance conceptual and actionable.
</guidance>
"""
    return Prompt(
        system=(
            "You are the Guidance Model, acting as a strategic navigator for an open-ended scientific "
            "discovery process.\n\n"
            "Your primary objective is to provide **evolutionary guidance**. Do not write final code "
            "or focus on low-level implementation details. Instead, your task is to propose high-level "
            "directional shifts, conceptual mutations, and novel pathways to explore the search space."
        ),
        user=user,
    )


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
    prompt_mode: str = PROMPT_MODE_SUMMARY_ONLY,
) -> Prompt:
    _ = global_best_entries
    prompt_mode = normalize_prompt_mode(prompt_mode)
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
    selected_parent = _parent_code_for_prompt(
        selected_entry,
        solution_language=solution_language,
        raw_score_label=raw_score_label,
    )
    if prompt_mode == PROMPT_MODE_CODE_DELTA:
        summary_contract = """Write a concise natural-language summary only of how the submitted solution differs from the supplied parent code.

The summary should explain:

1. what algorithmic mechanisms or strategies were concretely changed;
2. how those changes implement the supplied guidance;
3. which guidance-suggested components were simplified, approximated, or omitted.

Do not re-summarize the complete algorithm or list unchanged mechanisms, except when an unchanged invariant is essential to explain the modification. Describe only mechanisms that are actually present in the submitted solution. Do not include source code, code fences, copied constants, hard-coded arrays, raw candidate parameters, benchmark-specific profile values, or the output-format instructions."""
    else:
        summary_contract = """Write a concise natural-language summary of the candidate.

The summary should explain:

1. the implemented algorithmic idea;
2. exactly what was changed from the supplied parent code in response to the guidance and what important mechanisms were preserved;
3. the main search, refinement, or optimization mechanisms actually used.

Include enough information for a later model to understand the candidate’s overall algorithmic approach from the summary alone.

Only describe mechanisms that are present in the implementation. If a guidance-suggested component was not implemented, explicitly say that it was simplified, approximated, or omitted. Focus on conceptually important implementation choices and do not include source code."""

    user = f"""<problem>
{problem_prompt}
</problem>

The next sections provide the selected parent candidate and the guidance for improving it.

<selected_parent>
{selected_parent}
</selected_parent>

<guidance>
{prompt_guidance}
</guidance>

Use the problem statement as the authoritative task specification.
Use the selected parent code as the runnable baseline for this attempt.
Score direction: {score_direction}.

# Guidance Adherence Contract
Treat the guidance as the binding specification for improving the selected parent. Apply its proposed algorithmic mechanisms and strategic refinements as faithfully and completely as possible while preserving the parent candidate's working behavior outside the requested changes.

Modify the supplied parent code rather than replacing it with a generic baseline or an unrelated implementation. Preserve its input/output contract and unaffected working mechanisms. Every important actionable component in the guidance must be reflected concretely in the new solution and must interact as intended. Return one complete updated program, not a patch or diff.

{contract}

Your response must contain exactly three top-level XML blocks and no extra text before, between, or after them.
You must output all three XML blocks exactly as shown below.
The <execution_thinking>...</execution_thinking> block is mandatory and must use angle brackets.
The <solution> block is mandatory and must contain a fenced ```{fenced_language} code block.
The <summary>...</summary> block is mandatory and must be closed.
Do not output only execution_thinking, only a summary, or plain natural language.
Do not omit angle brackets from XML tags.

Required output format:

<execution_thinking>
A short explanation of how the guidance was converted into the submitted algorithm.
Do not include code.
</execution_thinking>

<solution>
```{fenced_language}
{placeholder}
```
</solution>

<summary>
{summary_contract}

</summary>

Any response that does not follow this exact three-block structure should be treated as invalid.
"""
    return Prompt(
        system=(
            f"You are the execution model. Improve the supplied parent {language_name} candidate by applying "
            "the guidance, then return one complete runnable candidate.\n\n"
            "Use the parent code as the implementation baseline. Follow the guidance faithfully, preserve "
            "unaffected working mechanisms, and keep the required input/output contract. Implement all specified "
            "algorithmic mechanisms and strategic details as precisely as possible. Do not silently omit, replace, "
            "or substantially simplify any "
            "important component. If an exact implementation is infeasible, use the closest valid alternative "
            "and explain the deviation in the summary.\n\n"
            "Output execution thinking first, then the code block, then the summary."
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


def extract_terminal_tag_or_none(text: str, tag: str) -> str | None:
    tagged = extract_tag_or_none(text, tag)
    if tagged is not None:
        return tagged

    open_tag = f"<{tag}>"
    start = text.rfind(open_tag)
    if start == -1:
        return None
    value_start = start + len(open_tag)
    tail = text[value_start:].strip()
    if not tail:
        return None
    if "<" in tail:
        return None
    return tail


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
