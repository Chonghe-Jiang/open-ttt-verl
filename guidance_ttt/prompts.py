from __future__ import annotations

import re
from dataclasses import dataclass

from guidance_ttt.state import LibraryEntry, LibraryNode


@dataclass
class Prompt:
    system: str
    user: str


def _clip(text: str | None, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[truncated]"


_FENCED_CODE_RE = re.compile(r"```(?:python)?\s*([\s\S]*?)```")
_LONG_PROFILE_ARRAY_RE = re.compile(
    r"(\b(?:h|h_values|h_profile|profile)\s*=?\s*)\[[^\]]{120,}\]",
    flags=re.I,
)


def _compact_long_profile_arrays(text: str | None) -> str:
    return _LONG_PROFILE_ARRAY_RE.sub(
        r"\1[profile values omitted here; use canonical summary profile facts]",
        text or "",
    )


def _code_block_facts(code: str) -> str:
    facts: list[str] = []
    n_points = re.search(r"\bn_points\s*=\s*([0-9]+)", code)
    if n_points:
        facts.append(f"n_points={n_points.group(1)}")
    profile_match = re.search(
        r"\b(?:h\w*|profile\w*|base_profile|initial_profile)\s*=\s*np\.array\(\s*\[([\s\S]*?)\]",
        code,
    )
    if profile_match:
        values = re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:e[-+]?\d+)?", profile_match.group(1), flags=re.I)
        if 5 <= len(values) <= 64:
            if len(values) <= 12:
                profile_text = "profile h=[" + ", ".join(values) + "]"
            else:
                profile_text = (
                    f"profile h length={len(values)}, head=["
                    + ", ".join(values[:6])
                    + "], tail=["
                    + ", ".join(values[-4:])
                    + "]"
                )
            facts.append(profile_text)
    for helper in ("project_to_box_sum", "c5_score"):
        if helper in code:
            facts.append(f"uses {helper}")
    if "np.correlate" in code:
        facts.append("scores with np.correlate verifier normalization")
    if "SLSQP" in code:
        facts.append("uses SLSQP")
    maxiter = re.search(r"['\"]maxiter['\"]\s*:\s*([0-9]+)", code)
    if maxiter:
        facts.append(f"SLSQP maxiter={maxiter.group(1)}")
    ftol = re.search(r"['\"]ftol['\"]\s*:\s*([0-9.eE+-]+)", code)
    if ftol:
        facts.append(f"SLSQP ftol={ftol.group(1)}")
    if "bounds=" in code or "bounds =" in code:
        facts.append("uses box bounds")
    if "constraints=" in code or "constraints =" in code:
        facts.append("uses equality sum constraint")
    loops = []
    for value in re.findall(r"range\((\d+)\)", code):
        if value not in loops:
            loops.append(value)
    if loops:
        facts.append("loop counts=" + ",".join(loops[:4]))
    constants = []
    for value in re.findall(r"(?<![A-Za-z0-9_.])\d+(?:\.\d+)?e[-+]\d+", code):
        if value not in constants:
            constants.append(value)
    if constants:
        facts.append("scientific constants=" + ",".join(constants[:8]))
    if "default_rng" in code or "np.random.seed" in code:
        facts.append("uses deterministic seed/RNG")
    if not facts:
        return "[code omitted; no compact facts extracted]"
    return "[code omitted; extracted implementation facts: " + "; ".join(facts) + "]"


def _summary_for_guidance(summary: str | None) -> str:
    summary = (summary or "").strip()
    if not summary:
        return ""
    extracted_facts: list[str] = []

    def replace_code(match: re.Match[str]) -> str:
        extracted_facts.append(_code_block_facts(match.group(1)))
        return "[code omitted; see extracted attached-code facts above]"

    summary_without_code = _FENCED_CODE_RE.sub(replace_code, summary)
    if not extracted_facts:
        return summary_without_code
    facts_text = "\n".join(f"- {fact}" for fact in extracted_facts)
    return "Extracted attached-code facts:\n" + facts_text + "\n\nSummary text:\n" + summary_without_code


def _verified_profile_artifact_facts(entry: LibraryEntry) -> str:
    artifacts = (entry.metadata or {}).get("verification_artifacts") or {}
    h_values = artifacts.get("h_values")
    if entry.verifier_status != "valid" or not isinstance(h_values, list) or not h_values:
        return ""
    try:
        values = [float(value) for value in h_values]
    except (TypeError, ValueError):
        return ""
    if len(values) > 32:
        values_text = (
            f"h length={len(values)}, head=["
            + ", ".join(repr(value) for value in values[:6])
            + "], tail=["
            + ", ".join(repr(value) for value in values[-4:])
            + "]"
        )
    else:
        values_text = "h=[" + ", ".join(repr(value) for value in values) + "]"
    n_points = artifacts.get("n_points", len(values))
    c5_bound = artifacts.get("c5_bound", entry.verifier_raw_score)
    raw_score = entry.verifier_raw_score if entry.verifier_raw_score is not None else c5_bound
    return (
        "Verified returned profile artifacts (authoritative initialization): "
        f"n_points={int(n_points)}, raw C5={float(raw_score)!r}, "
        f"c5_bound={float(c5_bound)!r}, {values_text}"
    )


def _h_values_facts(values: list[float]) -> str:
    if len(values) > 32:
        return (
            f"h length={len(values)}, head=["
            + ", ".join(repr(value) for value in values[:6])
            + "], tail=["
            + ", ".join(repr(value) for value in values[-4:])
            + "]"
        )
    return "h=[" + ", ".join(repr(value) for value in values) + "]"


def _root_initial_construction_facts(node: LibraryNode) -> str:
    if node.parent_id is not None:
        return ""
    metadata = node.metadata or {}
    h_values = metadata.get("h_values")
    if not isinstance(h_values, list) or not h_values:
        return ""
    try:
        values = [float(value) for value in h_values]
    except (TypeError, ValueError):
        return ""
    n_points = metadata.get("n_points", len(values))
    c5_bound = metadata.get("c5_bound", node.raw_score)
    initialization = metadata.get("initialization", "initial_construction")
    return (
        "Current initial construction (reference state to improve): "
        f"initialization={initialization}, n_points={int(n_points)}, "
        f"raw C5={float(node.raw_score)!r}, c5_bound={float(c5_bound)!r}, "
        f"{_h_values_facts(values)}"
    )


def _entry_summary(entry: LibraryEntry | None, *, include_solution: bool) -> str:
    if entry is None:
        return "No previous library entry is attached."
    summary_text = entry.summary if include_solution else _summary_for_guidance(entry.summary)
    summary_clip = 2600 if include_solution else 420
    guidance_clip = 360 if include_solution else 30
    verifier_clip = 180 if include_solution else 60
    idea_clip = 220 if include_solution else 40
    parts = [
        f"Entry id: {entry.id}",
        f"Reward: {entry.verifier_reward}",
        f"Raw score: {entry.verifier_raw_score}",
        f"Verifier status: {entry.verifier_status}",
        f"Verifier message: {_clip(entry.verifier_message, verifier_clip)}",
    ]
    profile_facts = _verified_profile_artifact_facts(entry)
    if profile_facts:
        parts.append(profile_facts)
    parts.extend(
        [
            f"Canonical summary:\n{_clip(summary_text, summary_clip)}",
            f"Previous guidance: {_clip(_compact_long_profile_arrays(entry.guidance), guidance_clip)}",
        ]
    )
    parts.extend(
        [
            f"Reusable idea: {_clip(entry.reusable_idea, idea_clip)}",
        ]
    )
    if entry.failure_mode:
        parts.append(f"Failure mode: {entry.failure_mode}")
    return "\n".join(parts)


def _summary_only_for_guidance(entry: LibraryEntry | None) -> str:
    if entry is None:
        return "No previous summary is attached."
    return _clip(_summary_for_guidance(entry.summary), 420) or "No previous summary is attached."


def build_guidance_prompt(
    *,
    problem_prompt: str,
    selected_node: LibraryNode,
    selected_entry: LibraryEntry | None,
    global_best_entries: list[LibraryEntry],
    local_failure_entries: list[LibraryEntry],
) -> Prompt:
    selected_entry_id = selected_entry.id if selected_entry is not None else None
    best_entries_for_prompt = [
        entry for entry in global_best_entries if entry is not None and entry.id != selected_entry_id
    ]
    best_text = "\n\n".join(_summary_only_for_guidance(entry) for entry in best_entries_for_prompt)
    if not best_text and global_best_entries and selected_entry_id:
        best_text = "The selected summary is also the current global best visible summary."
    failure_text = "\n\n".join(_summary_only_for_guidance(entry) for entry in local_failure_entries)
    best_valid = _best_valid_entry(global_best_entries, selected_entry)
    best_valid_raw_score = (
        best_valid.verifier_raw_score
        if best_valid is not None and best_valid.verifier_raw_score is not None
        else selected_node.raw_score
    )
    best_valid_target = best_valid_raw_score if best_valid_raw_score is not None else selected_node.raw_score
    user = f"""<problem>
{problem_prompt}
</problem>

The next sections describe the current search state for this problem. Use them
as run-local context when deciding the next step.

<selected_summary>
{_summary_only_for_guidance(selected_entry)}
</selected_summary>

<global_best>
{best_text or "No global best summary yet."}
</global_best>

<local_failures>
{failure_text or "No local failure summaries yet."}
</local_failures>

# Objective
Your task is to provide the next **evolutionary guidance** to beat the current visible target raw score ({best_valid_target}). Lower raw C5 is better.

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


def _best_valid_entry(*entry_groups: list[LibraryEntry] | LibraryEntry | None) -> LibraryEntry | None:
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
    return min(valid_entries, key=lambda entry: float(entry.verifier_raw_score))


def build_execution_prompt(
    *,
    problem_prompt: str,
    selected_node: LibraryNode,
    selected_entry: LibraryEntry | None,
    global_best_entries: list[LibraryEntry] | None = None,
    guidance: str,
) -> Prompt:
    best_valid = _best_valid_entry(global_best_entries or [], selected_entry)
    best_valid_text = _entry_summary(best_valid, include_solution=True) if best_valid else "No global best valid entry yet."
    initial_construction_text = _root_initial_construction_facts(selected_node)
    user = f"""<problem>
{problem_prompt}
</problem>

The next sections describe the current search state for this problem. Use them
as run-local context when implementing the guided candidate.

<selected_library_node>
Node id: {selected_node.id}
Timestep: {selected_node.timestep}
Value: {selected_node.value}
Raw score: {selected_node.raw_score}
{initial_construction_text}
{_entry_summary(selected_entry, include_solution=True)}
</selected_library_node>

<global_best_valid_entry>
{best_valid_text}
</global_best_valid_entry>

<guidance>
{guidance}
</guidance>

Use the problem statement as the authoritative task specification.
Use the attached library context as historical evidence, not as code to copy blindly.
Implement one concrete solution that follows the guidance while satisfying the problem specification.

Return exactly these three blocks:

<execution_thinking>
Briefly explain how the guidance was translated into the submitted solution.
</execution_thinking>

<solution>
```python
# complete executable solution required by the problem
```
</solution>

<summary>
Use natural language to summarize the overall idea and method of the solution. Explain how the candidate was generated, including the search, refinement, or optimization strategy used. If the solution’s main contribution lies in specific implementation details, such as parameter fine-tuning, threshold adjustment, normalization choices, perturbation design, or constraint-handling tricks, explicitly emphasize those details and explain why they matter. Do not include code, hard-coded arrays, copied profile values, or raw candidate parameters.
</summary>
"""
    return Prompt(
        system=(
            "You are the execution model. Turn guidance into one concrete runnable Python "
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
