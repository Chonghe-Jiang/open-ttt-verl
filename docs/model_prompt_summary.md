# Guidance / Execution Prompt Source Summary

This file attaches the current source code that constructs the guidance model
and execution model prompts.

## Agent Loop Call Site

Source: `guidance_ttt/agent_loop.py`

````python
context = library.context_for_node(selected_node, visible_timestep_exclusive=int(global_step))
selected_entry = context["selected_entry"]
guidance_prompt = build_guidance_prompt(
    problem_prompt=problem_prompt,
    selected_node=selected_node,
    selected_entry=selected_entry,
    global_best_entries=context["global_best_entries"],
    local_failure_entries=context["local_failure_entries"],
    objective_text=task_spec.guidance_objective(
        task_spec.best_target(
            selected_node,
            selected_entry,
            context["global_best_entries"],
        )
    ),
    raw_score_label=task_spec.raw_score_label,
)
prompt_ids = await self.apply_chat_template(
    [
        {"role": "system", "content": guidance_prompt.system},
        {"role": "user", "content": guidance_prompt.user},
    ]
)
guidance_generation = await self._generate_guidance_response(prompt_ids, sampling_params)
response_ids = guidance_generation.response_ids
guidance_text = guidance_generation.text
guidance, guidance_format_ok = extract_guidance_or_format_error(guidance_text)

execution_prompt = build_execution_prompt(
    problem_prompt=problem_prompt,
    selected_node=selected_node,
    selected_entry=selected_entry,
    guidance=guidance,
    solution_language=task_spec.solution_language,
    solution_contract=task_spec.execution_solution_contract,
    raw_score_label=task_spec.raw_score_label,
)
````

## Shared Prompt Attachment Helpers

Source: `guidance_ttt/prompts.py`

````python
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
````

## Guidance Prompt Builder

Source: `guidance_ttt/prompts.py`

````python
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
1. Analyze the search history.
   Use `<selected_summary>` and `<local_failures>` to identify what has already been tried, what worked, what failed, and what bottleneck the next attempt should address.

2. Stay at the algorithmic-strategy level.
   Propose high-level algorithmic directions and ideas. Do not write code, implementation details, or parameter schedules.

3. Produce exactly the required XML structure.
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
            "directional shifts, conceptual mutations, and novel pathways to explore the search space.\n\n"
            "Focus on how the current ideas can *evolve* to escape local optima and discover "
            "fundamentally new mechanisms."
        ),
        user=user,
    )
````

## Execution Prompt Builder

Source: `guidance_ttt/prompts.py`

````python
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
{guidance}
</guidance>

Use the problem statement as the authoritative task specification.
Use the attached library context as historical evidence, not as code to copy blindly.
Score direction: {score_direction}.
Implement one concrete solution that follows the guidance while satisfying the problem specification.
{contract}

Your response must contain exactly three top-level XML blocks and no extra text before, between, or after them.

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
A concise natural-language summary of the candidate.

This summary must describe the implemented algorithm, the guidance-driven change from the prior idea, and the main search/refinement/optimization mechanisms used. If a suggested guidance component was not actually implemented, explicitly state that it was simplified or omitted. Mention implementation details only when they are conceptually important, such as placement ordering, orientation normalization, feasibility checks, local search, restart strategy, board-size selection, or constraint handling.

Do not include source code, code fences, copied constants, hard-coded arrays, raw candidate parameters, benchmark-specific profile values, or the output-format instructions themselves.

</summary>

Any response that does not follow this exact three-block structure should be treated as invalid.
"""
    return Prompt(
        system=(
            f"You are the execution model. Turn guidance into one concrete runnable {language_name} "
            "candidate. Output execution thinking first, then the code block, then the summary."
        ),
        user=user,
    )
````

## Task-Specific Prompt Configuration

Source: `guidance_ttt/tasks/__init__.py`

````python
def _erdos_objective(target: float | None) -> str:
    return (
        "Your task is to provide the next **evolutionary guidance** to beat the current visible "
        f"target raw score ({target}). Lower raw C5 is better."
    )


def _polyomino_objective(target: float | None) -> str:
    return (
        "Your task is to provide the next **evolutionary guidance** to beat the current visible "
        f"FrontierCS score target ({target}). Higher FrontierCS score is better."
    )


_TASKS: dict[str, TaskSpec] = {
    "erdos_min_overlap": TaskSpec(
        task_id="erdos_min_overlap",
        problem_prompt=ERDOS_PROBLEM_PROMPT,
        solution_language="python",
        execution_solution_contract=(
            "The <solution> block must contain one complete executable Python candidate in a ```python fenced block."
        ),
        score_direction="min",
        raw_score_label="Raw C5",
        create_root_node=create_erdos_root_node,
        verifier=_verify_erdos,
        solution_extractor=_extract_python_solution,
        guidance_objective=_erdos_objective,
    ),
    "polyomino_packing": TaskSpec(
        task_id="polyomino_packing",
        problem_prompt=POLYOMINO_PROBLEM_PROMPT,
        solution_language="cpp",
        execution_solution_contract=(
            "The <solution> block must contain one complete C++17 program in a ```cpp fenced block. "
            "It must read the Polyomino Packing instance from stdin and write the placement to stdout."
        ),
        score_direction="max",
        raw_score_label="FrontierCS score",
        create_root_node=create_polyomino_root_node,
        verifier=_verify_polyomino,
        solution_extractor=extract_cpp_solution_code,
        guidance_objective=_polyomino_objective,
    ),
}
````
