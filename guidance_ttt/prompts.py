from __future__ import annotations

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


def _entry_summary(entry: LibraryEntry | None, *, include_solution: bool) -> str:
    if entry is None:
        return "No previous library entry is attached."
    parts = [
        f"Entry id: {entry.id}",
        f"Summary: {_clip(entry.summary, 360)}",
        f"Previous guidance: {_clip(entry.guidance, 360)}",
        f"Previous execution thinking: {_clip(entry.execution_thinking, 240)}",
        f"Reward: {entry.verifier_reward}",
        f"Raw score: {entry.verifier_raw_score}",
        f"Verifier status: {entry.verifier_status}",
        f"Verifier message: {_clip(entry.verifier_message, 180)}",
        f"Reusable idea: {_clip(entry.reusable_idea, 220)}",
    ]
    if entry.failure_mode:
        parts.append(f"Failure mode: {entry.failure_mode}")
    if include_solution:
        solution = _clip(entry.solution, 1800)
        if solution:
            parts.append("Previous solution code excerpt:\n```python\n" + solution + "\n```")
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

The selected node's raw score is the target to beat. Lower raw C5 is better.
Do not recommend the constant h[i] = 0.5 baseline unless the selected node is invalid;
it is verifier-valid but gives no training signal when copied.
Non-binary asymmetric h values and deterministic local/numerical search are allowed.
Prefer small-n minimax constructions over large random perturbation code. Evidence from
local numerical checks suggests n_points in the 9..25 range with a mirror-symmetric
profile and active correlation constraints can reach C5 near 0.381. Ask the execution
model to minimize the maximum value of np.correlate(h, 1-h, mode="full") * (2/n)
under 0 <= h <= 1 and sum(h) == n/2. Avoid vague RL-reward advice; request concrete
deterministic minimax optimization steps that can be implemented inside run().
Do not propose alternating 0/1 patterns; the verifier scores them poorly.

The preferred submitted guidance is the text inside the XML block below. Thinking is allowed,
but if the XML block is missing, only text outside any <think>...</think> block will be used
as the submitted guidance.

Return exactly one block and nothing else:
<guidance>
Hypothesis: ...
Plan:
1. ...
2. ...
What to preserve: useful low-C5 nonconstant/mirror-symmetric structure from the selected entry
What to change to beat raw score {selected_node.raw_score}: scan small n, solve a minimax correlation problem, and return the best verifier-computed candidate
Expected verifier signal: lower raw C5 than {selected_node.raw_score}, ideally below 0.382
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

Target: produce a valid candidate with raw C5 lower than {selected_node.raw_score}.
The constant h[i] = 0.5 construction is only a baseline and should not be returned
unchanged. If previous solution code is shown, treat it as a reference point to beat,
not as code to copy. The verifier permits fractional, non-binary h values.
Your code must compute the actual c5_bound for the returned h. If the computed
c5_bound is not lower than the target, run a deterministic minimax search and return
the best candidate found. Do not use assert as the only way to satisfy constraints;
explicitly project or adjust h so sum(h) == n_points / 2 before returning. Use a
box-constrained projection or deterministic repair step after every perturbation so
the final returned h satisfies sum(h) == n_points / 2 to verifier tolerance.
Immediately before return, recompute h.sum() and c5_bound from the final h; do not
return candidates with residual sum error.

Implementation direction:
- First try scipy.optimize.minimize with method="SLSQP" on variables h and t.
- For each n, constrain sum(h) == n/2, 0 <= h <= 1, and every correlation value <= t.
- scan n_points from 9 to 25, with several deterministic mirror-symmetric initializations.
- Define a helper named project_to_box_sum(h, target) for final box-constrained projection.
- If scipy is unavailable or SLSQP fails, use deterministic projected local search with
  symmetric coordinate perturbations and the same c5_bound objective.
- Prefer mirror-symmetric fractional profiles over binary patterns.
- Do not return an alternating 0/1 construction; it looks attractive but has scored badly.
- The desired target C5 below 0.382 is realistic for this verifier.
- Return exactly return [float(x) for x in h], float(c5_bound), int(n_points);
  never return string-valued n_points, numpy scalar objects, or unprojected arrays.

Return runnable Python first. Keep reasoning and summary short. Do not claim verifier
success because the verifier has not run yet.

Return:
```python
def run(seed=42, budget_s=1, **kwargs):
    # final runnable solution
```

<summary>
Outcome hypothesis: ...
Reusable idea: ...
Risk / possible failure mode: ...
What future guidance should preserve: ...
What future guidance should change: ...
</summary>

<execution_thinking>
brief reasoning
</execution_thinking>
"""
    return Prompt(
        system=(
            "You are the execution model. Turn guidance into one concrete runnable Python "
            "candidate. Output the code block first, then concise metadata."
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
