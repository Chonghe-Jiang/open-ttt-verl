from __future__ import annotations

from dataclasses import dataclass

from guidance_ttt.state import LibraryEntry, LibraryNode


@dataclass
class Prompt:
    system: str
    user: str


ERDOS_MINIMAX_EXECUTION_TEMPLATE = '''import numpy as np

def project_to_box_sum(h, target):
    h = np.clip(np.asarray(h, dtype=float), 0.0, 1.0)
    for _ in range(100):
        diff = float(target - h.sum())
        if abs(diff) < 1e-12:
            break
        free = (h > 1e-12) & (h < 1.0 - 1e-12)
        if not np.any(free):
            free = np.ones_like(h, dtype=bool)
        h[free] += diff / float(np.count_nonzero(free))
        h = np.clip(h, 0.0, 1.0)
    return h

def c5_score(h):
    n = int(len(h))
    return float(np.max(np.correlate(h, 1.0 - h, mode="full") * (2.0 / n)))

def run(seed=42, budget_s=1, **kwargs):
    n_points = 19
    target = n_points / 2.0
    h = np.array([
        0.9999877603639307, 0.9871689928539907, 0.6538052901835218,
        0.26496040864134757, 0.7921008518715473, 0.22244708724960485,
        0.43594987520258144, 0.31249596343709657, 0.013288180619505931,
        0.10447938652930353, 0.04501365418702798, 0.3119089519044072,
        0.4358802114412871, 0.22261948541845558, 0.7918283420600218,
        0.268186249932381, 0.6427085234261372, 0.9952347090654273,
        0.9999360756124249,
    ], dtype=float)
    h = project_to_box_sum(h, target)
    best_h = h.copy()
    best_c5 = c5_score(best_h)

    try:
        from scipy.optimize import minimize

        constraints = ({"type": "eq", "fun": lambda x: float(np.sum(x) - target)},)
        result = minimize(
            c5_score,
            best_h,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * n_points,
            constraints=constraints,
            options={"maxiter": 300, "ftol": 1e-13, "disp": False},
        )
        if result.success:
            candidate = project_to_box_sum(result.x, target)
            candidate_c5 = c5_score(candidate)
            if candidate_c5 <= best_c5:
                best_h = candidate
                best_c5 = candidate_c5
    except Exception:
        pass

    best_h = project_to_box_sum(best_h, target)
    c5_bound = c5_score(best_h)
    return [float(x) for x in best_h], float(c5_bound), int(n_points)
'''


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

The selected node's raw score is a local reference. The target to beat is the
current best valid raw score when one is visible. Lower raw C5 is better.
Do not recommend the constant h[i] = 0.5 baseline unless the selected node is invalid;
it is verifier-valid but gives no training signal when copied.
Non-binary asymmetric h values and deterministic local/numerical search are allowed.
Avoid vague RL-reward advice; request concrete deterministic improvement steps that
can be implemented inside run(). Do not propose alternating 0/1 patterns; the verifier
scores them poorly.

The guidance should make the execution model do controlled improvement, not restart
from scratch:
- Preserve the current best valid construction with raw_score = {best_valid_target}.
- Do not restart from the constant baseline.
- Use the best valid h profile as the initialization.
- Include known information directly in the guidance when useful: current best raw
  score, best valid solution excerpt/profile, verifier constraints, verifier
  normalization, and the strict improvement target.
- Search only small deterministic perturbations around it while preserving:
  1. 0 <= h[i] <= 1
  2. sum(h) = n_points / 2
  3. the same verifier normalization
  4. mirror/complement symmetry if present in the current best profile
- Optimize only the independent half of the variables.
- Use a small local search such as coordinate perturbation, projected line search,
  or SLSQP if available.
- For each candidate, immediately evaluate using the official verifier and keep the
  lowest raw C5.
- Use n_points in {{9, 11, 13, 15, 17, 21, 25}}; stop if no improvement after a small
  fixed number of trials.
- Target: strictly improve over {best_valid_target}, not merely beat 0.5.

The preferred submitted guidance is the text inside the XML block below. Thinking is allowed,
but if the XML block is missing, only text outside any <think>...</think> block will be used
as the submitted guidance.

Return exactly one block and nothing else:
<guidance>
Preserve the current best valid construction with raw_score = {best_valid_target}.
Do not restart from the constant baseline.
Include known information directly in the guidance: current best raw score, best valid
solution/profile, verifier constraints, verifier normalization, and the strict target.

Hypothesis: controlled local perturbations around the current best valid h can strictly
improve raw C5 without losing verifier validity.
Plan:
1. Use the best valid h profile as initialization, not the 0.5 baseline.
2. Search only small deterministic perturbations that preserve box, sum, verifier normalization, and symmetry constraints.
3. Optimize only the independent half variables and immediately score each candidate with the official verifier.
What to preserve: current best valid low-C5 structure and any mirror/complement symmetry
What to change to beat raw score {best_valid_target}: make controlled local repairs/perturbations around the best valid profile
Expected verifier signal: strictly lower raw C5 than {best_valid_target}
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
    target_raw_score = (
        best_valid.verifier_raw_score
        if best_valid is not None and best_valid.verifier_raw_score is not None
        else selected_node.raw_score
    )
    best_valid_text = _entry_summary(best_valid, include_solution=True) if best_valid else "No global best valid entry yet."
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

<global_best_valid_solution>
{best_valid_text}
</global_best_valid_solution>

<guidance>
{guidance}
</guidance>

Target: produce a valid candidate with raw C5 lower than {target_raw_score}.
The constant h[i] = 0.5 construction is only a baseline and should not be returned
unchanged. If previous solution code is shown, treat it as a reference point to beat,
not as code to copy. The verifier permits fractional, non-binary h values.
Your code must compute the actual c5_bound for the returned h. If the computed
c5_bound is not lower than the target, run a deterministic local search around the
best previous valid profile and return the best candidate found. Do not use assert as the only way to satisfy constraints;
explicitly project or adjust h so sum(h) == n_points / 2 before returning. Use a
box-constrained projection or deterministic repair step after every perturbation so
the final returned h satisfies sum(h) == n_points / 2 to verifier tolerance.
Immediately before return, recompute h.sum() and c5_bound from the final h; do not
return candidates with residual sum error.

Implementation direction:
- Initialize from this global best valid solution when available; otherwise use the
  selected previous solution code when it is valid and available.
- Use previous solution code as the initialization when it is valid and available.
- Search only small deterministic perturbations around the current best profile.
- Use n_points in {{9, 11, 13, 15, 17, 21, 25}} and preserve n_points when reusing
  a previous valid profile.
- Define a helper named project_to_box_sum(h, target) for final box-constrained projection.
- Use deterministic projected local search with symmetric coordinate perturbations and
  the same c5_bound objective; SLSQP is acceptable only as a bounded local refinement.
- Optimize only the independent half of variables when mirror/complement symmetry is
  present in the current best profile.
- Prefer mirror-symmetric fractional profiles over binary patterns.
- Do not return an alternating 0/1 construction; it looks attractive but has scored badly.
- The desired target is a strict improvement over the current best raw C5, not merely
  beating 0.5.
- Return exactly return [float(x) for x in h], float(c5_bound), int(n_points);
  never return string-valued n_points, numpy scalar objects, or unprojected arrays.

Known-good implementation skeleton for verifier-compatible scoring and projection.
Use it as a scoring/repair reference:
<known_good_minimax_template>
```python
{ERDOS_MINIMAX_EXECUTION_TEMPLATE}
```
</known_good_minimax_template>

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
