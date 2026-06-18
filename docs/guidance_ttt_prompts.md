# Guidance TTT Prompts

This document summarizes every prompt used by the current Erdos Guidance TTT
pipeline and how each prompt is constructed at runtime.

Source files:

- `guidance_ttt/tasks/erdos.py`: problem prompt.
- `guidance_ttt/prompts.py`: guidance prompt, execution prompt, minimax template.
- `guidance_ttt/agent_loop.py`: runtime construction, model calls, verification, metadata.

## Runtime Construction Flow

One rollout has two model calls:

1. The trainable guidance model receives a chat prompt and returns high-level
   guidance in a `<guidance>...</guidance>` block.
2. The execution model receives the Erdos problem, selected library context,
   and parsed guidance, then returns runnable Python plus summary metadata.

The agent loop builds the prompts as follows:

```text
run(...)
  extra_info.library_path -> GuidanceLibrary(...)
  global_step + uid -> group_uid = "{global_step}:{uid}"
  library.acquire_group(group_uid, visible_timestep_exclusive=global_step)
    -> selected_node
  library.context_for_node(selected_node, visible_timestep_exclusive=global_step)
    -> selected_entry
    -> global_best_entries
    -> local_failure_entries

  build_guidance_prompt(
    problem_prompt=ERDOS_PROBLEM_PROMPT,
    selected_node=selected_node,
    selected_entry=selected_entry,
    global_best_entries=global_best_entries,
    local_failure_entries=local_failure_entries,
  )

  apply_chat_template([
    {"role": "system", "content": guidance_prompt.system},
    {"role": "user", "content": guidance_prompt.user},
  ])

  guidance_model.generate(...)
  extract_guidance_or_format_error(raw_guidance_text)
    if <guidance>...</guidance> exists: use it, format_ok=True
    else: use text outside <think>...</think>, format_ok=False
    else: synthesize a formatting-failure guidance, format_ok=False

  build_execution_prompt(
    problem_prompt=ERDOS_PROBLEM_PROMPT,
    selected_node=selected_node,
    selected_entry=selected_entry,
    guidance=parsed_guidance,
  )

  execution_client.complete(
    system=execution_prompt.system,
    user=execution_prompt.user,
    model=execution_llm_config.model,
    temperature=execution_llm_config.temperature,
    max_tokens=execution_llm_config.max_tokens,
  )

  verify_erdos_solution_text(execution_text)
    if valid: store execution_text
    else: store the invalid execution status directly; no automatic solution fallback is used
```

Each library entry stores the exact prompts and outputs in metadata:

```json
{
  "guidance_prompt": {"system": "...", "user": "..."},
  "raw_guidance_text": "...",
  "raw_guidance_with_specials": "...",
  "guidance_format_ok": true,
  "execution_prompt": {"system": "...", "user": "..."},
  "execution_text": "...",
  "original_execution_text": "...",
  "execution_fallback_used": false,
  "execution_fallback_reason": null
}
```

Run summaries and best-rollout exports copy these exact fields from `library.json`.

## Shared Entry Summary Construction

`_entry_summary(entry, include_solution=...)` is injected into both guidance
and execution prompts.

If no previous entry exists:

```text
No previous library entry is attached.
```

If an entry exists, the following fields are included:

```text
Entry id: {entry.id}
Summary: {entry.summary clipped to 360 chars}
Previous guidance: {entry.guidance clipped to 360 chars}
Previous execution thinking: {entry.execution_thinking clipped to 240 chars}
Reward: {entry.verifier_reward}
Raw score: {entry.verifier_raw_score}
Verifier status: {entry.verifier_status}
Verifier message: {entry.verifier_message clipped to 180 chars}
Reusable idea: {entry.reusable_idea clipped to 220 chars}
Failure mode: {entry.failure_mode, only if present}
```

For the execution prompt only, `include_solution=True` also appends:

````text
Previous solution code excerpt:
```python
{entry.solution clipped to 1800 chars}
```
````

## Problem Prompt

This is `ERDOS_PROBLEM_PROMPT`.

```text
You are solving the Erdos minimum overlap problem.

Find a step function h: [0, 2] -> [0, 1] that minimizes:

C5 = max_k integral h(x)(1 - h(x+k)) dx

Discretize h as n_points samples over [0, 2]. The verifier expects Python code
with:

def run(seed=42, budget_s=..., **kwargs):
    return (h_values, c5_bound, n_points)

Constraints:
- 0 <= h[i] <= 1
- sum(h) == n_points / 2
- c5_bound must match max(np.correlate(h, 1-h, mode="full") * (2.0 / n_points))

Lower raw C5 is better. The reward used for RL is 1 / (1e-8 + C5).
```

## Guidance Model Prompt

### Guidance System Prompt

```text
You are the trainable guidance model. Produce high level ideas, not final code. You may think first, but the final submitted guidance should be in one <guidance>...</guidance> block.
```

### Guidance User Prompt Template

```text
<problem>
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
{global_best entry summaries, or "No global best entry yet."}
</global_best>

<local_failures>
{local failure entry summaries, or "No local failure entries yet."}
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
```

### Guidance Parsing Rule

```text
If raw output has <guidance>...</guidance>:
  submitted guidance = text inside that tag
  guidance_format_ok = true
Else if raw output has text outside <think>...</think>:
  submitted guidance = outside-think text
  guidance_format_ok = false
Else:
  submitted guidance = synthetic formatting failure text
  guidance_format_ok = false
```

The synthetic formatting failure text is:

```text
Hypothesis: The guidance model did not emit a valid <guidance> block.
Plan:
1. Treat this attempt as a formatting failure because no text was found outside the thinking block.
2. Retry with an explicit tagged guidance response on the next rollout.
What to preserve: The selected library context and Erdos verifier constraints.
What to change: Emit exactly one tagged guidance block after any thinking.
Expected verifier signal: formatting_error
```

## Execution Model Prompt

### Execution System Prompt

```text
You are the execution model. Turn guidance into one concrete runnable Python candidate. Output the code block first, then concise metadata.
```

### Execution User Prompt Template

````text
<problem>
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

Known-good implementation skeleton. You may copy and adapt this exact structure:
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
````

## Known-Good Minimax Execution Template

This template is embedded inside the execution prompt as a reference skeleton.
It is not automatically substituted after a failed execution.

```python
import numpy as np

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
```

## No Automatic Execution Fallback

If the execution model call fails, returns empty text, returns unparseable code,
or produces a verifier-invalid result, that failure is stored directly in the
library entry. The agent loop no longer replaces failed output with the minimax
template.

Failure metadata:

```json
{
  "execution_fallback_used": false,
  "execution_fallback_reason": null,
  "original_execution_text": "{execution model raw output}",
  "execution_text": "{same execution model raw output}",
  "verifier_status": "parse_error | execution_error | invalid | valid"
}
```

## Prompt Design Changes In The Current Version

Compared with the earlier version, the prompts now push the model toward:

- Small `n_points` in the 9..25 range instead of broad random search.
- Fractional, non-binary, mirror-symmetric profiles.
- Direct minimax optimization of `max(np.correlate(h, 1-h, mode="full") * (2/n))`.
- Strict `sum(h) == n_points / 2` repair with `project_to_box_sum`.
- Avoiding constant 0.5, alternating 0/1, vague RL-reward advice, and unverified claims.
- A no-fallback verifier path, so invalid execution outputs become real training
  signals instead of being replaced by a fixed candidate.
