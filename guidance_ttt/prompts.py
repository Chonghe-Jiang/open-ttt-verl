from __future__ import annotations

import re
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
    n_points = 63
    target = n_points / 2.0
    h = np.array([
        2.7755575615628914e-17, 2.7755575615628914e-17, 1.293663231381232e-16,
        2.7755575615628914e-17, 0.0131852640431291, 2.7755575615628914e-17,
        2.7755575615628914e-17, 0.26212656632295017, 0.8518260937824772,
        0.7892316177941204, 0.5257445487871693, 0.37585906591985196,
        0.3005418542306866, 0.1574795850486234, 0.4373323751700511,
        0.7429601749522273, 0.7750101335843718, 0.498685678735848,
        0.4580696661912471, 0.5793512512721171, 0.6885196496503867,
        0.4812942335026287, 0.5538085240635235, 0.8242838767355658,
        0.8992749542536581, 0.7465362244428376, 0.7197319122277341,
        0.7756815030976699, 0.9999999999999998, 1.0, 0.7912613201330126,
        0.9388727802180509, 1.0, 1.0, 0.9999999999999999,
        0.6324778951288466, 0.7197319122277515, 0.7644975815058611,
        0.8813135971906164, 0.8242838767355828, 0.5538085240635041,
        0.4812942335026238, 0.6885196496503996, 0.5793512512721313,
        0.458069666191245, 0.4986856787358537, 0.775010133584365,
        0.7429601749522213, 0.4373323751700534, 0.1574795850486218,
        0.30054185423067215, 0.3758590659198784, 0.5257445487871585,
        0.7892316177941104, 0.8518260937824886, 0.26212656632293563,
        1.6091794829207338e-16, 2.7755575615628914e-17, 0.01318526404314024,
        7.860053295334456e-17, 2.7755575615628914e-17,
        2.7755575615628914e-17, 2.7755575615628914e-17,
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


KNOWN_GOOD_ERDOS_RAW_C5 = 0.3810181186942784
KNOWN_GOOD_ERDOS_H_PROFILE = (
    "2.7755575615628914e-17, 2.7755575615628914e-17, 1.293663231381232e-16, "
    "2.7755575615628914e-17, 0.0131852640431291, 2.7755575615628914e-17, "
    "2.7755575615628914e-17, 0.26212656632295017, 0.8518260937824772, "
    "0.7892316177941204, 0.5257445487871693, 0.37585906591985196, "
    "0.3005418542306866, 0.1574795850486234, 0.4373323751700511, "
    "0.7429601749522273, 0.7750101335843718, 0.498685678735848, "
    "0.4580696661912471, 0.5793512512721171, 0.6885196496503867, "
    "0.4812942335026287, 0.5538085240635235, 0.8242838767355658, "
    "0.8992749542536581, 0.7465362244428376, 0.7197319122277341, "
    "0.7756815030976699, 0.9999999999999998, 1.0, 0.7912613201330126, "
    "0.9388727802180509, 1.0, 1.0, 0.9999999999999999, "
    "0.6324778951288466, 0.7197319122277515, 0.7644975815058611, "
    "0.8813135971906164, 0.8242838767355828, 0.5538085240635041, "
    "0.4812942335026238, 0.6885196496503996, 0.5793512512721313, "
    "0.458069666191245, 0.4986856787358537, 0.775010133584365, "
    "0.7429601749522213, 0.4373323751700534, 0.1574795850486218, "
    "0.30054185423067215, 0.3758590659198784, 0.5257445487871585, "
    "0.7892316177941104, 0.8518260937824886, 0.26212656632293563, "
    "1.6091794829207338e-16, 2.7755575615628914e-17, 0.01318526404314024, "
    "7.860053295334456e-17, 2.7755575615628914e-17, "
    "2.7755575615628914e-17, 2.7755575615628914e-17"
)

KNOWN_GOOD_ERDOS_PROGRESS_LADDER = (
    "0.4821225382225115 -> 0.3949181335382675 -> 0.3853523965077289 -> "
    "0.3836709166860949 -> 0.3823673633758575 -> 0.38159399612835 -> "
    "0.3813987425845457 -> 0.3813386418994406 -> 0.3812922283535478 -> "
    "0.38114264344042986 -> 0.3811360647897058 -> 0.3810928200855021 -> "
    "0.38106453367736337 -> 0.3810633365287946 -> 0.3810215348642786 -> "
    "0.3810206626281053 -> 0.3810193840681338 -> 0.3810190847194418 -> "
    "0.3810181186942784"
)


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
        return "[code omitted; see extracted attached-solution facts above]"

    summary_without_code = _FENCED_CODE_RE.sub(replace_code, summary)
    if not extracted_facts:
        return summary_without_code
    facts_text = "\n".join(f"- {fact}" for fact in extracted_facts)
    return "Extracted attached-solution facts:\n" + facts_text + "\n\nSummary text:\n" + summary_without_code


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
    if include_solution and entry.solution and "```python" not in (entry.summary or ""):
        solution = _clip(entry.solution, 1800)
        if solution:
            parts.append("Previous solution code excerpt (backward-compatible):\n```python\n" + solution + "\n```")
    return "\n".join(parts)


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
    best_text = "\n\n".join(_entry_summary(entry, include_solution=False) for entry in best_entries_for_prompt)
    if not best_text and global_best_entries and selected_entry_id:
        best_text = "The selected library node is also the current global best visible entry; use its summary above."
    failure_text = "\n\n".join(_entry_summary(entry, include_solution=False) for entry in local_failure_entries)
    best_valid = _best_valid_entry(global_best_entries, selected_entry)
    best_valid_raw_score = (
        best_valid.verifier_raw_score
        if best_valid is not None and best_valid.verifier_raw_score is not None
        else selected_node.raw_score
    )
    if best_valid is None and (
        best_valid_raw_score is None or float(best_valid_raw_score) > KNOWN_GOOD_ERDOS_RAW_C5
    ):
        best_valid_target = KNOWN_GOOD_ERDOS_RAW_C5
    else:
        best_valid_target = best_valid_raw_score if best_valid_raw_score is not None else selected_node.raw_score
    aspiration_target = 0.381019
    if best_valid is None:
        root_default_text = f"""- If no previous valid summary/profile is visible, use the known-good 63-point
  fractional seed as the default starting point, not a newly invented small-n profile:
  n_points=63, raw C5 approximately {KNOWN_GOOD_ERDOS_RAW_C5}, project_to_box_sum, c5_score with
  np.correlate(..., mode="full") * (2.0 / n), SLSQP maxiter=300, ftol=1e-13, box
  bounds, equality sum constraint, and h=[{KNOWN_GOOD_ERDOS_H_PROFILE}]. This seed
  came from an iterative 63-point improvement ladder with raw C5 values
  {KNOWN_GOOD_ERDOS_PROGRESS_LADDER}. Do not
  propose n_points=9, a linear ramp, a cosine profile, or any fresh symmetric toy
  profile in the root/default case."""
    else:
        root_default_text = f"""- If no previous valid summary/profile is visible, use the known-good 63-point
  fractional seed, not a newly invented small-n profile: n_points=63, raw C5 approximately {KNOWN_GOOD_ERDOS_RAW_C5},
  project_to_box_sum, c5_score with np.correlate(..., mode="full") * (2.0 / n),
  SLSQP maxiter=300, ftol=1e-13, box bounds, equality sum constraint. This seed
  came from an iterative 63-point improvement ladder with raw C5 values
  {KNOWN_GOOD_ERDOS_PROGRESS_LADDER}. Do not
  propose n_points=9, a linear ramp, a cosine profile, or any fresh symmetric toy
  profile."""
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
from scratch. It must conclude the previous summary before proposing a delta:
- Preserve the current best valid construction with raw_score = {best_valid_target}.
- Do not restart from the constant baseline.
- Use the best valid h profile as the initialization.
{root_default_text}
- Inherit concrete selected/global details: n_points, raw C5, h/profile source,
  step sizes, iteration counts, seeds, projection helper names, optimizer choices,
  bounds, equality constraints, symmetry assumptions, and known failure modes.
  Do not collapse actual values into vague phrases.
- If "Verified returned profile artifacts" are present, they are the authoritative
  best initialization. Restate their n_points, raw C5, and exact h=[...] values in
  Previous Summary Facts or Concrete Search Recipe, then propose a delta around that
  actual returned profile rather than around the original seed.
- If extracted attached-solution facts contain concrete profile facts such as h head,
  tail, length, or h=[...], restate those profile facts in Previous Summary Facts or
  Concrete Search Recipe. The initialization must not be reduced to "known-good seed"
  when numeric profile facts are visible.
- Head/tail profile facts are summaries, not a complete vector. Do not concatenate
  head and tail values to invent a full h array. If full h is not visible, say to
  initialize by rerunning/reusing the previous verified solution code or verified
  returned profile, then perturb that actual profile.
- Treat the summary's Next Guidance Delta as the first candidate delta, then sharpen it
  into executable instructions.
- The training signal should show step-to-step improvement. Name the parent/selected
  raw C5 and current best valid raw C5, then propose a delta intended to lower both.
  If the previous step did not improve, change at least two search knobs and state why.
- Step-to-step progress is the primary objective. A rollout that returns the inherited
  profile unchanged is a failed experiment even if it is verifier-valid. Do not frame
  copying the current best as the safe choice; instead force the next rollout to test
  a concrete non-copy candidate family and report which deltas were rejected.
- Plateau guard / active-lag escape: if the selected/global best raw C5 equals a visible previous best,
  do not return the same profile and do not say improvement is impossible. Use a
  two-family escape: first repeat the best known coordinate/pair sweep with a smaller
  schedule [5e-5, 2e-5, 1e-5, 5e-6, 1e-6], then compute the active lag attaining
  max(np.correlate(h, 1-h, mode="full") * 2/n), list the top contributing pairs
  (i, i-lag), try signed mass transfers on those pairs with deltas [1e-6, 5e-7, 1e-7],
  project/evaluate after each move, and run SLSQP from the top distinct candidates.
- Strong-seed guard: if the current best is at or below 0.38103, assume it came from
  the known 63-point improvement ladder, whose verified milestones were
  {KNOWN_GOOD_ERDOS_PROGRESS_LADDER}. Do not regress to the old 19-point 0.3811138496
  profile or generic coordinate sweeps. If ultra-small active-lag transfers plateau,
  switch families: run deterministic finite-difference Adam/smooth-max refinement
  around the 63-point profile, perturbing the strong profile at scales
  [1e-2, 3e-3, 1e-3, 3e-4] before projected active-lag polishing.
- Explicit step-progress requirement: the next guidance must name the previous raw C5
  and a lower target for the next step. Prefer a modest strict improvement over a
  risky reset; the desired training trace is a descending chain, not one copied
  incumbent repeated across steps.
- The New Guidance Delta must be a real algorithmic change relative to the inherited
  summary. It cannot be "reuse/copy the template", "run the same SLSQP again", "no
  further search", or only "keep the same parameters".
- If the previous summary says there was no distinct new idea, or if the empirical
  outcome did not beat the current best valid raw score, change at least two concrete
  search knobs: perturbation scale schedule, coordinate/pair order, number of trials,
  restart seeds, top-K candidates passed to SLSQP, optimizer initialization, or
  n_points choice.
- Make the delta executable: state initialization/source, coordinate or pair order,
  signed delta schedule, passes/trials, top-K refinements, and fallback next delta.
- If the selected/global guidance already used the same coordinate sweep family, do
  not repeat it unchanged. Choose and name a distinct deterministic family such as
  pairwise mass-transfer sweeps, reverse-order half sweeps, two-coordinate
  antisymmetric perturbations, deterministic seeded direction probes, or a controlled
  n_points extension only after exploiting the current 19-point basin.
- Prefer a deterministic non-copy search schedule such as paired/coordinate sweeps over
  deltas [5e-4, 2e-4, 1e-4, 5e-5, 1e-5], projecting after every candidate, evaluating
  c5_score immediately, keeping the best candidate, then running SLSQP only from the
  top near-tie/improved candidates.
- Include known information directly in the guidance when useful: current best raw
  score, best valid profile facts, verifier constraints, verifier normalization, and
  the strict improvement target.
- Search only controlled deterministic perturbations around it while preserving:
  0 <= h[i] <= 1, sum(h) = n_points / 2, the same verifier normalization,
  and mirror/complement symmetry if present in the current best profile.
- Optimize only the independent half of the variables.
- Use a small local search such as coordinate perturbation, projected line search,
  or SLSQP if available.
- For each candidate, immediately evaluate using the official verifier and keep the
  lowest raw C5.
- Preserve n_points from the inherited best profile by default. For the known-good
  root/default profile this means n_points=63. Only propose alternate n_points values
  as a secondary controlled experiment after the inherited profile has been exploited,
  and state the interpolation/initialization rule for the alternate size.
- Target: strictly improve over {best_valid_target}, not merely beat 0.5.
- Aspirational target: search for a path toward raw C5 <= {aspiration_target}; if this
  is not reachable in one rollout, the guidance should still specify the smallest
  concrete local improvement to try next.

Thinking is allowed, but the submitted guidance must be inside the XML block below.
If the XML block is missing, only text outside any <think>...</think> block will be
used as the submitted guidance.

Return exactly one block and nothing else. The block must use these headings and must
contain at least five concrete inherited details when the summaries provide them:
<guidance>
Previous Summary Facts to Preserve:
- ...

Hypothesis:
...

Inherited Implementation Details:
- ...

New Guidance Delta:
- ...

Expected Step Improvement:
- Parent/selected raw C5: ...
- Current best valid raw C5: ...
- Why this delta should lower the next step: ...
- Plateau escape if equal to current best: include active lag, top contributing pairs,
  tiny signed deltas, and top-K refinement.

Concrete Search Recipe:
- Initialization: include exact n_points and h=[...] values if visible; otherwise name
  the inherited seed/profile source. If only head/tail are visible, do not reconstruct
  the missing middle values; initialize from the previous verified solution/profile.
- Sweep/order: ...
- Signed deltas: ...
- Trials/passes/top-K/SLSQP: ...
- Fallback next delta: ...

No-op Guard:
- Returning the inherited profile unchanged, or summarizing that the safest approach
  is to keep the current best, is not acceptable. Specify the non-copy search that
  must run before any fallback.

Execution Instructions:
1. ...
2. ...
3. ...

Acceptance / Fallback:
- Beat raw C5 {best_valid_target}; aspire toward <= {aspiration_target}.
- If no improvement appears, report the bottleneck and the next smaller delta rather than returning a copied baseline.
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
    if best_valid is None and (target_raw_score is None or float(target_raw_score) > KNOWN_GOOD_ERDOS_RAW_C5):
        target_raw_score = KNOWN_GOOD_ERDOS_RAW_C5
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
If the known template or previous best already scores approximately {target_raw_score},
copying it or rerunning the exact same SLSQP setup is not an improvement. In that case,
perform a real deterministic search over new projected perturbations before returning.
Do not answer that the safest approach is to return the existing best profile; that
creates a zero-delta training step. Implement the search, evaluate candidates, and
let the verifier decide the score.
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
- Use at least one concrete non-copy search change from the guidance, such as a
  multi-scale coordinate/pair sweep, changed restart seeds, or SLSQP initialized from
  top perturbed candidates instead of the unchanged profile.
- If the guidance or previous summary says the last step plateaued, implement at least
  two candidate families before returning: a smaller multi-scale coordinate/pair sweep
  and an active-lag/top-contributor mass-transfer sweep. Track the best non-identical
  feasible candidate separately from the inherited profile, and mention rejected
  deltas in the summary.
- If guidance mentions active-lag search, implement it concretely: compute the
  full-correlation vector, identify the max-score lag and top contributing index
  pairs, test tiny signed projected mass transfers on those pairs, and keep only
  candidates with strictly lower c5_bound before optional SLSQP refinement.
- If the inherited profile already scores at or below 0.38103, preserve that exact
  63-point profile as the incumbent but do not restrict search to copy-only
  refinements. First try active-lag transfers; if they do not strictly improve, run
  deterministic finite-difference Adam/smooth-max escape around the incumbent with
  perturbation scales [1e-2, 3e-3, 1e-3, 3e-4], then polish the best candidate.
- Preserve n_points when reusing a previous valid profile. For the known-good template
  this means keep n_points=63 unless the guidance explicitly gives an alternate-size
  interpolation experiment.
- The known-good 63-point lineage already showed step-to-step progress:
  {KNOWN_GOOD_ERDOS_PROGRESS_LADDER}. The returned summary must say whether the new
  candidate improved over the inherited raw C5 and which perturbation family caused
  the best non-copy candidate.
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

Return exactly these three blocks in this order:
1. <execution_thinking>...</execution_thinking>
2. One fenced Python code block containing def run(seed=42, budget_s=1, **kwargs):
3. <summary>...</summary>

Keep reasoning and summary short. Do not claim verifier success because the verifier has not run yet.

<execution_thinking>
brief reasoning
</execution_thinking>

```python
def run(seed=42, budget_s=1, **kwargs):
    # final runnable solution
```

<summary>
Execution Interpretation
...

Implemented Algorithm
...

New Ideas Introduced
...

Empirical Outcome
Pending verifier execution.

Failure / Bottleneck Analysis
...

Next Guidance Delta
If no strict improvement was found, name the exact perturbation families and delta
scales that failed, then propose at least two changed knobs for the next step. Do not
recommend copying the same profile unchanged.
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
