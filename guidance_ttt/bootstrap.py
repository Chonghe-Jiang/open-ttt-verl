# ruff: noqa: E501

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from guidance_ttt.agent_loop import (
    _execution_max_tokens,
    _extract_reusable_idea,
    _verifier_config_from_task_config,
    _verify_execution_without_fallback,
)
from guidance_ttt.library import GuidanceLibrary
from guidance_ttt.llm_client import make_llm_client
from guidance_ttt.prompts import (
    EXECUTION_PROMPT_STYLE_GPT_API_BRIEF_THINKING,
    EXECUTION_PROMPT_STYLE_QWEN_NATIVE,
    EXECUTION_PROMPT_STYLE_QWEN_NO_THINKING,
    PROMPT_MODE_CODE_DELTA,
    PROMPT_MODE_SUMMARY_ONLY,
    Prompt,
    extract_tag_or_none,
    normalize_execution_prompt_style,
    normalize_prompt_mode,
)
from guidance_ttt.state import LibraryEntry, LLMRequest, VerificationResult, _jsonable
from guidance_ttt.tasks import TaskSpec, get_task_spec

BOOTSTRAP_GUIDANCE = "Bootstrap execution without guidance."
BOOTSTRAP_SOURCE_TASK_BASELINE = "task_baseline"
BOOTSTRAP_SOURCE_SCRATCH = "scratch"
SUPPORTED_BOOTSTRAP_SOURCES = frozenset(
    {BOOTSTRAP_SOURCE_TASK_BASELINE, BOOTSTRAP_SOURCE_SCRATCH}
)


def normalize_bootstrap_source(source: str | None) -> str:
    normalized = str(source or BOOTSTRAP_SOURCE_TASK_BASELINE).strip().lower()
    if normalized not in SUPPORTED_BOOTSTRAP_SOURCES:
        supported = ", ".join(sorted(SUPPORTED_BOOTSTRAP_SOURCES))
        raise ValueError(f"Unsupported bootstrap source {source!r}; expected one of: {supported}")
    return normalized


def build_bootstrap_execution_prompt(
    *,
    task_spec: TaskSpec,
    execution_prompt_style: str | None = None,
    baseline_verification: VerificationResult | None = None,
    prompt_mode: str = PROMPT_MODE_SUMMARY_ONLY,
    bootstrap_source: str = BOOTSTRAP_SOURCE_TASK_BASELINE,
) -> Prompt:
    execution_prompt_style = normalize_execution_prompt_style(execution_prompt_style)
    prompt_mode = normalize_prompt_mode(prompt_mode)
    bootstrap_source = normalize_bootstrap_source(bootstrap_source)
    use_task_baseline = bool(
        bootstrap_source == BOOTSTRAP_SOURCE_TASK_BASELINE and task_spec.bootstrap_solution
    )
    fenced_language = "cpp" if task_spec.solution_language.lower() in {"cpp", "c++", "cxx"} else "python"
    language_name = "C++17" if fenced_language == "cpp" else "Python"
    placeholder = (
        "A complete self-contained C++17 program."
        if fenced_language == "cpp"
        else "A complete self-contained Python program."
    )
    if execution_prompt_style in {
        EXECUTION_PROMPT_STYLE_GPT_API_BRIEF_THINKING,
        EXECUTION_PROMPT_STYLE_QWEN_NATIVE,
        EXECUTION_PROMPT_STYLE_QWEN_NO_THINKING,
    }:
        if execution_prompt_style == EXECUTION_PROMPT_STYLE_QWEN_NATIVE:
            prompt_style_preamble = (
                "Qwen native thinking is enabled by the chat template. Use that native reasoning channel; "
                "do not manually emit <think> or <execution_thinking> in the final answer."
            )
        elif execution_prompt_style == EXECUTION_PROMPT_STYLE_QWEN_NO_THINKING:
            prompt_style_preamble = (
                "Thinking mode is disabled. Do not output <think> or <execution_thinking>; respond directly "
                "with the required solution and summary."
            )
        else:
            prompt_style_preamble = (
                "Think carefully about whether a conservative, correctness-preserving improvement to the supplied "
                "baseline is safe before producing the seed program."
                if use_task_baseline
                else "Think carefully about the public problem specification before producing the scratch seed."
            )
        output_contract = f"""{prompt_style_preamble}

Your final answer must contain exactly two top-level XML blocks and no extra final-answer text before, between, or after them.
The <solution> block is mandatory and must contain a fenced ```{fenced_language} code block.
The <summary>...</summary> block is mandatory and must be closed.
The final characters of your final answer must be </summary>.

Required final-answer format:

<solution>
```{fenced_language}
{placeholder}
```
</solution>

<summary>
A concise natural-language summary of the candidate.

{_bootstrap_summary_contract(task_spec, prompt_mode=prompt_mode, use_task_baseline=use_task_baseline)}

Do not include source code, code fences, copied constants, hard-coded arrays, raw candidate parameters, benchmark-specific profile values, or the output-format instructions themselves.
</summary>

Any final answer that does not follow this exact two-block structure should be treated as invalid."""
    else:
        output_contract = f"""Your response must contain exactly three top-level XML blocks and no extra text before, between, or after them.
You must output all three XML blocks exactly as shown below.
The <execution_thinking>...</execution_thinking> block is mandatory and must use angle brackets.
The <solution> block is mandatory and must contain a fenced ```{fenced_language} code block.
The <summary>...</summary> block is mandatory and must be closed.
Do not output only execution_thinking, only a summary, or plain natural language.
Do not omit angle brackets from XML tags.
The final characters of your response must be </summary>.

Required output format:

<execution_thinking>
A short explanation of the seed strategy used to produce the initial candidate.
Do not include code.
</execution_thinking>

<solution>
```{fenced_language}
{placeholder}
```
</solution>

<summary>
A concise natural-language summary of the candidate.

{_bootstrap_summary_contract(task_spec, prompt_mode=prompt_mode, use_task_baseline=use_task_baseline)}

Do not include source code, code fences, copied constants, hard-coded arrays, raw candidate parameters, benchmark-specific profile values, or the output-format instructions themselves.
</summary>

Any response that does not follow this exact three-block structure should be treated as invalid."""

    if use_task_baseline:
        raw_score = (
            "None"
            if baseline_verification is None or baseline_verification.raw_score is None
            else repr(float(baseline_verification.raw_score))
        )
        reward = 0.0 if baseline_verification is None else float(baseline_verification.reward)
        status = "not_evaluated" if baseline_verification is None else baseline_verification.status
        message = "Not evaluated yet." if baseline_verification is None else baseline_verification.message
        baseline_context = f"""<baseline_candidate>
<baseline_summary>
{task_spec.bootstrap_summary or "No baseline summary is available."}
</baseline_summary>

<baseline_score>
Verifier status: {status}
{task_spec.raw_score_label}: {raw_score}
Reward: {reward!r}
Verifier message: {message}
</baseline_score>

<baseline_code>
```{fenced_language}
{task_spec.bootstrap_solution.strip()}
```
</baseline_code>
</baseline_candidate>

Create the safest valid seed from the supplied baseline while preserving its required interface and correctness. Attempt a conservative improvement only when you can implement it confidently. If no such improvement is safe, reproduce the verified baseline exactly rather than risking an invalid seed, and state explicitly in <summary> that no code change was made. Return one complete file, not a patch or diff."""
        bootstrap_instruction = (
            "You are generating a safe initial seed candidate for a Guidance-TTT run from the official verified "
            "baseline. Later guidance/execution rollouts will perform the main evolutionary search."
        )
    else:
        baseline_context = ""
        task_scratch_constraint = (
            f"\n\nTask-specific scratch-seed constraints:\n{task_spec.scratch_bootstrap_constraint}"
            if task_spec.scratch_bootstrap_constraint
            else ""
        )
        bootstrap_instruction = (
            "You are generating a scratch seed candidate for a Guidance-TTT run.\n\n"
            "There is no parent candidate, prior solution, library summary, guidance, verifier score, or search "
            "history. Design one concrete solution solely from the public problem statement and rules above. "
            "Do not assume, reconstruct, or refer to any unpublished or previously optimized implementation. "
            "The returned program must be complete and independently verifier-valid so it can seed future search."
            f"{task_scratch_constraint}"
        )

    user = f"""<problem>
{task_spec.problem_prompt}
</problem>

{bootstrap_instruction}

{baseline_context}

{task_spec.execution_solution_contract}

{output_contract}
"""
    return Prompt(
        system=(
            f"You are the bootstrap execution model. Produce one concrete runnable {language_name} "
            + (
                (
                    "seed based on the supplied verified baseline and a precise delta summary for seeding a "
                    "Guidance-TTT library. Prioritize verifier-valid code over bootstrap novelty."
                    if prompt_mode == PROMPT_MODE_CODE_DELTA
                    else "seed based on the supplied verified baseline and a self-contained algorithm summary for "
                    "seeding a Guidance-TTT library. Prioritize verifier-valid code over bootstrap novelty."
                )
                if use_task_baseline
                else "scratch seed and a self-contained summary for seeding a Guidance-TTT library."
            )
        ),
        user=user,
    )


def _bootstrap_summary_contract(
    task_spec: TaskSpec,
    *,
    prompt_mode: str,
    use_task_baseline: bool | None = None,
) -> str:
    if use_task_baseline is None:
        use_task_baseline = bool(task_spec.bootstrap_solution)
    if use_task_baseline and prompt_mode == PROMPT_MODE_CODE_DELTA:
        return (
            "Describe only the concrete algorithmic changes from the supplied baseline, how they reduce the "
            "identified bottleneck, and which planned mechanisms were simplified or omitted. If the baseline is "
            "preserved unchanged for safety, say so explicitly. Do not claim a verifier outcome because "
            "verification happens after generation."
        )
    if use_task_baseline:
        return (
            "Describe the complete algorithm implemented by the submitted candidate, including the important "
            "mechanisms preserved from the supplied baseline and any concrete changes made to it. The summary must "
            "stand alone because the guidance model will not receive source code in summary_only mode. If the "
            "baseline is preserved unchanged for safety, say so explicitly and still summarize its complete "
            "algorithm. Do not claim a verifier outcome because verification happens after generation."
        )
    return (
        "Describe the complete algorithm implemented by this scratch candidate, including its main construction, "
        "search, refinement, or optimization mechanisms and the important correctness constraints it enforces. "
        "The summary must stand alone because the guidance model will not receive source code in summary_only mode. "
        "Describe only mechanisms actually present in the submitted solution, and do not claim a verifier outcome "
        "because verification happens after generation."
    )


def bootstrap_library_entries(
    library_path: str | Path,
    *,
    task_config: dict[str, Any],
    execution_llm_config: dict[str, Any],
    verifier_timeout_s: int,
    max_attempts: int = 2,
    overwrite_existing: bool = False,
    execution_client: Any | None = None,
    prompt_mode: str = PROMPT_MODE_SUMMARY_ONLY,
    bootstrap_source: str = BOOTSTRAP_SOURCE_TASK_BASELINE,
) -> dict[str, Any]:
    return asyncio.run(
        _bootstrap_library_entries_async(
            library_path,
            task_config=task_config,
            execution_llm_config=execution_llm_config,
            verifier_timeout_s=verifier_timeout_s,
            max_attempts=max_attempts,
            overwrite_existing=overwrite_existing,
            execution_client=execution_client,
            prompt_mode=prompt_mode,
            bootstrap_source=bootstrap_source,
        )
    )


async def _bootstrap_library_entries_async(
    library_path: str | Path,
    *,
    task_config: dict[str, Any],
    execution_llm_config: dict[str, Any],
    verifier_timeout_s: int,
    max_attempts: int,
    overwrite_existing: bool,
    execution_client: Any | None,
    prompt_mode: str,
    bootstrap_source: str,
) -> dict[str, Any]:
    task_config = _jsonable(task_config)
    if not isinstance(task_config, dict):
        raise TypeError(f"task_config must normalize to a dict, got {type(task_config).__name__}")
    execution_llm_config = _jsonable(execution_llm_config)
    if not isinstance(execution_llm_config, dict):
        raise TypeError(
            f"execution_llm_config must normalize to a dict, got {type(execution_llm_config).__name__}"
        )
    task_spec = get_task_spec(str(task_config.get("id", "erdos_min_overlap")))
    prompt_mode = normalize_prompt_mode(prompt_mode)
    bootstrap_source = normalize_bootstrap_source(bootstrap_source)
    use_task_baseline = bool(
        bootstrap_source == BOOTSTRAP_SOURCE_TASK_BASELINE and task_spec.bootstrap_solution
    )
    execution_prompt_style = normalize_execution_prompt_style(execution_llm_config.get("prompt_style"))
    baseline_verification = (
        _verify_bootstrap_baseline(
            task_spec=task_spec,
            task_config=task_config,
            verifier_timeout_s=verifier_timeout_s,
        )
        if use_task_baseline
        else None
    )
    prompt = build_bootstrap_execution_prompt(
        task_spec=task_spec,
        execution_prompt_style=execution_prompt_style,
        baseline_verification=baseline_verification,
        prompt_mode=prompt_mode,
        bootstrap_source=bootstrap_source,
    )
    client = execution_client or make_llm_client(execution_llm_config)
    library = GuidanceLibrary(library_path)
    snapshot = library.snapshot()
    root_nodes = [node for node in snapshot["nodes"].values() if node.get("parent_id") is None]
    if not root_nodes:
        raise RuntimeError(f"No root node found for task {task_spec.task_id!r} in {library_path}")

    created_entries: list[str] = []
    skipped_roots: list[str] = []
    failed_roots: dict[str, str] = {}
    for root in root_nodes:
        root_id = str(root["id"])
        if root.get("entry_id") and not overwrite_existing:
            skipped_roots.append(root_id)
            continue
        try:
            entry = await _create_bootstrap_entry(
                root_id=root_id,
                task_spec=task_spec,
                problem_id=str(root.get("problem_id") or task_spec.task_id),
                task_config=task_config,
                execution_llm_config=execution_llm_config,
                execution_client=client,
                prompt=prompt,
                verifier_timeout_s=verifier_timeout_s,
                max_attempts=max_attempts,
                prompt_mode=prompt_mode,
                execution_prompt_style=execution_prompt_style,
                baseline_verification=baseline_verification,
                bootstrap_source=(
                    BOOTSTRAP_SOURCE_TASK_BASELINE
                    if use_task_baseline
                    else BOOTSTRAP_SOURCE_SCRATCH
                ),
                attempt_log_path=Path(library_path).with_name("bootstrap_attempts.json"),
            )
        except Exception as exc:
            failed_roots[root_id] = str(exc)
            continue
        library.attach_entry_to_root(root_id, entry, overwrite_existing=overwrite_existing)
        created_entries.append(entry.id)

    if failed_roots:
        raise RuntimeError(f"Bootstrap failed for root nodes: {json.dumps(failed_roots, sort_keys=True)}")
    return {
        "library_path": str(library_path),
        "root_count": len(root_nodes),
        "created_count": len(created_entries),
        "skipped_count": len(skipped_roots),
        "created_entry_ids": created_entries,
        "skipped_root_ids": skipped_roots,
        "bootstrap_source": (
            BOOTSTRAP_SOURCE_TASK_BASELINE if use_task_baseline else BOOTSTRAP_SOURCE_SCRATCH
        ),
    }


async def _create_bootstrap_entry(
    *,
    root_id: str,
    task_spec: TaskSpec,
    problem_id: str,
    task_config: dict[str, Any],
    execution_llm_config: dict[str, Any],
    execution_client: Any,
    prompt: Prompt,
    verifier_timeout_s: int,
    max_attempts: int,
    prompt_mode: str,
    execution_prompt_style: str,
    baseline_verification: VerificationResult | None,
    bootstrap_source: str,
    attempt_log_path: Path,
) -> LibraryEntry:
    last_error = "bootstrap did not run"
    attempts = max(1, int(max_attempts))
    attempt_records: list[dict[str, Any]] = []
    for attempt in range(1, attempts + 1):
        attempt_record: dict[str, Any] = {"attempt": attempt, "accepted": False}
        try:
            response = await execution_client.complete(
                LLMRequest(
                    system=prompt.system,
                    user=prompt.user,
                    model=execution_llm_config.get("model", "mock-exec"),
                    temperature=float(execution_llm_config.get("temperature", 0.2)),
                    max_tokens=_execution_max_tokens(execution_llm_config),
                    metadata={"purpose": "execution", "bootstrap": True},
                )
            )
            execution_text = response.text
            attempt_record.update(
                {
                    "execution_text": execution_text,
                    "execution_reasoning": response.reasoning,
                    "execution_finish_reason": response.finish_reason,
                    "execution_response_metadata": response.metadata,
                    "execution_response_usage": response.usage,
                }
            )
            _validate_required_blocks(execution_text, execution_prompt_style=execution_prompt_style)
            execution_result = _verify_execution_without_fallback(
                execution_text=execution_text,
                guidance=BOOTSTRAP_GUIDANCE,
                timeout_s=verifier_timeout_s,
                task_spec=task_spec,
                verifier_config=_verifier_config_from_task_config(task_config),
                prompt_mode=prompt_mode,
                execution_reasoning=response.reasoning,
            )
            attempt_record["verification"] = {
                "reward": execution_result.verification.reward,
                "raw_score": execution_result.verification.raw_score,
                "status": execution_result.verification.status,
                "valid": execution_result.verification.valid,
                "message": execution_result.verification.message,
                "artifacts": execution_result.verification.artifacts,
            }
            if not execution_result.verification.valid:
                raise RuntimeError(
                    "bootstrap candidate failed verification: "
                    f"{execution_result.verification.status}: {execution_result.verification.message}"
                )
            raw_model_summary = execution_result.model_summary
            metadata = {
                "bootstrap": True,
                "bootstrap_source": bootstrap_source,
                "bootstrap_attempts": attempt,
                "execution_prompt": {"system": prompt.system, "user": prompt.user},
                "task": task_config,
                "execution_text": execution_result.execution_text,
                "raw_model_summary": raw_model_summary,
                "prompt_mode": prompt_mode,
                "summary_semantics": (
                    "delta_from_bootstrap_baseline"
                    if bootstrap_source == BOOTSTRAP_SOURCE_TASK_BASELINE
                    and prompt_mode == PROMPT_MODE_CODE_DELTA
                    else "canonical_full_candidate"
                    if bootstrap_source == BOOTSTRAP_SOURCE_SCRATCH
                    or prompt_mode == PROMPT_MODE_SUMMARY_ONLY
                    else "baseline"
                ),
                "execution_provider": execution_llm_config.get("provider", "mock"),
                "execution_model": execution_llm_config.get("model", "mock-exec"),
                "execution_response_metadata": response.metadata,
                "execution_response_usage": response.usage,
                "execution_finish_reason": response.finish_reason,
                "verification_artifacts": execution_result.verification.artifacts,
                "execution_fallback_used": execution_result.fallback_used,
                "execution_fallback_reason": execution_result.fallback_reason,
                "original_execution_text": execution_result.original_execution_text,
            }
            if baseline_verification is not None:
                metadata["bootstrap_baseline_verification"] = {
                    "reward": baseline_verification.reward,
                    "raw_score": baseline_verification.raw_score,
                    "status": baseline_verification.status,
                    "message": baseline_verification.message,
                    "artifacts": baseline_verification.artifacts,
                }
            attempt_record["accepted"] = True
            attempt_records.append(attempt_record)
            _write_bootstrap_attempt_log(
                attempt_log_path,
                prompt=prompt,
                bootstrap_source=bootstrap_source,
                attempts=attempt_records,
            )
            return LibraryEntry(
                id=str(uuid4()),
                parent_id=root_id,
                problem_id=problem_id,
                timestep=0,
                guidance=BOOTSTRAP_GUIDANCE,
                execution_thinking=execution_result.execution_thinking,
                solution=execution_result.solution,
                verifier_reward=execution_result.verification.reward,
                verifier_raw_score=execution_result.verification.raw_score,
                verifier_status=execution_result.verification.status,
                verifier_message=execution_result.verification.message,
                summary=execution_result.summary,
                reusable_idea=(
                    execution_result.summary
                    if prompt_mode == PROMPT_MODE_CODE_DELTA
                    else _extract_reusable_idea(execution_result.summary)
                ),
                failure_mode=None if execution_result.verification.valid else execution_result.verification.status,
                metadata=metadata,
            )
        except Exception as exc:
            last_error = str(exc)
            attempt_record["error"] = last_error
            attempt_records.append(attempt_record)
            _write_bootstrap_attempt_log(
                attempt_log_path,
                prompt=prompt,
                bootstrap_source=bootstrap_source,
                attempts=attempt_records,
            )
    raise RuntimeError(
        f"bootstrap execution did not produce a valid strict candidate after {attempts} attempt(s): {last_error}"
    )


def _write_bootstrap_attempt_log(
    path: Path,
    *,
    prompt: Prompt,
    bootstrap_source: str,
    attempts: list[dict[str, Any]],
) -> None:
    payload = {
        "bootstrap_source": bootstrap_source,
        "prompt": {"system": prompt.system, "user": prompt.user},
        "attempts": attempts,
    }
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True))


def _verify_bootstrap_baseline(
    *,
    task_spec: TaskSpec,
    task_config: dict[str, Any],
    verifier_timeout_s: int,
) -> VerificationResult | None:
    if not task_spec.bootstrap_solution:
        return None
    fenced_language = "cpp" if task_spec.solution_language.lower() in {"cpp", "c++", "cxx"} else "python"
    baseline_text = (
        f"<solution>\n```{fenced_language}\n{task_spec.bootstrap_solution.strip()}\n```\n</solution>"
    )
    verification = task_spec.verify_execution_text(
        baseline_text,
        timeout_s=verifier_timeout_s,
        config=_verifier_config_from_task_config(task_config),
    )
    if not verification.valid:
        raise RuntimeError(
            "Configured bootstrap baseline failed verification: "
            f"{verification.status}: {verification.message}"
        )
    return verification


def _validate_required_blocks(execution_text: str, *, execution_prompt_style: str | None = None) -> None:
    execution_prompt_style = normalize_execution_prompt_style(execution_prompt_style)
    required_tags = ["solution", "summary"]
    if execution_prompt_style not in {
        EXECUTION_PROMPT_STYLE_GPT_API_BRIEF_THINKING,
        EXECUTION_PROMPT_STYLE_QWEN_NATIVE,
        EXECUTION_PROMPT_STYLE_QWEN_NO_THINKING,
    }:
        required_tags.insert(0, "execution_thinking")
    missing = [tag for tag in required_tags if extract_tag_or_none(execution_text, tag) is None]
    if missing:
        raise RuntimeError(f"Bootstrap execution missing required XML block(s): {', '.join(missing)}")
