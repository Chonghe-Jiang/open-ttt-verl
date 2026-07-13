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
    PROMPT_MODE_CODE_DELTA,
    PROMPT_MODE_SUMMARY_ONLY,
    Prompt,
    extract_tag_or_none,
    normalize_prompt_mode,
)
from guidance_ttt.state import LLMRequest, LibraryEntry, _jsonable
from guidance_ttt.tasks import TaskSpec, get_task_spec


BOOTSTRAP_GUIDANCE = "Bootstrap execution without guidance."


def build_bootstrap_execution_prompt(*, task_spec: TaskSpec) -> Prompt:
    fenced_language = "cpp" if task_spec.solution_language.lower() in {"cpp", "c++", "cxx"} else "python"
    language_name = "C++17" if fenced_language == "cpp" else "Python"
    placeholder = (
        "A complete self-contained C++17 program."
        if fenced_language == "cpp"
        else "A complete self-contained Python program."
    )
    user = f"""<problem>
{task_spec.problem_prompt}
</problem>

You are generating the initial bootstrap candidate for a Guidance-TTT run.

There is no previous library summary and no guidance yet. Produce one concrete baseline solution that satisfies the problem statement and can seed the future search history.

{task_spec.execution_solution_contract}

Your response must contain exactly three top-level XML blocks and no extra text before, between, or after them.
You must output all three XML blocks exactly as shown below.
The <execution_thinking>...</execution_thinking> block is mandatory and must use angle brackets.
The <solution> block is mandatory and must contain a fenced ```{fenced_language} code block.
The <summary>...</summary> block is mandatory and must be closed.
Do not output only execution_thinking, only a summary, or plain natural language.
Do not omit angle brackets from XML tags.
The final characters of your response must be </summary>.

Required output format:

<execution_thinking>
A short explanation of the baseline strategy used to produce the initial candidate.
Do not include code.
</execution_thinking>

<solution>
```{fenced_language}
{placeholder}
```
</solution>

<summary>
A concise natural-language summary of the candidate.

This summary must describe the implemented baseline algorithm, the main construction/search/refinement mechanism, and what future guidance could improve. If the solution intentionally uses a simple heuristic rather than a full optimization method, state that clearly.

Do not include source code, code fences, copied constants, hard-coded arrays, raw candidate parameters, benchmark-specific profile values, or the output-format instructions themselves.
</summary>

Any response that does not follow this exact three-block structure should be treated as invalid.
"""
    return Prompt(
        system=(
            f"You are the bootstrap execution model. Produce one concrete runnable {language_name} "
            "baseline candidate and a concise summary for seeding a Guidance-TTT library."
        ),
        user=user,
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
    prompt = build_bootstrap_execution_prompt(task_spec=task_spec)
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
) -> LibraryEntry:
    last_error = "bootstrap did not run"
    attempts = max(1, int(max_attempts))
    for attempt in range(1, attempts + 1):
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
            _validate_required_blocks(execution_text)
            execution_result = _verify_execution_without_fallback(
                execution_text=execution_text,
                guidance=BOOTSTRAP_GUIDANCE,
                timeout_s=verifier_timeout_s,
                task_spec=task_spec,
                verifier_config=_verifier_config_from_task_config(task_config),
                prompt_mode=prompt_mode,
            )
            raw_model_summary = execution_result.model_summary
            metadata = {
                "bootstrap": True,
                "bootstrap_attempts": attempt,
                "execution_prompt": {"system": prompt.system, "user": prompt.user},
                "task": task_config,
                "execution_text": execution_result.execution_text,
                "raw_model_summary": raw_model_summary,
                "prompt_mode": prompt_mode,
                "summary_semantics": "baseline",
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
    raise RuntimeError(f"bootstrap execution did not produce strict XML after {attempts} attempt(s): {last_error}")


def _validate_required_blocks(execution_text: str) -> None:
    missing = [
        tag
        for tag in ("execution_thinking", "solution", "summary")
        if extract_tag_or_none(execution_text, tag) is None
    ]
    if missing:
        raise RuntimeError(f"Bootstrap execution missing required XML block(s): {', '.join(missing)}")
