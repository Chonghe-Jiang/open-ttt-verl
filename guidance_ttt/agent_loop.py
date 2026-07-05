from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from guidance_ttt.library import GuidanceLibrary
from guidance_ttt.llm_client import make_llm_client
from guidance_ttt.prompts import (
    build_execution_prompt,
    build_guidance_prompt,
    extract_guidance_or_format_error,
    extract_tag_or_none,
)
from guidance_ttt.state import LLMRequest, LibraryEntry, VerificationResult, _jsonable
from guidance_ttt.tasks import TaskSpec, get_task_spec


@dataclass
class GuidanceGeneration:
    text: str
    response_ids: list[int]
    response_logprobs: list[float] | None
    attempts: int
    raw_text: str
    stop_reason: str | None


@dataclass
class ExecutionVerification:
    execution_text: str
    execution_thinking: str
    solution: str
    summary: str
    model_summary: str | None
    verification: VerificationResult
    fallback_used: bool
    fallback_reason: str | None
    original_execution_text: str


try:
    from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput, register
except ModuleNotFoundError:
    AgentLoopBase = object

    def register(name: str):
        def decorator(cls):
            return cls

        return decorator

    @dataclass
    class AgentLoopMetrics:
        generate_sequences: float = 0.0
        tool_calls: float = 0.0
        num_preempted: int = -1

    @dataclass
    class AgentLoopOutput:
        prompt_ids: list[int]
        response_ids: list[int]
        response_mask: list[int]
        response_logprobs: list[float] | None = None
        multi_modal_data: dict[str, Any] = field(default_factory=dict)
        reward_score: float | None = None
        num_turns: int = 0
        metrics: AgentLoopMetrics = field(default_factory=AgentLoopMetrics)
        extra_fields: dict[str, Any] = field(default_factory=dict)


def build_agent_loop_output(
    *,
    prompt_ids: list[int],
    response_ids: list[int],
    response_logprobs: list[float] | None,
    reward: float,
    extra_fields: dict[str, Any],
) -> AgentLoopOutput:
    return AgentLoopOutput(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        response_mask=[1] * len(response_ids),
        response_logprobs=response_logprobs,
        multi_modal_data={},
        reward_score=float(reward),
        num_turns=3,
        metrics=AgentLoopMetrics(),
        extra_fields=extra_fields,
    )


def _decode_response(tokenizer: Any, response_ids: list[int]) -> str:
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(response_ids, skip_special_tokens=True)
    return "".join(str(token_id) for token_id in response_ids)


def _decode_response_with_specials(tokenizer: Any, response_ids: list[int]) -> str:
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(response_ids, skip_special_tokens=False)
    return "".join(str(token_id) for token_id in response_ids)


@register("guidance_execution_erdos")
@register("guidance_execution_task")
class GuidanceExecutionAgentLoop(AgentLoopBase):
    """verl agent loop: train guidance tokens, execute/summarize with external LLMs."""

    def __init__(
        self,
        *args,
        execution_llm: dict[str, Any] | None = None,
        verifier_timeout_s: int = 60,
        problem_prompt: str | None = None,
        task: dict[str, Any] | str | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        fallback_task = task if task is not None else self._task_config_from_rollout_config()
        self.task_config = _normalize_task_config(fallback_task)
        self.task_spec = get_task_spec(str(self.task_config.get("id", "erdos_min_overlap")))
        self.execution_llm_config = execution_llm or self._execution_llm_from_rollout_config() or {
            "provider": "mock",
            "model": "mock-exec",
        }
        self.execution_client = make_llm_client(self.execution_llm_config)
        self.verifier_timeout_s = int(verifier_timeout_s)
        self.problem_prompt_override = problem_prompt
        self.problem_prompt = problem_prompt or self.task_spec.problem_prompt
        if hasattr(self, "rollout_config"):
            self.response_length = self.rollout_config.response_length

    def _agent_loop_config_from_rollout_config(self) -> dict[str, Any] | None:
        rollout_config = getattr(self, "rollout_config", None)
        agent_config = _config_get(rollout_config, "agent")
        agent_loop_config_path = _config_get(agent_config, "agent_loop_config_path")
        if not agent_loop_config_path:
            return None
        default_agent_loop = _config_get(agent_config, "default_agent_loop") or "guidance_execution_task"
        candidate_names = list(dict.fromkeys([default_agent_loop, "guidance_execution_task", "guidance_execution_erdos"]))
        try:
            from omegaconf import OmegaConf

            loaded = OmegaConf.load(Path(str(agent_loop_config_path)).expanduser())
            for agent_loop_config in loaded:
                if _config_get(agent_loop_config, "name") not in candidate_names:
                    continue
                return _jsonable(OmegaConf.to_container(agent_loop_config, resolve=True))
        except Exception:
            return None
        return None

    def _execution_llm_from_rollout_config(self) -> dict[str, Any] | None:
        agent_loop_config = self._agent_loop_config_from_rollout_config()
        if agent_loop_config is None:
            return None
        execution_llm = _config_get(agent_loop_config, "execution_llm")
        if execution_llm:
            return _jsonable(execution_llm)
        return None

    def _task_config_from_rollout_config(self) -> dict[str, Any] | None:
        agent_loop_config = self._agent_loop_config_from_rollout_config()
        if agent_loop_config is None:
            return None
        task_config = _config_get(agent_loop_config, "task")
        if task_config:
            return _normalize_task_config(task_config)
        return None

    def _task_config_for_extra_info(self, extra_info: dict[str, Any], task_spec: TaskSpec) -> dict[str, Any]:
        extra_task_config = extra_info.get("task_config")
        if extra_task_config is not None:
            normalized = _normalize_task_config(extra_task_config)
            normalized["id"] = str(normalized.get("id") or task_spec.task_id)
            return normalized
        task_id = str(extra_info.get("task") or "")
        if task_id and task_id != str(self.task_config.get("id", "")):
            return {"id": task_spec.task_id}
        return self._task_config_for_spec(task_spec)

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> Any:
        if not hasattr(self, "server_manager"):
            raise RuntimeError("verl dependencies are required to run GuidanceExecutionAgentLoop")

        extra_info = dict(kwargs.get("extra_info") or {})
        task_spec = self._task_spec_for_extra_info(extra_info)
        task_config = self._task_config_for_extra_info(extra_info, task_spec)
        problem_prompt = self.problem_prompt_override or task_spec.problem_prompt
        library_path = extra_info.get("library_path") or extra_info.get("archive_path")
        if not library_path:
            raise ValueError("extra_info must include library_path")
        uid = str(kwargs.get("uid") or extra_info.get("uid") or extra_info.get("slot_id") or uuid4().hex)
        trajectory = dict(kwargs.get("trajectory") or {})
        global_step = kwargs.get("global_steps", kwargs.get("global_step", trajectory.get("step", 0)))
        group_uid = f"{global_step}:{uid}"

        library = GuidanceLibrary(
            library_path,
            rollout_n=int(extra_info.get("rollout_n", extra_info.get("group_size", 1))),
            puct_c=float(extra_info.get("puct_c", 1.0)),
            max_buffer_size=int(extra_info.get("max_buffer_size", 1000)),
            topk_children=int(extra_info.get("topk_children", 2)),
        )
        selected_node = library.acquire_group(group_uid, visible_timestep_exclusive=int(global_step))
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
            score_direction=task_spec.score_direction,
            raw_score_label=task_spec.raw_score_label,
        )
        verification: VerificationResult
        execution_text = ""
        execution_error: str | None = None
        try:
            execution_response = await self.execution_client.complete(
                LLMRequest(
                    system=execution_prompt.system,
                    user=execution_prompt.user,
                    model=self.execution_llm_config.get("model", "mock-exec"),
                    temperature=float(self.execution_llm_config.get("temperature", 0.2)),
                    max_tokens=_execution_max_tokens(self.execution_llm_config),
                    metadata={"purpose": "execution"},
                )
            )
            execution_text = execution_response.text
            execution_response_metadata = execution_response.metadata
            execution_response_usage = execution_response.usage
        except Exception as exc:
            execution_response_metadata = {}
            execution_response_usage = {}
            execution_error = str(exc)
        execution_result = _verify_execution_without_fallback(
            execution_text=execution_text,
            guidance=guidance,
            timeout_s=self.verifier_timeout_s,
            task_spec=task_spec,
            verifier_config=_verifier_config_from_task_config(task_config),
            initial_error=execution_error,
        )
        execution_text = execution_result.execution_text
        execution_thinking = execution_result.execution_thinking
        solution = execution_result.solution
        summary = execution_result.summary
        raw_model_summary = execution_result.model_summary
        verification = execution_result.verification

        entry = LibraryEntry(
            id=str(uuid4()),
            parent_id=selected_node.id,
            problem_id=selected_node.problem_id,
            timestep=int(global_step),
            guidance=guidance,
            execution_thinking=execution_thinking,
            solution=solution,
            verifier_reward=verification.reward,
            verifier_raw_score=verification.raw_score,
            verifier_status=verification.status,
            verifier_message=verification.message,
            summary=summary,
            reusable_idea=_extract_reusable_idea(summary),
            failure_mode=None if verification.valid else verification.status,
            metadata={
                "group_uid": group_uid,
                "selected_node_id": selected_node.id,
                "guidance_format_ok": guidance_format_ok,
                "guidance_generation_attempts": guidance_generation.attempts,
                "raw_guidance_with_specials": guidance_generation.raw_text,
                "guidance_stop_reason": guidance_generation.stop_reason,
                "raw_guidance_text": guidance_text,
                "guidance_prompt": {"system": guidance_prompt.system, "user": guidance_prompt.user},
                "execution_prompt": {"system": execution_prompt.system, "user": execution_prompt.user},
                "task": task_config,
                "execution_text": execution_text,
                "raw_model_summary": raw_model_summary,
                "execution_provider": self.execution_llm_config.get("provider", "mock"),
                "execution_model": self.execution_llm_config.get("model", "mock-exec"),
                "execution_response_metadata": execution_response_metadata,
                "execution_response_usage": execution_response_usage,
                "verification_artifacts": verification.artifacts,
                "execution_fallback_used": execution_result.fallback_used,
                "execution_fallback_reason": execution_result.fallback_reason,
                "original_execution_text": execution_result.original_execution_text,
            },
        )
        child = library.submit_child(group_uid, entry)

        return build_agent_loop_output(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_logprobs=guidance_generation.response_logprobs,
            reward=verification.reward,
            extra_fields={
                "group_uid": group_uid,
                "selected_node_id": selected_node.id,
                "child_node_id": child.id,
                "guidance": guidance,
                "guidance_format_ok": guidance_format_ok,
                "guidance_generation_attempts": guidance_generation.attempts,
                "raw_guidance_with_specials": guidance_generation.raw_text,
                "guidance_stop_reason": guidance_generation.stop_reason,
                "raw_guidance_text": guidance_text,
                "execution_text": execution_text,
                "execution_fallback_used": execution_result.fallback_used,
                "execution_fallback_reason": execution_result.fallback_reason,
                "original_execution_text": execution_result.original_execution_text,
                "execution_thinking": execution_thinking,
                "solution": solution,
                "verification": verification.to_dict(),
                "summary": summary,
                "library_entry_id": entry.id,
                "task": task_config,
            },
        )

    def _task_spec_for_extra_info(self, extra_info: dict[str, Any]) -> TaskSpec:
        task_id = str(extra_info.get("task") or self.task_config.get("id") or self.task_spec.task_id)
        if task_id == self.task_spec.task_id:
            return self.task_spec
        return get_task_spec(task_id)

    def _task_config_for_spec(self, task_spec: TaskSpec) -> dict[str, Any]:
        task_config = dict(self.task_config)
        task_config.setdefault("id", task_spec.task_id)
        return task_config

    async def _generate_guidance_response(self, prompt_ids: list[int], sampling_params: dict[str, Any]) -> GuidanceGeneration:
        first = await self.server_manager.generate(
            request_id=uuid4().hex,
            prompt_ids=prompt_ids,
            sampling_params=dict(sampling_params),
        )
        first_response_ids = first.token_ids[: self.response_length]
        first_text = _decode_response(self.tokenizer, first_response_ids)
        first_raw_text = _decode_response_with_specials(self.tokenizer, first_response_ids)
        if first_text.strip():
            return GuidanceGeneration(
                text=first_text,
                response_ids=first_response_ids,
                response_logprobs=first.log_probs[: len(first_response_ids)] if first.log_probs else None,
                attempts=1,
                raw_text=first_raw_text,
                stop_reason=getattr(first, "stop_reason", None),
            )

        retry_sampling_params = dict(sampling_params)
        retry_max_tokens = int(retry_sampling_params.get("max_tokens", retry_sampling_params.get("max_new_tokens", self.response_length)))
        retry_max_tokens = max(1, min(int(self.response_length), retry_max_tokens))
        retry_sampling_params["max_tokens"] = retry_max_tokens
        retry_sampling_params.pop("max_new_tokens", None)
        retry_sampling_params["min_tokens"] = min(16, retry_max_tokens)
        retry_sampling_params["ignore_eos"] = False
        retry = await self.server_manager.generate(
            request_id=uuid4().hex,
            prompt_ids=prompt_ids,
            sampling_params=retry_sampling_params,
        )
        retry_response_ids = retry.token_ids[: self.response_length]
        retry_text = _decode_response(self.tokenizer, retry_response_ids)
        retry_raw_text = _decode_response_with_specials(self.tokenizer, retry_response_ids)
        return GuidanceGeneration(
            text=retry_text,
            response_ids=retry_response_ids,
            response_logprobs=retry.log_probs[: len(retry_response_ids)] if retry.log_probs else None,
            attempts=2,
            raw_text=retry_raw_text or first_raw_text,
            stop_reason=getattr(retry, "stop_reason", None),
        )


EXECUTION_SUMMARY_SECTIONS = (
    "Execution Interpretation",
    "Implemented Algorithm",
    "New Ideas Introduced",
    "Empirical Outcome",
    "Failure / Bottleneck Analysis",
    "Next Guidance Delta",
)


def _extract_summary_line(summary: str, prefix: str) -> str:
    for line in summary.splitlines():
        if line.strip().lower().startswith(prefix.lower() + ":"):
            return line.split(":", 1)[1].strip()
    return ""


def _extract_reusable_idea(summary: str) -> str:
    sections = _parse_execution_summary_sections(summary)
    for section_name in ("Next Guidance Delta", "New Ideas Introduced"):
        value = sections.get(section_name, "").strip()
        if value:
            return value
    return _extract_summary_line(summary, "Reusable idea")


def _parse_execution_summary_sections(summary: str | None) -> dict[str, str]:
    summary = (summary or "").strip()
    if not summary:
        return {}
    canonical_by_lower = {name.lower(): name for name in EXECUTION_SUMMARY_SECTIONS}
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in summary.splitlines():
        stripped = line.strip()
        heading = stripped.lstrip("#").strip().rstrip(":").strip()
        canonical = canonical_by_lower.get(heading.lower())
        if canonical is not None:
            current = canonical
            sections.setdefault(current, [])
            continue
        if current is not None:
            sections.setdefault(current, []).append(line.rstrip())
    return {name: "\n".join(lines).strip() for name, lines in sections.items()}


def _format_verifier_outcome(verification: VerificationResult, *, raw_score_label: str = "Raw C5") -> str:
    raw_score = "None" if verification.raw_score is None else repr(float(verification.raw_score))
    lines = [
        f"Verifier status: {verification.status}",
        f"{raw_score_label}: {raw_score}",
        f"Reward: {float(verification.reward)!r}",
        f"Verifier message: {verification.message}",
    ]
    profile = _format_verified_profile(verification)
    if profile:
        lines.append(profile)
    return "\n".join(lines)


def _format_verified_profile(verification: VerificationResult) -> str:
    if not verification.valid:
        return ""
    artifacts = verification.artifacts or {}
    h_values = artifacts.get("h_values")
    if not isinstance(h_values, list) or not h_values:
        return ""
    try:
        values = [float(value) for value in h_values]
    except (TypeError, ValueError):
        return ""
    n_points = artifacts.get("n_points", len(values))
    c5_bound = artifacts.get("c5_bound", verification.raw_score)
    head = ", ".join(repr(value) for value in values[:6])
    tail = ", ".join(repr(value) for value in values[-4:])
    return (
        "Verified returned profile: "
        f"n_points={int(n_points)}, c5_bound={float(c5_bound)!r}, "
        f"h length={len(values)}, head=[{head}], tail=[{tail}]"
    )


def _fenced_solution(solution: str, *, solution_language: str = "python") -> str:
    solution = (solution or "").strip()
    if not solution:
        return "No solution code was extracted."
    fenced_language = "cpp" if solution_language.lower() in {"cpp", "c++", "cxx"} else "python"
    return f"```{fenced_language}\n" + solution + "\n```"


def build_execution_summary(
    *,
    model_summary: str | None,
    execution_thinking: str,
    solution: str,
    guidance: str,
    verification: VerificationResult,
    solution_language: str = "python",
    raw_score_label: str = "Raw C5",
) -> str:
    parsed = _parse_execution_summary_sections(model_summary)
    model_summary_text = (model_summary or "").strip()
    thinking = (execution_thinking or "").strip() or "Execution thinking was not provided."

    interpretation_parts = [thinking]
    model_interpretation = parsed.get("Execution Interpretation", "").strip()
    if model_interpretation and model_interpretation not in interpretation_parts:
        interpretation_parts.append(model_interpretation)

    implemented_parts: list[str] = []
    model_algorithm = parsed.get("Implemented Algorithm", "").strip()
    if model_algorithm:
        implemented_parts.append(model_algorithm)
    implemented_parts.append(_fenced_solution(solution, solution_language=solution_language))

    new_ideas = parsed.get("New Ideas Introduced", "").strip()
    if not new_ideas:
        if model_summary_text and not parsed:
            new_ideas = model_summary_text
        else:
            new_ideas = "No distinct new idea was provided beyond the submitted guidance."

    failure_analysis = parsed.get("Failure / Bottleneck Analysis", "").strip()
    if not failure_analysis:
        failure_analysis = (
            "No verifier failure reported."
            if verification.valid
            else f"Verifier reported {verification.status}: {verification.message}"
        )

    next_guidance = parsed.get("Next Guidance Delta", "").strip()
    if not next_guidance:
        next_guidance = f"Continue from the submitted guidance: {guidance[:500].strip()}"

    section_values = {
        "Execution Interpretation": "\n\n".join(interpretation_parts).strip(),
        "Implemented Algorithm": "\n\n".join(implemented_parts).strip(),
        "New Ideas Introduced": new_ideas,
        "Empirical Outcome": _format_verifier_outcome(verification, raw_score_label=raw_score_label),
        "Failure / Bottleneck Analysis": failure_analysis,
        "Next Guidance Delta": next_guidance,
    }
    return "\n\n".join(f"{name}\n{section_values[name]}" for name in EXECUTION_SUMMARY_SECTIONS)


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)


def _execution_max_tokens(config: dict[str, Any]) -> int | None:
    if "max_tokens" not in config:
        return 8192
    value = config.get("max_tokens")
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "auto", "context", "unlimited"}:
        return None
    value_int = int(value)
    return None if value_int <= 0 else value_int


def _normalize_task_config(task: dict[str, Any] | str | None) -> dict[str, Any]:
    if task is None:
        return {"id": "erdos_min_overlap"}
    if isinstance(task, str):
        return {"id": task}
    normalized = _jsonable(task)
    if not isinstance(normalized, dict):
        raise TypeError(f"task config must normalize to a dict, got {type(normalized).__name__}")
    return normalized


def _verifier_config_from_task_config(task_config: dict[str, Any]) -> dict[str, Any]:
    frontiercs = _jsonable(task_config.get("frontiercs") or {})
    if not isinstance(frontiercs, dict):
        raise TypeError(f"task.frontiercs must normalize to a dict, got {type(frontiercs).__name__}")
    return frontiercs


def _extract_solution_code(execution_text: str, *, task_spec: TaskSpec | None = None) -> str:
    task_spec = task_spec or get_task_spec("erdos_min_overlap")
    tagged_solution = extract_tag_or_none(execution_text, "solution")
    if tagged_solution is not None:
        return task_spec.solution_extractor(tagged_solution) or tagged_solution.strip()
    return task_spec.solution_extractor(execution_text) or ""


def _verify_execution_without_fallback(
    *,
    execution_text: str,
    guidance: str,
    timeout_s: int,
    task_spec: TaskSpec | None = None,
    verifier_config: dict[str, Any] | None = None,
    initial_error: str | None = None,
) -> ExecutionVerification:
    task_spec = task_spec or get_task_spec("erdos_min_overlap")
    original_execution_text = execution_text
    if initial_error is None:
        verification = task_spec.verify_execution_text(
            execution_text,
            timeout_s=timeout_s,
            config=verifier_config or {},
        )
    else:
        verification = VerificationResult.execution_error(initial_error)
    execution_thinking = extract_tag_or_none(execution_text, "execution_thinking") or ""
    solution = _extract_solution_code(execution_text, task_spec=task_spec)
    model_summary = extract_tag_or_none(execution_text, "summary")
    summary = build_execution_summary(
        model_summary=model_summary,
        execution_thinking=execution_thinking,
        solution=solution,
        guidance=guidance,
        verification=verification,
        solution_language=task_spec.solution_language,
        raw_score_label=task_spec.raw_score_label,
    )
    return ExecutionVerification(
        execution_text=execution_text,
        execution_thinking=execution_thinking,
        solution=solution,
        summary=summary,
        model_summary=model_summary,
        verification=verification,
        fallback_used=False,
        fallback_reason=None,
        original_execution_text=original_execution_text,
    )
