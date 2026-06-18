from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from guidance_ttt.library import GuidanceLibrary
from guidance_ttt.llm_client import make_llm_client
from guidance_ttt.prompts import (
    ERDOS_MINIMAX_EXECUTION_TEMPLATE,
    build_execution_prompt,
    build_guidance_prompt,
    extract_guidance_or_format_error,
    extract_tag,
)
from guidance_ttt.state import LLMRequest, LibraryEntry, VerificationResult
from guidance_ttt.tasks.erdos import ERDOS_PROBLEM_PROMPT
from guidance_ttt.verifier.erdos import verify_erdos_solution_text
from guidance_ttt.verifier.sandbox import extract_python_code


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
class GuidanceExecutionAgentLoop(AgentLoopBase):
    """verl agent loop: train guidance tokens, execute/summarize with external LLMs."""

    def __init__(
        self,
        *args,
        execution_llm: dict[str, Any] | None = None,
        verifier_timeout_s: int = 60,
        problem_prompt: str = ERDOS_PROBLEM_PROMPT,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.execution_llm_config = execution_llm or self._execution_llm_from_rollout_config() or {
            "provider": "mock",
            "model": "mock-exec",
        }
        self.execution_client = make_llm_client(self.execution_llm_config)
        self.verifier_timeout_s = int(verifier_timeout_s)
        self.problem_prompt = problem_prompt
        if hasattr(self, "rollout_config"):
            self.response_length = self.rollout_config.response_length

    def _execution_llm_from_rollout_config(self) -> dict[str, Any] | None:
        rollout_config = getattr(self, "rollout_config", None)
        agent_config = _config_get(rollout_config, "agent")
        agent_loop_config_path = _config_get(agent_config, "agent_loop_config_path")
        if not agent_loop_config_path:
            return None
        default_agent_loop = _config_get(agent_config, "default_agent_loop") or "guidance_execution_erdos"
        try:
            from omegaconf import OmegaConf

            loaded = OmegaConf.load(Path(str(agent_loop_config_path)).expanduser())
            for agent_loop_config in loaded:
                if _config_get(agent_loop_config, "name") != default_agent_loop:
                    continue
                execution_llm = _config_get(agent_loop_config, "execution_llm")
                if execution_llm:
                    return OmegaConf.to_container(execution_llm, resolve=True)
        except Exception:
            return None
        return None

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> Any:
        if not hasattr(self, "server_manager"):
            raise RuntimeError("verl dependencies are required to run GuidanceExecutionAgentLoop")

        extra_info = dict(kwargs.get("extra_info") or {})
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
        )
        selected_node = library.acquire_group(group_uid, visible_timestep_exclusive=int(global_step))
        context = library.context_for_node(selected_node, visible_timestep_exclusive=int(global_step))
        selected_entry = context["selected_entry"]
        guidance_prompt = build_guidance_prompt(
            problem_prompt=self.problem_prompt,
            selected_node=selected_node,
            selected_entry=selected_entry,
            global_best_entries=context["global_best_entries"],
            local_failure_entries=context["local_failure_entries"],
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
            problem_prompt=self.problem_prompt,
            selected_node=selected_node,
            selected_entry=selected_entry,
            guidance=guidance,
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
                    max_tokens=int(self.execution_llm_config.get("max_tokens", 8192)),
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
        execution_result = _verify_execution_with_minimax_fallback(
            execution_text=execution_text,
            guidance=guidance,
            timeout_s=self.verifier_timeout_s,
            initial_error=execution_error,
        )
        execution_text = execution_result.execution_text
        execution_thinking = execution_result.execution_thinking
        solution = execution_result.solution
        summary = execution_result.summary
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
            reusable_idea=_extract_summary_line(summary, "Reusable idea"),
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
                "execution_text": execution_text,
                "execution_provider": self.execution_llm_config.get("provider", "mock"),
                "execution_model": self.execution_llm_config.get("model", "mock-exec"),
                "execution_response_metadata": execution_response_metadata,
                "execution_response_usage": execution_response_usage,
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
            },
        )

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


def _extract_summary_line(summary: str, prefix: str) -> str:
    for line in summary.splitlines():
        if line.strip().lower().startswith(prefix.lower() + ":"):
            return line.split(":", 1)[1].strip()
    return ""


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)


def _verify_execution_with_minimax_fallback(
    *,
    execution_text: str,
    guidance: str,
    timeout_s: int,
    initial_error: str | None = None,
) -> ExecutionVerification:
    original_execution_text = execution_text
    if initial_error is None:
        verification = verify_erdos_solution_text(execution_text, timeout_s=timeout_s)
    else:
        verification = VerificationResult.execution_error(initial_error)
    if verification.valid:
        return ExecutionVerification(
            execution_text=execution_text,
            execution_thinking=extract_tag(execution_text, "execution_thinking"),
            solution=extract_python_code(execution_text) or "",
            summary=_execution_summary_or_fallback(execution_text=execution_text, guidance=guidance),
            verification=verification,
            fallback_used=False,
            fallback_reason=None,
            original_execution_text=original_execution_text,
        )

    fallback_reason = verification.status
    fallback_text = _minimax_fallback_execution_text(fallback_reason=fallback_reason, guidance=guidance)
    fallback_verification = verify_erdos_solution_text(fallback_text, timeout_s=timeout_s)
    if fallback_verification.valid:
        return ExecutionVerification(
            execution_text=fallback_text,
            execution_thinking=extract_tag(fallback_text, "execution_thinking"),
            solution=extract_python_code(fallback_text) or "",
            summary=_execution_summary_or_fallback(execution_text=fallback_text, guidance=guidance),
            verification=fallback_verification,
            fallback_used=True,
            fallback_reason=fallback_reason,
            original_execution_text=original_execution_text,
        )

    return ExecutionVerification(
        execution_text=execution_text,
        execution_thinking=extract_tag(execution_text, "execution_thinking"),
        solution=extract_python_code(execution_text) or "",
        summary=_execution_summary_or_fallback(execution_text=execution_text, guidance=guidance),
        verification=verification,
        fallback_used=False,
        fallback_reason=f"fallback_failed:{fallback_verification.status}",
        original_execution_text=original_execution_text,
    )


def _minimax_fallback_execution_text(*, fallback_reason: str, guidance: str) -> str:
    return f"""```python
{ERDOS_MINIMAX_EXECUTION_TEMPLATE}
```

<summary>
Outcome hypothesis: minimax fallback supplies a verifier-valid small-n construction after execution failure.
Reusable idea: Copy the deterministic n=19 minimax fallback, then improve it with tighter SLSQP search.
Risk / possible failure mode: fallback_reason={fallback_reason}
What future guidance should preserve: {guidance[:240]}
What future guidance should change: make the execution model return parseable projected minimax code directly.
</summary>

<execution_thinking>
Used minimax fallback because the original execution output was not verifier-valid: {fallback_reason}.
</execution_thinking>
"""


def _execution_summary_or_fallback(*, execution_text: str, guidance: str) -> str:
    summary = extract_tag(execution_text, "summary")
    if summary != execution_text.strip():
        return summary
    return (
        "Outcome hypothesis: execution did not provide summary.\n"
        f"Reusable idea: {guidance[:300]}\n"
        "Risk / possible failure mode: missing_summary\n"
        "What future guidance should preserve: selected library context.\n"
        "What future guidance should change: request clearer executable plan."
    )
