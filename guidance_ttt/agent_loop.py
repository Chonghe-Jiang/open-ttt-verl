from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from guidance_ttt.library import GuidanceLibrary
from guidance_ttt.llm_client import make_llm_client
from guidance_ttt.prompts import build_execution_prompt, build_guidance_prompt, build_summary_prompt, extract_tag
from guidance_ttt.state import LLMRequest, LibraryEntry, VerificationResult
from guidance_ttt.tasks.erdos import ERDOS_PROBLEM_PROMPT
from guidance_ttt.verifier.erdos import verify_erdos_solution_text
from guidance_ttt.verifier.sandbox import extract_python_code


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


@register("guidance_execution_erdos")
class GuidanceExecutionAgentLoop(AgentLoopBase):
    """verl agent loop: train guidance tokens, execute/summarize with external LLMs."""

    def __init__(
        self,
        *args,
        execution_llm: dict[str, Any] | None = None,
        summarizer_llm: dict[str, Any] | None = None,
        verifier_timeout_s: int = 60,
        problem_prompt: str = ERDOS_PROBLEM_PROMPT,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.execution_llm_config = execution_llm or {"provider": "mock", "model": "mock-exec"}
        self.summarizer_llm_config = summarizer_llm or {"provider": "mock", "model": "mock-summary"}
        self.execution_client = make_llm_client(self.execution_llm_config)
        self.summarizer_client = make_llm_client(self.summarizer_llm_config)
        self.verifier_timeout_s = int(verifier_timeout_s)
        self.problem_prompt = problem_prompt
        if hasattr(self, "rollout_config"):
            self.response_length = self.rollout_config.response_length

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> Any:
        if not hasattr(self, "server_manager"):
            raise RuntimeError("verl dependencies are required to run GuidanceExecutionAgentLoop")

        extra_info = dict(kwargs.get("extra_info") or {})
        library_path = extra_info.get("library_path") or extra_info.get("archive_path")
        if not library_path:
            raise ValueError("extra_info must include library_path")
        uid = str(kwargs.get("uid") or extra_info.get("uid") or extra_info.get("slot_id") or uuid4().hex)
        global_step = kwargs.get("global_steps", kwargs.get("global_step", 0))
        group_uid = f"{global_step}:{uid}"

        library = GuidanceLibrary(library_path, rollout_n=int(extra_info.get("rollout_n", 1)))
        selected_node = library.acquire_group(group_uid)
        context = library.context_for_node(selected_node)
        selected_entry = context["selected_entry"]
        guidance_prompt = build_guidance_prompt(
            problem_prompt=self.problem_prompt,
            selected_node=selected_node,
            selected_entry=selected_entry,
            global_best_entries=context["global_best_entries"],
            local_failure_entries=context["local_failure_entries"],
        )
        prompt_ids = await self.apply_chat_template([{"role": "user", "content": guidance_prompt.user}])
        output = await self.server_manager.generate(
            request_id=uuid4().hex,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
        )
        response_ids = output.token_ids[: self.response_length]
        guidance_text = _decode_response(self.tokenizer, response_ids)
        guidance = extract_tag(guidance_text, "guidance")

        execution_prompt = build_execution_prompt(
            problem_prompt=self.problem_prompt,
            selected_node=selected_node,
            selected_entry=selected_entry,
            guidance=guidance,
        )
        verification: VerificationResult
        execution_text = ""
        execution_thinking = ""
        solution = ""
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
            execution_thinking = extract_tag(execution_text, "execution_thinking")
            solution = extract_python_code(execution_text) or ""
            verification = verify_erdos_solution_text(execution_text, timeout_s=self.verifier_timeout_s)
        except Exception as exc:
            verification = VerificationResult.execution_error(str(exc))

        summary_prompt = build_summary_prompt(
            selected_entry=selected_entry,
            guidance=guidance,
            execution_thinking=execution_thinking,
            solution=solution,
            reward=verification.reward,
            raw_score=verification.raw_score,
            status=verification.status,
            message=verification.message,
        )
        try:
            summary_response = await self.summarizer_client.complete(
                LLMRequest(
                    system=summary_prompt.system,
                    user=summary_prompt.user,
                    model=self.summarizer_llm_config.get("model", "mock-summary"),
                    temperature=float(self.summarizer_llm_config.get("temperature", 0.0)),
                    max_tokens=int(self.summarizer_llm_config.get("max_tokens", 2048)),
                    metadata={"purpose": "summary"},
                )
            )
            summary = extract_tag(summary_response.text, "summary")
        except Exception as exc:
            summary = (
                f"Outcome: summarizer failed: {exc}\n"
                f"Reusable idea: {guidance[:300]}\n"
                f"Failure mode: {verification.status if not verification.valid else 'None'}"
            )

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
            metadata={"group_uid": group_uid, "selected_node_id": selected_node.id},
        )
        child = library.submit_child(group_uid, entry)

        return build_agent_loop_output(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_logprobs=output.log_probs[: len(response_ids)] if output.log_probs else None,
            reward=verification.reward,
            extra_fields={
                "group_uid": group_uid,
                "selected_node_id": selected_node.id,
                "child_node_id": child.id,
                "guidance": guidance,
                "execution_text": execution_text,
                "execution_thinking": execution_thinking,
                "solution": solution,
                "verification": verification.to_dict(),
                "summary": summary,
                "library_entry_id": entry.id,
            },
        )


def _extract_summary_line(summary: str, prefix: str) -> str:
    for line in summary.splitlines():
        if line.strip().lower().startswith(prefix.lower() + ":"):
            return line.split(":", 1)[1].strip()
    return ""

