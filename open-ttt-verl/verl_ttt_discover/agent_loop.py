from __future__ import annotations

from typing import Any
from uuid import uuid4

from verl_ttt_discover.archive import PUCTArchive
from verl_ttt_discover.erdos_env import build_erdos_prompt, score_erdos_result
from verl_ttt_discover.sandbox import evaluate_python_code, extract_python_code


def make_group_uid(*, global_step: int | str, uid: str) -> str:
    return f"{global_step}:{uid}"


try:
    from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, AgentLoopMetrics, register
    from verl.utils.profiler import simple_timer
    from verl.workers.rollout.replica import TokenOutput
except ModuleNotFoundError:
    AgentLoopBase = object
    AgentLoopOutput = None
    AgentLoopMetrics = None
    TokenOutput = None

    def register(name: str):
        def decorator(cls):
            return cls

        return decorator

    class simple_timer:
        def __init__(self, name: str, metrics: dict[str, Any]):
            self.name = name
            self.metrics = metrics

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False


@register("ttt_discover_erdos")
class TTTDiscoverAgentLoop(AgentLoopBase):
    """Single-turn TTT-Discover rollout loop for the Erdos task."""

    def __init__(self, *args, budget_s: int = 1000, cpus: int = 1, target_c5: float = 0.3808, **kwargs):
        super().__init__(*args, **kwargs)
        self.budget_s = int(budget_s)
        self.cpus = int(cpus)
        self.target_c5 = float(target_c5)
        if hasattr(self, "rollout_config"):
            self.prompt_length = self.rollout_config.prompt_length
            self.response_length = self.rollout_config.response_length

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> Any:
        if AgentLoopOutput is None:
            raise RuntimeError("verl dependencies are required to run TTTDiscoverAgentLoop")

        extra_info = dict(kwargs.get("extra_info") or {})
        uid = str(kwargs.get("uid") or extra_info.get("uid") or uuid4().hex)
        global_step = kwargs.get("global_steps", kwargs.get("global_step", 0))
        group_uid = make_group_uid(global_step=global_step, uid=uid)
        archive = PUCTArchive(
            extra_info["archive_path"],
            rollout_n=int(extra_info.get("rollout_n", self.config.actor_rollout_ref.rollout.n)),
        )
        state = archive.acquire_group(group_uid)
        prompt = build_erdos_prompt(state, budget_s=self.budget_s, cpus=self.cpus, target_c5=self.target_c5)
        prompt_ids = await self.apply_chat_template([{"role": "user", "content": prompt}])

        metrics: dict[str, Any] = {}
        with simple_timer("generate_sequences", metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
            )

        response_ids = output.token_ids[: self.response_length]
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        code = extract_python_code(response_text)

        reward_score = 0.0
        reward_extra_info: dict[str, Any] = {
            "group_uid": group_uid,
            "state_id": state.id,
            "raw_score": None,
            "valid": False,
            "message": "",
            "error": "",
        }
        if code:
            sandbox_result = evaluate_python_code(code, state=state, timeout_s=self.budget_s)
            if sandbox_result.error is None:
                try:
                    scored = score_erdos_result(
                        sandbox_result.output,
                        code=code,
                        timestep=int(global_step),
                        stdout=sandbox_result.stdout,
                    )
                    reward_score = scored.reward
                    reward_extra_info.update({"raw_score": scored.raw_score, "valid": True, "message": scored.message})
                    archive.submit_child(group_uid, scored.state)
                except Exception as exc:
                    reward_extra_info.update({"error": str(exc)})
                    archive.submit_child(group_uid, None)
            else:
                reward_extra_info.update({"error": sandbox_result.error})
                archive.submit_child(group_uid, None)
        else:
            reward_extra_info.update({"error": "No python code block found"})
            archive.submit_child(group_uid, None)

        response_mask = [1] * len(response_ids)
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=output.log_probs[: len(response_ids)] if output.log_probs else None,
            multi_modal_data={},
            reward_score=reward_score,
            num_turns=2,
            metrics=AgentLoopMetrics(**metrics),
            extra_fields={
                "reward_extra_info": reward_extra_info,
                "turn_scores": [],
                "tool_rewards": [],
                "sample_uid": f"{group_uid}:{uuid4().hex}",
            },
        )
        return output
