# Copyright 2026 Chonghe Jiang and/or contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from verl_ttt_discover.archive import PUCTArchive
from verl_ttt_discover.cpp_sandbox import extract_cpp_code
from verl_ttt_discover.erdos_env import build_erdos_prompt, score_erdos_result
from verl_ttt_discover.polyomino_env import PolyominoEnv, build_polyomino_prompt, evaluate_polyomino_cpp, score_polyomino_result
from verl_ttt_discover.sandbox import evaluate_python_code, extract_python_code


def make_group_uid(*, global_step: int | str, uid: str) -> str:
    return f"{global_step}:{uid}"


def phase1_generation_budget(*, prompt_len: int, response_len: int, phase1_max_tokens: int | None) -> int:
    if phase1_max_tokens is None or phase1_max_tokens <= 0:
        return response_len
    return max(0, min(response_len, int(phase1_max_tokens) - int(prompt_len)))


def _clip_text(text: str | None, limit: int = 1200) -> str:
    if not text:
        return ""
    if len(text) <= limit * 2:
        return text
    return text[:limit] + "\n...[truncated]...\n" + text[-limit:]


def _append_rollout_debug(*, archive_path: str, row: dict[str, Any]) -> None:
    debug_path = Path(archive_path).parent / "rollout_debug.jsonl"
    lock_path = debug_path.with_suffix(debug_path.suffix + ".lock")
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        with debug_path.open("a") as debug_file:
            debug_file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        fcntl.flock(lock_file, fcntl.LOCK_UN)


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

    def __init__(
        self,
        *args,
        budget_s: int = 1000,
        cpus: int = 1,
        target_c5: float = 0.3808,
        phase1_max_tokens: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.budget_s = int(budget_s)
        self.cpus = int(cpus)
        self.target_c5 = float(target_c5)
        self.phase1_max_tokens = None if phase1_max_tokens is None else int(phase1_max_tokens)
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
        sampling_params = dict(sampling_params)
        sampling_params["max_tokens"] = phase1_generation_budget(
            prompt_len=len(prompt_ids),
            response_len=self.response_length,
            phase1_max_tokens=self.phase1_max_tokens,
        )

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

        _append_rollout_debug(
            archive_path=extra_info["archive_path"],
            row={
                "group_uid": group_uid,
                "state_id": state.id,
                "global_step": global_step,
                "valid": reward_extra_info["valid"],
                "reward_score": reward_score,
                "raw_score": reward_extra_info["raw_score"],
                "message": reward_extra_info["message"],
                "error": reward_extra_info["error"],
                "has_python_block": "```python" in response_text,
                "has_extracted_code": code is not None,
                "response_chars": len(response_text),
                "code_chars": len(code or ""),
                "response_excerpt": _clip_text(response_text),
                "code_excerpt": _clip_text(code or ""),
            },
        )

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


@register("ttt_discover_polyomino")
class TTTDiscoverPolyominoAgentLoop(AgentLoopBase):
    """Single-turn TTT-Discover rollout loop for Frontier-CS Polyomino."""

    def __init__(
        self,
        *args,
        budget_s: int = 8,
        phase1_max_tokens: int | None = None,
        invalid_reward: float = -0.1,
        cpp_prefill_fallback: bool = True,
        use_chat_template: bool = False,
        cpp_min_tokens: int = 512,
        faithful_discover: bool = False,
        update_puct_per_rollout: bool = False,
        official_concurrency: int = 16,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.budget_s = int(budget_s)
        self.phase1_max_tokens = None if phase1_max_tokens is None else int(phase1_max_tokens)
        self.invalid_reward = float(invalid_reward)
        self.cpp_prefill_fallback = bool(cpp_prefill_fallback)
        self.use_chat_template = bool(use_chat_template)
        self.cpp_min_tokens = int(cpp_min_tokens)
        self.faithful_discover = bool(faithful_discover)
        self.update_puct_per_rollout = bool(update_puct_per_rollout)
        self.official_concurrency = int(official_concurrency)
        self.cpp_completion_prefix = "\n```cpp\n#include <bits/stdc++.h>\nusing namespace std;\n\n"
        if hasattr(self, "rollout_config"):
            self.prompt_length = self.rollout_config.prompt_length
            self.response_length = self.rollout_config.response_length

    def _encode_plain(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    async def _generate_once(self, prompt_ids: list[int], sampling_params: dict[str, Any]) -> TokenOutput:
        return await self.server_manager.generate(
            request_id=uuid4().hex,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
        )

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> Any:
        if AgentLoopOutput is None:
            raise RuntimeError("verl dependencies are required to run TTTDiscoverPolyominoAgentLoop")

        extra_info = dict(kwargs.get("extra_info") or {})
        uid = str(kwargs.get("uid") or extra_info.get("uid") or uuid4().hex)
        global_step = kwargs.get("global_steps", kwargs.get("global_step", 0))
        group_uid = make_group_uid(global_step=global_step, uid=uid)
        archive = PUCTArchive(
            extra_info["archive_path"],
            rollout_n=int(extra_info.get("rollout_n", self.config.actor_rollout_ref.rollout.n)),
            update_puct_per_rollout=bool(extra_info.get("update_puct_per_rollout", self.update_puct_per_rollout)),
        )
        state = archive.acquire_group(group_uid)
        if self.faithful_discover:
            os.environ["FRONTIER_CS_POLYOMINO_CONCURRENCY"] = str(self.official_concurrency)
        env = PolyominoEnv(
            initial_state=state,
            problem_type=str(extra_info.get("task", "frontier_cs_polyomino")),
            log_path=str(Path(extra_info["archive_path"]).parent),
            eval_timeout=self.budget_s,
            num_cpus_per_task=int(extra_info.get("num_cpus_per_task", 1)),
        )
        prompt = env.get_question() if self.faithful_discover else build_polyomino_prompt(state, budget_s=self.budget_s)
        if self.use_chat_template:
            prompt_ids = await self.apply_chat_template([{"role": "user", "content": prompt}])
            candidate_prefix = ""
        else:
            candidate_prefix = self.cpp_completion_prefix
            prompt_ids = self._encode_plain(prompt + candidate_prefix)
        sampling_params = dict(sampling_params)
        sampling_params["max_tokens"] = phase1_generation_budget(
            prompt_len=len(prompt_ids),
            response_len=self.response_length,
            phase1_max_tokens=self.phase1_max_tokens,
        )
        if sampling_params["max_tokens"] > 0 and self.cpp_min_tokens > 0:
            sampling_params["min_tokens"] = min(self.cpp_min_tokens, sampling_params["max_tokens"])
        if not self.use_chat_template:
            sampling_params.setdefault("stop", ["```"])
        requested_max_tokens = sampling_params.get("max_tokens")
        requested_min_tokens = sampling_params.get("min_tokens")

        metrics: dict[str, Any] = {}
        used_prefill_fallback = False
        prefill_ids: list[int] = []
        prefill_offset: int | None = None
        with simple_timer("generate_sequences", metrics):
            first_output: TokenOutput = await self._generate_once(prompt_ids, sampling_params)

            response_ids = first_output.token_ids[: self.response_length]
            response_logprobs = first_output.log_probs[: len(response_ids)] if first_output.log_probs else None
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            raw_response_text = self.tokenizer.decode(response_ids, skip_special_tokens=False)
            candidate_text = candidate_prefix + response_text
            raw_candidate_text = candidate_prefix + raw_response_text
            code = extract_cpp_code(candidate_text)

            if (
                not self.faithful_discover
                and self.use_chat_template
                and code is None
                and self.cpp_prefill_fallback
                and self.response_length > 16
            ):
                used_prefill_fallback = True
                prefill_ids = self._encode_plain("\n```cpp\n")
                prefill_offset = 0
                fallback_prompt_ids = prompt_ids + prefill_ids
                fallback_params = dict(sampling_params)
                fallback_budget = max(0, self.response_length - len(prefill_ids))
                fallback_params["max_tokens"] = fallback_budget
                if fallback_budget > 0:
                    second_output: TokenOutput = await self._generate_once(fallback_prompt_ids, fallback_params)
                    continuation_ids = second_output.token_ids[:fallback_budget]
                else:
                    continuation_ids = []
                response_ids = (prefill_ids + continuation_ids)[: self.response_length]
                response_logprobs = None
                response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                raw_response_text = self.tokenizer.decode(response_ids, skip_special_tokens=False)
                candidate_text = response_text
                raw_candidate_text = raw_response_text
                code = extract_cpp_code(candidate_text)

        reward_score = 0.0 if self.faithful_discover else self.invalid_reward
        reward_extra_info: dict[str, Any] = {
            "group_uid": group_uid,
            "state_id": state.id,
            "raw_score": None,
            "valid": False,
            "message": "",
            "error": "",
        }
        if self.faithful_discover:
            step = env.step_response(candidate_text, int(global_step), parse_success=True)
            code = step.parsed_code or None
            reward_score = step.reward
            reward_extra_info.update(
                {
                    "raw_score": step.verify_result.raw_score if step.verify_result.correctness > 0 else None,
                    "valid": step.verify_result.correctness > 0,
                    "message": step.verify_result.msg if step.verify_result.correctness > 0 else "",
                    "error": "" if step.verify_result.correctness > 0 else step.verify_result.msg,
                }
            )
            archive.submit_child(group_uid, step.next_state)
        elif code:
            raw_score, message = evaluate_polyomino_cpp(code, timeout_s=self.budget_s)
            if raw_score > 0:
                scored = score_polyomino_result(
                    raw_score=raw_score,
                    code=code,
                    timestep=int(global_step),
                    message=message,
                )
                reward_score = scored.reward
                reward_extra_info.update({"raw_score": scored.raw_score, "valid": True, "message": scored.message})
                archive.submit_child(group_uid, scored.state)
            else:
                reward_extra_info.update({"raw_score": raw_score, "error": message})
                archive.submit_child(group_uid, None)
        else:
            reward_extra_info.update({"error": "No C++ code block found"})
            archive.submit_child(group_uid, None)

        if used_prefill_fallback and prefill_ids and prefill_offset is not None:
            response_mask = [1] * len(response_ids)
            for idx in range(prefill_offset, min(len(response_ids), prefill_offset + len(prefill_ids))):
                response_mask[idx] = 0
        else:
            response_mask = [1] * len(response_ids)

        _append_rollout_debug(
            archive_path=extra_info["archive_path"],
            row={
                "group_uid": group_uid,
                "state_id": state.id,
                "global_step": global_step,
                "valid": reward_extra_info["valid"],
                "reward_score": reward_score,
                "raw_score": reward_extra_info["raw_score"],
                "message": reward_extra_info["message"],
                "error": reward_extra_info["error"],
                "task": "frontier_cs_polyomino",
                "prompt_mode": "chat_template" if self.use_chat_template else "raw_completion_cpp_prefix",
                "candidate_prefix_chars": len(candidate_prefix),
                "cpp_min_tokens": self.cpp_min_tokens,
                "sampling_max_tokens": requested_max_tokens,
                "sampling_min_tokens": requested_min_tokens,
                "used_prefill_fallback": used_prefill_fallback,
                "has_cpp_block": "```cpp" in candidate_text or "```c++" in candidate_text,
                "has_extracted_code": code is not None,
                "faithful_discover": self.faithful_discover,
                "update_puct_per_rollout": self.update_puct_per_rollout,
                "official_concurrency": self.official_concurrency,
                "response_tokens": len(response_ids),
                "response_chars": len(response_text),
                "candidate_chars": len(candidate_text),
                "code_chars": len(code or ""),
                "response_excerpt": _clip_text(response_text),
                "raw_response_excerpt": _clip_text(raw_response_text),
                "candidate_excerpt": _clip_text(candidate_text),
                "raw_candidate_excerpt": _clip_text(raw_candidate_text),
                "code_excerpt": _clip_text(code or ""),
                "kwargs_keys": sorted(str(key) for key in kwargs.keys()),
            },
        )

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            multi_modal_data={},
            reward_score=reward_score,
            num_turns=2 if used_prefill_fallback else 1,
            metrics=AgentLoopMetrics(**metrics),
            extra_fields={
                "reward_extra_info": reward_extra_info,
                "turn_scores": [],
                "tool_rewards": [],
                "sample_uid": f"{group_uid}:{uuid4().hex}",
            },
        )
