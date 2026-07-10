# Copyright 2026 Chonghe Jiang and/or contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from verl_ttt_discover.state import DiscoveryState


def last_codeblock_postprocess(
    input_text: str,
    codeblock_seps: list[str] | None = None,
    *,
    last_response_strict: bool = True,
    keep_separators: bool = True,
) -> str:
    """Mirror Discover's final-code-block parser.

    Discover tasks define the language list in the environment, then parse the
    final matching code block from the assistant response before verification.
    """

    if codeblock_seps is None:
        codeblock_seps = ["python", "cpp", "java", "cuda"]
    languages_pattern = "|".join(map(re.escape, codeblock_seps))
    pattern = re.compile(rf"```({languages_pattern})\n(?!```)(.*?)(?:\n```)?(?=\n```|$)", re.DOTALL)
    matches = list(pattern.finditer(input_text))
    if not matches:
        return "" if last_response_strict else input_text

    last_match = matches[-1]
    language = last_match.group(1)
    code_content = last_match.group(2).rstrip()
    if not code_content or not code_content.strip():
        return "" if last_response_strict else input_text
    if keep_separators:
        return f"```{language}\n{code_content}\n```"
    return code_content


@dataclass
class VerifyResult:
    reward: float
    msg: str
    correctness: float
    raw_score: float
    result_construction: Any
    stdout: str
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiscoverStep:
    reward: float
    next_state: DiscoveryState | None
    metrics: dict[str, Any]
    parsed_code: str
    correct_format: bool
    verify_result: VerifyResult


class BaseRewardEvaluator:
    fail_score = 0.0
    worst_perf_log = 0.0

    def __init__(
        self,
        *,
        problem_type: str,
        log_dir: str,
        eval_timeout: int,
        num_cpus_per_task: int = 1,
    ) -> None:
        self.problem_type = problem_type
        self.log_dir = log_dir
        self.eval_timeout = int(eval_timeout)
        self.num_cpus_per_task = int(num_cpus_per_task)

    def get_reward(self, code: str, state: DiscoveryState) -> dict[str, Any]:
        raise NotImplementedError

    def failure_entry(self, msg: str) -> dict[str, Any]:
        return {
            "reward": self.fail_score,
            "msg": msg,
            "correctness": 0.0,
            "raw_score": self.worst_perf_log,
            "result_construction": None,
            "stdout": getattr(self, "_last_stdout", ""),
        }


class Environment:
    reward_function: type[BaseRewardEvaluator] = BaseRewardEvaluator
    state_type: type[DiscoveryState] = DiscoveryState
    max_construction_len: int | None = None

    def __init__(
        self,
        *,
        initial_state: DiscoveryState,
        problem_type: str = "",
        log_path: str = "",
        eval_timeout: int = 300,
        num_cpus_per_task: int = 1,
    ) -> None:
        self.initial_state = initial_state
        self.state = initial_state
        self.problem_type = problem_type
        self.log_path = log_path
        self.eval_timeout = int(eval_timeout)
        self.num_cpus_per_task = int(num_cpus_per_task)

    @classmethod
    def create_initial_state(cls, problem_type: str) -> DiscoveryState:
        return cls.state_type(timestep=-1, construction=None, code="", value=0.0, raw_score=0.0)

    def get_question(self) -> str:
        raise NotImplementedError

    def is_maximize(self) -> bool:
        return True

    def _get_code_languages(self) -> list[str]:
        return ["python"]

    def _should_keep_code_separators(self) -> bool:
        return True

    def check_format(self, parsed_code: str) -> bool:
        return bool(parsed_code and parsed_code.strip())

    def check_answer(self, parsed_code: str) -> VerifyResult:
        if not self.check_format(parsed_code):
            return VerifyResult(
                reward=0.0,
                msg="Invalid code",
                correctness=0.0,
                raw_score=0.0,
                result_construction=None,
                stdout="",
            )

        task = self.reward_function(
            problem_type=self.problem_type,
            log_dir=self.log_path,
            eval_timeout=self.eval_timeout,
            num_cpus_per_task=self.num_cpus_per_task,
        )
        out = task.get_reward(parsed_code, state=self.state)
        return VerifyResult(
            reward=float(out["reward"]),
            msg=str(out["msg"]),
            correctness=float(out["correctness"]),
            raw_score=float(out["raw_score"]),
            result_construction=out.get("result_construction"),
            stdout=str(out.get("stdout", "")),
            metrics=dict(out.get("metrics", {})),
        )

    def _create_next_state(self, step_idx: int, parsed_code: str, outs: VerifyResult) -> DiscoveryState:
        value = outs.raw_score if self.is_maximize() else -outs.raw_score
        return self.state_type(
            timestep=step_idx,
            construction=outs.result_construction,
            code=parsed_code,
            value=value,
            raw_score=outs.raw_score,
            observation=outs.stdout,
        )

    def step_response(self, response: str, step_idx: int, *, parse_success: bool = True) -> DiscoverStep:
        parsed_code = last_codeblock_postprocess(
            response,
            codeblock_seps=self._get_code_languages(),
            keep_separators=self._should_keep_code_separators(),
        )
        correct_format = bool(parse_success) and self.check_format(parsed_code)
        outs = self.check_answer(parsed_code)
        next_state = self._create_next_state(step_idx, parsed_code, outs) if outs.correctness > 0 else None
        metrics = {
            "format": float(correct_format),
            "reward": outs.reward,
            "correctness": outs.correctness,
            "raw_score": outs.raw_score if outs.correctness > 0 else None,
            "initial_raw_score": self.initial_state.value,
            "msg": outs.msg,
            "prompt": self.get_question(),
            "response": response,
            "parsed_code": parsed_code,
        }
        metrics.update(outs.metrics)
        return DiscoverStep(
            reward=outs.reward,
            next_state=next_state,
            metrics=metrics,
            parsed_code=parsed_code,
            correct_format=correct_format,
            verify_result=outs,
        )
