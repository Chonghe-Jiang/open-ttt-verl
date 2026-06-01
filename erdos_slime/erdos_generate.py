from __future__ import annotations
import asyncio
import math
import os
import json
import aiohttp

BUFFER_PATH = '/root/workspace/erdos/data/shared_buffer.json'


class PUCTBuffer:
    def __init__(self, c: float = 1.0, maxsize: int = 64):
        self.c = c
        self.maxsize = maxsize

    def _load(self) -> list:
        try:
            with open(BUFFER_PATH, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save(self, buf: list):
        os.makedirs(os.path.dirname(BUFFER_PATH), exist_ok=True)
        with open(BUFFER_PATH, 'w') as f:
            json.dump(buf, f)

    def select(self):
        buf = self._load()
        if not buf:
            return None
        total_n = sum(e['n'] for e in buf)
        scores = [
            e['reward'] + self.c * math.sqrt(math.log(total_n + 1) / (e['n'] + 1))
            for e in buf
        ]
        return buf[scores.index(max(scores))]


_BUFFER = PUCTBuffer()
_TOKENIZER = None


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained(
            '/root/workspace/models/Qwen3-8B', trust_remote_code=True
        )
    return _TOKENIZER


def _build_prompt(best_entry=None) -> str:
    current_c5 = best_entry['c5'] if best_entry else 0.381
    if best_entry:
        code_section = (
            f"Here is the current best code (C5={current_c5:.6f}):\n"
            f"```python\n{best_entry['code']}\n```\n\n"
            "Improve upon this. Try different algorithmic ideas, "
            "adjust hyperparameters, or use a completely different approach."
        )
        construction_section = (
            f"\nThe current best construction (n={len(best_entry['h'])} points) "
            "is available as `initial_h_values` (pre-imported numpy array).\n"
        )
    else:
        code_section = "Write code to find an initial good solution."
        construction_section = ""

    return f"""You are an expert in harmonic analysis, numerical optimization, and mathematical discovery.
Your task is to find an improved upper bound for the Erdos minimum overlap problem constant C5.

## Problem
Find a step function h: [0, 2] -> [0, 1] that minimizes:
  C5 = max_k integral h(x)(1 - h(x+k)) dx

Discretized as: C5 = max(np.correlate(h, 1-h, mode="full") * (2.0/n_points))

Constraints:
1. 0 <= h[i] <= 1 for all i
2. sum(h) == n_points / 2

## Rules
- Define `run(seed=42, budget_s=60, **kwargs)` returning `(h_values, c5_bound, n_points)`
- Use numpy, scipy, cvxpy only. No filesystem/network IO.
- `evaluate_erdos_solution(result)` and `initial_h_values` are pre-imported.
- Complete within budget_s seconds. Return best solution found.

Current best: C5 <= {current_c5:.6f}
Target: C5 <= 0.38080
{construction_section}
Lower C5 = higher reward.

{code_section}

WARNING: DO NOT write long mathematical derivations or endless comments. 
You MUST output the code immediately. 
Write your solution in a SINGLE ```python code block. 
Keep your reasoning brief (under 300 words) before providing the code.
"""


async def generate(args, sample, sampling_params):
    """
    custom_generate_func: 직접 SGLang에 요청해서 생성까지 완료해야 함.
    slime은 이 함수 반환 후 내부 generate를 다시 호출하지 않음.
    """
    from slime.utils.http_utils import post

    best_entry = _BUFFER.select()
    tok = _get_tokenizer()
    prompt_text = _build_prompt(best_entry)
    messages = [{"role": "user", "content": prompt_text}]
    prompt_str = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tok(prompt_str, add_special_tokens=False)["input_ids"]

    # SGLang에 직접 요청
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    payload = {
        "input_ids": prompt_ids,
        "sampling_params": sampling_params,
        "return_logprob": True,
    }

    output = await post(url, payload)

    # response token logprobs 추출
    if "output_token_logprobs" in output["meta_info"]:
        new_response_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
        new_response_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
    else:
        new_response_tokens = []
        new_response_log_probs = []

    # sample 필드 채우기 (slime sglang_rollout.py와 동일한 방식)
    sample.prompt = prompt_str
    sample.tokens = prompt_ids + new_response_tokens
    sample.response_length = len(new_response_tokens)
    sample.response = output["text"]
    sample.rollout_log_probs = new_response_log_probs

    # finish_reason 처리
    finish_reason = output["meta_info"].get("finish_reason", {}).get("type", "stop")
    if finish_reason == "length":
        sample.status = sample.Status.TRUNCATED
    elif finish_reason == "abort":
        sample.status = sample.Status.ABORTED
    else:
        sample.status = sample.Status.COMPLETED

    return sample