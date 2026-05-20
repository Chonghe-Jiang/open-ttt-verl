"""
Yuchen-aligned iterative tool-calling rollout for Frontier-CS.

Protocol (matches Human-Agent-Society/kelp PR#11):
- Each rollout = up to max_turns (8) of: model emits Python solution → call
  evaluate_solution tool (Docker eval, returns score 0-100) → model sees score
  and any error message → revise.
- temperature=0.9, max_tokens=2048 per turn.
- Prompt includes starter code (initial_greedy.py / seed solution) when available.
- Score for a rollout = max score across all 8 turns (model's best attempt).

This module is shared by:
- gen_solutions_iterative.py: base eval (sample N rollouts per task with this protocol)
- ttt_faithful_iterative.py: TTT-Discover RL (each step samples G rollouts with this protocol)
"""
from __future__ import annotations
import json
import re
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import requests


# Same prompt structure as Yuchen kelp PR cs_frontier.py:build_prompt
SYSTEM_PROMPT = (
    "You are an expert programmer solving a coding optimization problem "
    "from the Frontier-CS benchmark."
)


def build_initial_user_message(task_meta: dict, readme: str, starter_code: str | None,
                                language: str = "python") -> str:
    problem_id = f"{task_meta['problem']}/{task_meta['variant']}" if task_meta.get('variant') else task_meta['problem']
    msg = (
        f"**Problem**: {problem_id}\n"
        f"**Language**: {language}\n\n"
        f"## Problem Statement\n\n{readme}\n\n"
    )
    if starter_code:
        msg += f"## Starter Code\n\n```{language}\n{starter_code}\n```\n\n"
    msg += (
        "Write your solution and use the `evaluate_solution` tool to test it. "
        "Your score is 0-100 based on solution quality. Higher is better. "
        "You have up to 8 attempts to improve your solution.\n\n"
        "When you have a solution, output it in this exact format:\n"
        "```python\n# your code here\n```\n"
        "Then I will run it and tell you the score, then you may revise."
    )
    return msg


def extract_python_block(text: str) -> str:
    m = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).rstrip() + "\n"
    m = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).rstrip() + "\n"
    return ""


def vllm_chat_complete(server: str, model: str, messages: list[dict], max_tokens: int,
                        temperature: float, top_p: float, timeout: int) -> dict:
    """Call vLLM /v1/chat/completions with chat-format messages."""
    payload = {
        "model": model, "messages": messages, "max_tokens": max_tokens,
        "temperature": temperature, "top_p": top_p, "stream": False,
        "logprobs": True, "top_logprobs": 1,  # for IS correction in TTT-Discover
    }
    r = requests.post(f"{server}/v1/chat/completions", json=payload, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    choice = j["choices"][0]
    text = choice["message"]["content"]
    lp_obj = choice.get("logprobs") or {}
    token_logprobs = []
    if lp_obj.get("content"):
        token_logprobs = [t.get("logprob", 0.0) for t in lp_obj["content"]]
    return {
        "text": text,
        "token_logprobs": token_logprobs,
        "completion_tokens": j.get("usage", {}).get("completion_tokens", 0),
    }


def evaluate_with_frontier(frontier_root: Path, problem: str, variant: str | None,
                           code: str, scratch: Path, timeout: int = 1800) -> tuple[float, str]:
    """Run frontier eval Docker on one solution. Returns (reward in [0,1], stdout_tail)."""
    if not code.strip():
        return 0.0, "empty code"
    sol = scratch / f"sol_{int(time.time()*1e6)}_{uuid.uuid4().hex[:6]}.py"
    sol.write_text(code)
    pid = f"{problem}/{variant}" if variant else problem
    cmd = ["frontier", "eval", "research", pid, str(sol), "--backend", "docker"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        out = proc.stdout
        m = re.search(r"Score:\s*([\-\d\.eE]+)", out)
        score = float(m.group(1)) if m else 0.0
        if score < 0:
            score = 0.0
        # short feedback to feed back to model
        feedback = out[-800:] if proc.returncode == 0 else (proc.stderr or out)[-800:]
        return score / 100.0, feedback
    except subprocess.TimeoutExpired:
        return 0.0, f"<timeout after {timeout}s>"
    except Exception as e:
        return 0.0, f"<exception: {e}>"
    finally:
        try:
            sol.unlink()
        except Exception:
            pass


def do_iterative_rollout(
    server: str,
    model: str,
    task_meta: dict,
    readme: str,
    starter_code: str | None,
    frontier_root: Path,
    scratch: Path,
    max_turns: int = 8,
    max_tokens_per_turn: int = 2048,
    temperature: float = 0.9,
    top_p: float = 0.95,
    gen_timeout: int = 600,
    eval_timeout: int = 1200,
    extra_lora_in_prompt: dict | None = None,
) -> dict:
    """One iterative rollout = up to max_turns of (generate, evaluate, feedback).
    Returns:
      {
        "final_score": float (max score across turns, in [0,1]),
        "best_code": str (code that achieved final_score),
        "turns": [{"text": ..., "code": ..., "score": ..., "feedback": ...}, ...],
        "messages": [{"role": ..., "content": ...}, ...],  full chat history
      }
    """
    problem = task_meta["problem"]
    variant = task_meta.get("variant")

    user_msg = build_initial_user_message(task_meta, readme, starter_code,
                                           language=task_meta.get("language", "python"))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    turns = []
    best_score = 0.0
    best_code = ""

    for turn_idx in range(max_turns):
        # Generate
        try:
            gen = vllm_chat_complete(server, model, messages,
                                      max_tokens=max_tokens_per_turn,
                                      temperature=temperature, top_p=top_p,
                                      timeout=gen_timeout)
        except Exception as e:
            turns.append({"turn": turn_idx, "error": f"gen failed: {e}",
                          "text": "", "code": "", "score": 0.0, "feedback": ""})
            break

        text = gen["text"]
        code = extract_python_block(text)

        if not code.strip():
            # model didn't output a code block — feed back hint and try again
            feedback = ("No Python code block found in your response. "
                        "Please output your solution as ```python\n...\n``` block.")
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": feedback})
            turns.append({"turn": turn_idx, "text": text, "code": "",
                          "score": 0.0, "feedback": feedback,
                          "token_logprobs": gen.get("token_logprobs", []),
                          "completion_tokens": gen.get("completion_tokens", 0)})
            continue

        # Evaluate
        score, feedback = evaluate_with_frontier(frontier_root, problem, variant,
                                                  code, scratch, timeout=eval_timeout)
        turns.append({
            "turn": turn_idx, "text": text, "code": code,
            "score": score, "feedback": feedback,
            "token_logprobs": gen.get("token_logprobs", []),
            "completion_tokens": gen.get("completion_tokens", 0),
        })

        if score > best_score:
            best_score = score
            best_code = code

        # Build feedback message for next turn
        if score >= 1.0 - 1e-6:
            # already perfect, stop
            break
        feedback_msg = (
            f"Your solution scored {score*100:.2f}/100.\n\n"
            f"Evaluator output (last 800 chars):\n```\n{feedback[-800:]}\n```\n\n"
        )
        if turn_idx < max_turns - 1:
            feedback_msg += (
                "Please revise your solution to improve the score. "
                "Output the full revised solution as ```python\n...\n``` block."
            )
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": feedback_msg})

    return {
        "final_score": best_score,
        "best_code": best_code,
        "turns": turns,
        "messages": messages,
        "task_meta": task_meta,
    }


def do_rollouts_parallel(
    *, server: str, model: str, task_meta: dict, readme: str, starter_code: str | None,
    frontier_root: Path, scratch: Path,
    num_rollouts: int, max_turns: int = 8, max_tokens_per_turn: int = 2048,
    temperature: float = 0.9, top_p: float = 0.95,
    concurrency: int = 4, gen_timeout: int = 600, eval_timeout: int = 1200,
) -> list[dict]:
    """Sample N independent rollouts with bounded concurrency."""
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(do_iterative_rollout,
                           server=server, model=model, task_meta=task_meta,
                           readme=readme, starter_code=starter_code,
                           frontier_root=frontier_root, scratch=scratch,
                           max_turns=max_turns,
                           max_tokens_per_turn=max_tokens_per_turn,
                           temperature=temperature, top_p=top_p,
                           gen_timeout=gen_timeout, eval_timeout=eval_timeout)
                 for _ in range(num_rollouts)]
        for f in as_completed(futs):
            try:
                results.append(f.result())
            except Exception as e:
                results.append({"final_score": 0.0, "best_code": "", "turns": [],
                                 "messages": [], "task_meta": task_meta, "_err": repr(e)})
    return results
