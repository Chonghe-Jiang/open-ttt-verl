"""
Generate K rollouts of LLM solutions for each of 19 Frontier-CS tasks
using a vLLM server hosting DeepSeek-R1-Distill-Qwen-8B.

Writes solutions to:
    {frontier_cs_root}/research/solutions/{problem}/[{variant}/]{model_tag}_{rollout_idx}.py

Frontier-CS expects file naming convention {model}_{variant_idx}.py for variants.
We use rollout idx as the variant suffix, so the existing batch evaluator
already knows how to score 8 rollouts as Avg@8 / Score@8.

Designed to run on a single p5en GPU node (8x H200) with one vLLM server
serving all rollouts; concurrency = 8 inflight requests.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


PROMPT_TEMPLATE = """You are an expert Python engineer. Solve the following problem from the Frontier-CS benchmark.

PROBLEM SPECIFICATION:
{readme}

Write a complete, self-contained Python solution. Output ONLY the Python source code inside a single ```python ... ``` fenced block. No explanations outside the block. Your code must define the API specified in the problem (typically a `Solution` class). Imports of standard libs and the problem-provided framework are allowed. Do NOT include test code or examples.
"""


def extract_python_block(text: str) -> str:
    """Pull the first ```python ... ``` block, fallback to first ``` ... ``` block, fallback to raw."""
    m = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).rstrip() + "\n"
    m = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).rstrip() + "\n"
    return text.rstrip() + "\n"


def read_problem_readme(frontier_root: Path, problem: str, variant: str | None) -> str:
    """
    Find the task README. Frontier-CS uses one of:
      - research/problems/<problem>/readme
      - research/problems/<problem>/<variant>/readme
      - research/problems/<problem>/<variant>/<subtask_dir>/readme  (e.g. poc_generation has arvo_*/oss_fuzz_* subtasks)
    """
    candidates = []
    base = frontier_root / "research" / "problems" / problem
    if variant:
        base = base / variant
    candidates.append(base)
    # Also descend one level: pick the first subtask dir that has a readme
    if base.is_dir():
        for child in sorted(base.iterdir()):
            if child.is_dir() and child.name not in ("resources", "common", "__pycache__"):
                candidates.append(child)
    for c in candidates:
        for fname in ("readme", "README.md", "README"):
            p = c / fname
            if p.exists():
                return p.read_text()
    raise FileNotFoundError(f"No readme/README in {base} or its subtask dirs")


def call_vllm(server_url: str, model: str, prompt: str, max_tokens: int, temperature: float, top_p: float, timeout: int) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
    }
    t0 = time.time()
    r = requests.post(f"{server_url}/v1/completions", json=payload, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    completion = j["choices"][0]["text"]
    return {
        "completion": completion,
        "latency_s": time.time() - t0,
        "completion_tokens": j.get("usage", {}).get("completion_tokens"),
        "prompt_tokens": j.get("usage", {}).get("prompt_tokens"),
    }


def gen_one(args, task: dict, rollout_idx: int):
    """Generate and persist one rollout for one task."""
    problem = task["problem"]
    variant = task.get("variant")
    yuchen_name = task["yuchen_name"]

    sol_dir = Path(args.frontier_root) / "research" / "solutions" / problem
    if variant:
        sol_dir = sol_dir / variant
    sol_dir.mkdir(parents=True, exist_ok=True)
    sol_path = sol_dir / f"{args.model_tag}_{rollout_idx}.py"
    meta_path = Path(args.results_dir) / f"{yuchen_name}.r{rollout_idx}.meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    if sol_path.exists() and not args.overwrite:
        return {"task": yuchen_name, "rollout": rollout_idx, "status": "skipped", "path": str(sol_path)}

    readme = read_problem_readme(Path(args.frontier_root), problem, variant)
    prompt = PROMPT_TEMPLATE.format(readme=readme[: args.readme_max_chars])

    try:
        result = call_vllm(args.server, args.model, prompt, args.max_tokens, args.temperature, args.top_p, args.timeout)
    except Exception as e:
        return {"task": yuchen_name, "rollout": rollout_idx, "status": "gen_failed", "error": repr(e)}

    code = extract_python_block(result["completion"])
    sol_path.write_text(code)
    meta = {
        "task": yuchen_name,
        "problem": problem,
        "variant": variant,
        "rollout": rollout_idx,
        "model": args.model,
        "model_tag": args.model_tag,
        "solution_path": str(sol_path),
        "latency_s": result["latency_s"],
        "completion_tokens": result["completion_tokens"],
        "prompt_tokens": result["prompt_tokens"],
        "raw_completion_chars": len(result["completion"]),
        "extracted_code_chars": len(code),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return {"task": yuchen_name, "rollout": rollout_idx, "status": "ok", "path": str(sol_path), "latency_s": result["latency_s"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frontier-root", default="/fsx/xuanj/ttt-discover/src/frontier-cs")
    p.add_argument("--tasks-json", default="/fsx/xuanj/ttt-discover/bench/tasks_19.json")
    p.add_argument("--server", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
    p.add_argument("--model-tag", default="dsr1q3_8b_base")
    p.add_argument("--num-rollouts", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--readme-max-chars", type=int, default=8000)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--results-dir", default="/fsx/xuanj/ttt-discover/results/base/meta")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    tasks_data = json.loads(Path(args.tasks_json).read_text())
    tasks = []
    for t in tasks_data["in_distribution"]:
        if t.get("yuchen_name", "").startswith("_"):
            continue
        tasks.append(t)
    for t in tasks_data["out_of_distribution"]:
        if t.get("yuchen_name", "").startswith("_"):
            continue
        tasks.append(t)

    print(f"[gen] {len(tasks)} tasks x {args.num_rollouts} rollouts = {len(tasks)*args.num_rollouts} solutions to generate", flush=True)
    print(f"[gen] model={args.model} server={args.server} concurrency={args.concurrency}", flush=True)

    jobs = [(t, i) for t in tasks for i in range(args.num_rollouts)]
    results = []
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {ex.submit(gen_one, args, t, i): (t, i) for (t, i) in jobs}
        for j, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            results.append(r)
            elapsed = time.time() - t_start
            print(f"[gen {j}/{len(jobs)}] [{elapsed:7.1f}s] {r.get('task')} r={r.get('rollout')} -> {r.get('status')}", flush=True)

    summary_path = Path(args.results_dir) / "_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({
        "wall_time_s": time.time() - t_start,
        "num_tasks": len(tasks),
        "num_rollouts": args.num_rollouts,
        "model": args.model,
        "results": results,
    }, indent=2))
    print(f"[gen] done in {time.time() - t_start:.1f}s; summary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
