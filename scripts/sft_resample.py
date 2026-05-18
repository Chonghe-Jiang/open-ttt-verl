"""
After sft_pretrain.py produces a LoRA adapter, serve it via vLLM and resample 8 rollouts.
Then Docker-eval each rollout. The SFT'd policy SHOULD now produce non-zero rollouts
because it was trained on a 0.88-reward solution.

vLLM 0.12 supports LoRA hot-load via --enable-lora --lora-modules name=path.
"""
import argparse
import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


PROMPT_TEMPLATE = """You are an expert Python engineer. Solve the following problem from the Frontier-CS benchmark.

PROBLEM SPECIFICATION:
{readme}

Write a complete, self-contained Python solution. Output ONLY the Python source code inside a single ```python ... ``` fenced block. No explanations outside the block.
"""


def find_task(tasks_json: Path, name: str) -> dict:
    data = json.loads(tasks_json.read_text())
    for bucket in (data["in_distribution"], data["out_of_distribution"]):
        for t in bucket:
            if t.get("yuchen_name") == name:
                return t
    raise KeyError(name)


def read_readme(frontier_root: Path, problem: str, variant: str | None) -> str:
    candidates = []
    base = frontier_root / "research" / "problems" / problem
    if variant:
        base = base / variant
    candidates.append(base)
    if base.is_dir():
        for child in sorted(base.iterdir()):
            if child.is_dir() and child.name not in ("resources", "common", "__pycache__"):
                candidates.append(child)
    for c in candidates:
        for f in ("readme", "README.md", "README"):
            if (c / f).exists():
                return (c / f).read_text()
    raise FileNotFoundError(base)


def extract_python_block(text: str) -> str:
    m = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).rstrip() + "\n"
    m = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).rstrip() + "\n"
    return text.rstrip() + "\n"


def vllm_complete(server: str, model: str, prompt: str, max_tokens: int, timeout: int,
                  temperature: float = 1.0, top_p: float = 0.95) -> str:
    payload = {
        "model": model, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": temperature, "top_p": top_p, "stream": False,
    }
    r = requests.post(f"{server}/v1/completions", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["text"]


def evaluate_with_frontier(frontier_root: Path, problem: str, variant: str | None, code: str, scratch: Path) -> tuple[float, dict]:
    sol = scratch / f"sol_{int(time.time()*1e6)}.py"
    sol.write_text(code)
    pid = f"{problem}/{variant}" if variant else problem
    cmd = ["frontier", "eval", "research", pid, str(sol), "--backend", "docker"]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        out = proc.stdout
        m = re.search(r"Score:\s*([\-\d\.eE]+)", out)
        score = float(m.group(1)) if m else 0.0
        if score < 0:
            score = 0.0
        return score / 100.0, {"score_100": score, "stdout_tail": out[-500:], "rc": proc.returncode}
    except subprocess.TimeoutExpired:
        return 0.0, {"score_100": 0.0, "stdout_tail": "<timeout>", "rc": -1}
    finally:
        try:
            sol.unlink()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--tasks-json", default="/fsx/xuanj/ttt-discover/bench/tasks_19.json")
    ap.add_argument("--frontier-root", default="/fsx/xuanj/ttt-discover/src/frontier-cs")
    ap.add_argument("--vllm-url", default="http://127.0.0.1:8000")
    ap.add_argument("--lora-name", required=True, help="LoRA module name registered with vLLM (e.g. 'sft-cbl_multi__low')")
    ap.add_argument("--num-rollouts", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--gen-timeout", type=int, default=1800)
    ap.add_argument("--readme-max-chars", type=int, default=8000)
    ap.add_argument("--eval-concurrency", type=int, default=8)
    ap.add_argument("--scratch-dir", default="/fsx/xuanj/ttt-discover/scratch")
    ap.add_argument("--output-dir", default="/fsx/xuanj/ttt-discover/results/sft_eval")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="Set 0.0 for greedy. SFT-memorized solutions reproduce best at low temperature.")
    ap.add_argument("--top-p", type=float, default=0.95)
    args = ap.parse_args()

    task_meta = find_task(Path(args.tasks_json), args.task)
    problem = task_meta["problem"]
    variant = task_meta.get("variant")
    readme = read_readme(Path(args.frontier_root), problem, variant)
    prompt = PROMPT_TEMPLATE.format(readme=readme[: args.readme_max_chars])

    scratch = Path(args.scratch_dir) / args.task / "sft"
    scratch.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.output_dir) / args.task
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[sft-resample] task={args.task} lora={args.lora_name}", flush=True)
    t0 = time.time()
    completions = []
    with ThreadPoolExecutor(max_workers=args.num_rollouts) as ex:
        futures = [ex.submit(vllm_complete, args.vllm_url, args.lora_name, prompt, args.max_tokens, args.gen_timeout,
                              args.temperature, args.top_p)
                   for _ in range(args.num_rollouts)]
        for fut in as_completed(futures):
            try:
                completions.append(fut.result())
            except Exception as e:
                print(f"  gen error: {e}", flush=True)
                completions.append("")
    t_gen = time.time() - t0
    print(f"[sft-resample] sampled {len(completions)} rollouts in {t_gen:.1f}s", flush=True)

    codes = [extract_python_block(c) for c in completions]
    # Persist generated codes for debugging
    debug_dir = out_dir / "debug_generated"
    debug_dir.mkdir(parents=True, exist_ok=True)
    for i, (raw, code) in enumerate(zip(completions, codes)):
        (debug_dir / f"r{i}.completion.txt").write_text(raw)
        (debug_dir / f"r{i}.code.py").write_text(code)
    print(f"[sft-resample] saved {len(codes)} debug codes to {debug_dir}", flush=True)
    rewards = [0.0] * len(codes)
    metas = [None] * len(codes)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.eval_concurrency) as ex:
        futures = {ex.submit(evaluate_with_frontier, Path(args.frontier_root), problem, variant, codes[i], scratch): i
                   for i in range(len(codes)) if codes[i].strip()}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                rewards[i], metas[i] = fut.result()
            except Exception as e:
                rewards[i], metas[i] = 0.0, {"error": repr(e)}
    t_eval = time.time() - t0
    avg_r = sum(rewards) / max(1, len(rewards))
    max_r = max(rewards) if rewards else 0
    print(f"[sft-resample] avg={avg_r:.4f} max={max_r:.4f} eval_time={t_eval:.0f}s", flush=True)
    print(f"[sft-resample] rewards: {rewards}", flush=True)

    for i, r in enumerate(rewards):
        (out_dir / f"r{i}.eval.json").write_text(json.dumps({
            "task": args.task, "rollout": i, "reward_01": r, "score_100": r * 100,
            "score_clipped_100": max(0, r * 100), "status": "ok", "rc": (metas[i] or {}).get("rc", 0),
            "elapsed_s": 0, "stdout_tail": (metas[i] or {}).get("stdout_tail", ""),
            "stderr_tail": "", "solution_path": "",
        }, indent=2))
    (out_dir / "summary.json").write_text(json.dumps({
        "task": args.task, "rewards": rewards, "avg": avg_r, "max": max_r,
        "wall_s": time.time() - t0, "lora": args.lora_name,
    }, indent=2))


if __name__ == "__main__":
    main()
