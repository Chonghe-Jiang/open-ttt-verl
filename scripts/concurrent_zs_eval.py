"""
Concurrent zero-shot eval daemon. Watches the trainer's LoRA ckpt directory,
and as each new step_N/ ckpt appears, runs a quick eval pass on the OOD tasks.
After training finishes (sentinel `.training_done` appears), it does a full
final eval pass with N rollouts/task using the latest ckpt.

Strategy:
- Per-step: 1 rollout × 17 tasks = 17 quick rollouts to get a noisy reading on
  whether LoRA is improving. Saves to step_eval/step_NNN/.
- Final: --final-rollouts (default 8) rollouts × 17 tasks. Saves to eval/ for
  the report generator.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from iterative_rollout import do_iterative_rollout
from base_eval_iterative import find_task, read_readme, read_starter_code, all_task_names


def vllm_unload_lora(server, name, timeout=30):
    try:
        r = requests.post(f"{server}/v1/unload_lora_adapter",
                          json={"lora_name": name}, timeout=timeout)
        return r.status_code in (200, 404)
    except Exception as e:
        print(f"[zs-d] unload_lora err: {e}", flush=True)
        return False


def vllm_load_lora(server, name, path, timeout=120):
    try:
        r = requests.post(f"{server}/v1/load_lora_adapter",
                          json={"lora_name": name, "lora_path": path}, timeout=timeout)
        if r.status_code == 200:
            return True
        print(f"[zs-d] load_lora failed: {r.status_code} {r.text[:200]}", flush=True)
        return False
    except Exception as e:
        print(f"[zs-d] load_lora err: {e}", flush=True)
        return False


def eval_one_rollout(server, model, tasks_json, frontier_root, scratch_root,
                     output_dir, task, rollout_idx, max_turns, max_tokens,
                     temperature, gen_timeout, eval_timeout):
    task_meta = find_task(tasks_json, task)
    readme = read_readme(frontier_root, task_meta["problem"], task_meta.get("variant"))
    starter = read_starter_code(frontier_root, task_meta["problem"], task_meta.get("variant"))
    scratch = scratch_root / task / f"r{rollout_idx}"
    scratch.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    result = do_iterative_rollout(
        server=server, model=model, task_meta=task_meta,
        readme=readme, starter_code=starter,
        frontier_root=frontier_root, scratch=scratch,
        max_turns=max_turns, max_tokens_per_turn=max_tokens,
        temperature=temperature, gen_timeout=gen_timeout, eval_timeout=eval_timeout,
    )
    wall = time.time() - t0
    payload = {
        "task": task, "rollout": rollout_idx,
        "score_100": result["final_score"] * 100.0,
        "score_clipped_100": max(0, result["final_score"] * 100.0),
        "reward_01": result["final_score"],
        "status": "ok", "rc": 0, "elapsed_s": wall,
        "num_turns_used": len(result["turns"]),
        "stdout_tail": "", "stderr_tail": "", "solution_path": "",
        "best_code_chars": len(result["best_code"]),
    }
    return task, rollout_idx, payload, result["final_score"], wall


def run_eval_pass(server, model, tasks, num_rollouts, output_dir, full_dump_dir,
                  scratch_root, frontier_root, tasks_json,
                  max_turns, max_tokens, temperature, gen_timeout, eval_timeout,
                  concurrency=4, label="step"):
    output_dir.mkdir(parents=True, exist_ok=True)
    if full_dump_dir:
        full_dump_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(t, i) for t in tasks for i in range(num_rollouts)]
    print(f"[zs-d] {label}: {len(jobs)} rollouts on {len(tasks)} tasks", flush=True)
    t_pass = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(eval_one_rollout,
                          server, model, tasks_json, frontier_root, scratch_root,
                          output_dir, t, i, max_turns, max_tokens, temperature,
                          gen_timeout, eval_timeout): (t, i)
                for t, i in jobs}
        for f in as_completed(futs):
            try:
                task, ri, payload, score, wall = f.result()
                # Write eval JSON to output dir (will be overwritten by later passes)
                (output_dir / f"{task}.r{ri}.eval.json").write_text(json.dumps(payload, indent=2))
                if full_dump_dir:
                    (full_dump_dir / f"{task}.r{ri}.json").write_text(json.dumps(payload, indent=2))
                print(f"[zs-d] {label} {task} r{ri} score={score:.4f} wall={wall:.0f}s", flush=True)
            except Exception as e:
                print(f"[zs-d] {label} rollout {futs[f]} failed: {e}", flush=True)
    print(f"[zs-d] {label} pass done in {time.time()-t_pass:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-url", required=True)
    ap.add_argument("--lora-name", required=True)
    ap.add_argument("--lora-base-dir", required=True,
                    help="dir containing step_NNN/ ckpts and final/")
    ap.add_argument("--train-task", required=True,
                    help="task to exclude from zero-shot eval")
    ap.add_argument("--tasks-json", default="/fsx/xuanj/ttt-discover/bench/tasks_19.json")
    ap.add_argument("--frontier-root", default="/fsx/xuanj/ttt-discover/src/frontier-cs")
    ap.add_argument("--scratch-root", default="/fsx/xuanj/ttt-discover/scratch_iter_zs")
    ap.add_argument("--output-dir", default="/fsx/xuanj/ttt-discover/results/ttt_iter_zeroshot/eval")
    ap.add_argument("--step-eval-dir", default="/fsx/xuanj/ttt-discover/results/ttt_iter_zeroshot/step_eval")
    ap.add_argument("--rollouts-per-step", type=int, default=1)
    ap.add_argument("--final-rollouts", type=int, default=8)
    ap.add_argument("--max-turns", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--gen-timeout", type=int, default=1800)
    ap.add_argument("--eval-timeout", type=int, default=1800)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--poll-interval", type=int, default=30)
    args = ap.parse_args()

    tasks_json = Path(args.tasks_json)
    frontier_root = Path(args.frontier_root)
    scratch_root = Path(args.scratch_root)
    output_dir = Path(args.output_dir)
    step_eval_dir = Path(args.step_eval_dir)
    lora_base = Path(args.lora_base_dir)
    sentinel = lora_base / ".training_done"

    all_tasks = [t for t in all_task_names(tasks_json) if t != args.train_task]
    print(f"[zs-d] eval pool: {len(all_tasks)} tasks (excluding {args.train_task})", flush=True)

    seen_steps = set()
    last_loaded_step = None

    # Wait for at least one step ckpt
    print(f"[zs-d] waiting for first ckpt in {lora_base}", flush=True)
    while True:
        steps = sorted(p for p in lora_base.glob("step_*") if p.is_dir())
        if steps:
            break
        if sentinel.exists():
            print("[zs-d] sentinel present before any ckpt, abort", flush=True)
            return
        time.sleep(args.poll_interval)

    # Loop: poll for new steps, run quick eval; on sentinel, run final
    while True:
        # Discover new steps
        steps = sorted(p for p in lora_base.glob("step_*") if p.is_dir())
        new_steps = [s for s in steps if s.name not in seen_steps]
        for step_path in new_steps:
            step_name = step_path.name
            seen_steps.add(step_name)
            print(f"[zs-d] new ckpt: {step_path}", flush=True)
            # Reload LoRA
            vllm_unload_lora(args.vllm_url, args.lora_name)
            ok = vllm_load_lora(args.vllm_url, args.lora_name, str(step_path))
            if not ok:
                print(f"[zs-d] failed to load {step_path}, skipping", flush=True)
                continue
            last_loaded_step = step_path
            # Quick eval
            step_out = step_eval_dir / step_name
            run_eval_pass(args.vllm_url, args.lora_name, all_tasks,
                          args.rollouts_per_step, step_out, None, scratch_root,
                          frontier_root, tasks_json,
                          args.max_turns, args.max_tokens, args.temperature,
                          args.gen_timeout, args.eval_timeout,
                          concurrency=args.concurrency,
                          label=f"step={step_name}")

        if sentinel.exists():
            # Final pass
            print("[zs-d] sentinel detected, running final eval", flush=True)
            final_dir = lora_base / "final"
            target = final_dir if final_dir.is_dir() else last_loaded_step
            if target is None:
                print("[zs-d] no LoRA available for final, abort", flush=True)
                return
            print(f"[zs-d] final eval using LoRA: {target}", flush=True)
            vllm_unload_lora(args.vllm_url, args.lora_name)
            ok = vllm_load_lora(args.vllm_url, args.lora_name, str(target))
            if not ok:
                print(f"[zs-d] failed to load {target} for final, abort", flush=True)
                return
            run_eval_pass(args.vllm_url, args.lora_name, all_tasks,
                          args.final_rollouts, output_dir, None, scratch_root,
                          frontier_root, tasks_json,
                          args.max_turns, args.max_tokens, args.temperature,
                          args.gen_timeout, args.eval_timeout,
                          concurrency=args.concurrency, label="final")
            print("[zs-d] final eval done, exiting", flush=True)
            return

        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
