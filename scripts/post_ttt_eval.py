"""
Post-TTT eval: same iterative protocol as base_eval but with LoRA loaded.
Eval all 17 tasks in tasks_19.json (regardless of whether they're in train pool —
report will distinguish).
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from iterative_rollout import do_iterative_rollout
from base_eval_iterative import find_task, read_readme, read_starter_code, all_task_names


def run_one_rollout(*, task, rollout_idx, server, model, tasks_json, frontier_root,
                    scratch_root, output_dir, max_turns, max_tokens, temperature,
                    gen_timeout, eval_timeout):
    task_meta = find_task(tasks_json, task)
    problem = task_meta["problem"]
    variant = task_meta.get("variant")
    readme = read_readme(frontier_root, problem, variant)
    starter = read_starter_code(frontier_root, problem, variant)
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
    (output_dir / f"{task}.r{rollout_idx}.eval.json").write_text(json.dumps(payload, indent=2))
    full = {
        "task": task, "rollout": rollout_idx,
        "final_score": result["final_score"], "best_code": result["best_code"],
        "turns": [{k: v for k, v in t.items() if k != "token_logprobs"}
                  for t in result["turns"]],
        "wall_s": wall,
    }
    out_full_dir = output_dir / task
    out_full_dir.mkdir(parents=True, exist_ok=True)
    (out_full_dir / f"r{rollout_idx}.full.json").write_text(json.dumps(full, indent=2))
    print(f"[post-ttt] {task} r{rollout_idx} score={result['final_score']:.4f} wall={wall:.0f}s", flush=True)
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--model", required=True, help="LoRA name in vLLM")
    ap.add_argument("--tasks-json", default="/fsx/xuanj/ttt-discover/bench/tasks_19.json")
    ap.add_argument("--frontier-root", default="/fsx/xuanj/ttt-discover/src/frontier-cs")
    ap.add_argument("--scratch-root", default="/fsx/xuanj/ttt-discover/scratch_post_ttt")
    ap.add_argument("--output-dir", default="/fsx/xuanj/ttt-discover/results/ttt_iter_post_eval/eval")
    ap.add_argument("--num-rollouts", type=int, default=8)
    ap.add_argument("--max-turns", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--gen-timeout", type=int, default=1800)
    ap.add_argument("--eval-timeout", type=int, default=1800)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--skip-if-exists", action="store_true")
    args = ap.parse_args()

    tasks_json = Path(args.tasks_json)
    frontier_root = Path(args.frontier_root)
    scratch_root = Path(args.scratch_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_tasks = all_task_names(tasks_json)
    all_jobs = [(t, i) for t in all_tasks for i in range(args.num_rollouts)]
    jobs = [(t, i) for idx, (t, i) in enumerate(all_jobs)
            if idx % args.num_shards == args.shard_id]
    if args.skip_if_exists:
        kept = [(t, i) for (t, i) in jobs if not (output_dir / f"{t}.r{i}.eval.json").exists()]
        print(f"[post-ttt] skip-if-exists: {len(jobs)-len(kept)} already done, {len(kept)} to run", flush=True)
        jobs = kept
    print(f"[post-ttt] shard {args.shard_id}/{args.num_shards}: "
          f"{len(jobs)} of {len(all_jobs)} on {len(all_tasks)} tasks", flush=True)

    t_start = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(run_one_rollout,
                          task=t, rollout_idx=i, server=args.server, model=args.model,
                          tasks_json=tasks_json, frontier_root=frontier_root,
                          scratch_root=scratch_root, output_dir=output_dir,
                          max_turns=args.max_turns, max_tokens=args.max_tokens,
                          temperature=args.temperature, gen_timeout=args.gen_timeout,
                          eval_timeout=args.eval_timeout): (t, i)
                for t, i in jobs}
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as e:
                print(f"  rollout {futs[f]} failed: {e}", flush=True)

    print(f"[post-ttt] shard done in {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
