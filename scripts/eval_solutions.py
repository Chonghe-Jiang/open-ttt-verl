"""
Run Frontier-CS Docker eval on each generated solution. Persist per-rollout reward
to /fsx/xuanj/ttt-discover/results/<run_tag>/eval/.

Concurrency note: each Docker eval may spin up a heavy container with full deps.
We default to 4 in flight to avoid disk/CPU contention on a single node.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def parse_score(stdout: str) -> tuple[float, str]:
    # Frontier eval prints "Score: <number>" near the end.
    m = re.search(r"^\s*Score:\s*([\-\d\.eE]+)\s*$", stdout, re.MULTILINE)
    if not m:
        m = re.search(r"Score:\s*([\-\d\.eE]+)", stdout)
    if m:
        return float(m.group(1)), "ok"
    return 0.0, "no_score_match"


def run_eval(args, task: dict, rollout_idx: int) -> dict:
    pid = f"{task['problem']}/{task['variant']}" if task.get("variant") else task["problem"]
    sol_dir = Path(args.frontier_root) / "research" / "solutions" / task["problem"]
    if task.get("variant"):
        sol_dir = sol_dir / task["variant"]
    sol_path = sol_dir / f"{args.model_tag}_{rollout_idx}.py"
    out_path = Path(args.results_dir) / f"{task['yuchen_name']}.r{rollout_idx}.eval.json"

    if not sol_path.exists():
        return {"task": task["yuchen_name"], "rollout": rollout_idx, "status": "no_solution"}
    if out_path.exists() and not args.overwrite:
        existing = json.loads(out_path.read_text())
        return {"task": task["yuchen_name"], "rollout": rollout_idx, "status": "cached", "score": existing.get("score_100")}

    cmd = ["frontier", "eval", "research", pid, str(sol_path), "--backend", "docker"]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
        elapsed = time.time() - t0
        score_100, parse_status = parse_score(p.stdout)
        # Docker eval can return -100000 as penalty for missing deadline; clip to 0 for reward.
        clipped = max(0.0, score_100)
        out = {
            "task": task["yuchen_name"], "rollout": rollout_idx,
            "score_100": score_100, "score_clipped_100": clipped,
            "reward_01": clipped / 100.0,
            "status": parse_status, "rc": p.returncode, "elapsed_s": elapsed,
            "stdout_tail": p.stdout[-2000:], "stderr_tail": p.stderr[-1000:],
            "solution_path": str(sol_path),
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))
        return {"task": task["yuchen_name"], "rollout": rollout_idx, "status": "ok",
                "score_100": score_100, "elapsed_s": elapsed}
    except subprocess.TimeoutExpired as e:
        out = {"task": task["yuchen_name"], "rollout": rollout_idx,
               "score_100": 0.0, "score_clipped_100": 0.0, "reward_01": 0.0,
               "status": "timeout", "elapsed_s": time.time() - t0}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))
        return {"task": task["yuchen_name"], "rollout": rollout_idx, "status": "timeout", "elapsed_s": time.time() - t0}
    except Exception as e:
        out = {"task": task["yuchen_name"], "rollout": rollout_idx,
               "score_100": 0.0, "score_clipped_100": 0.0, "reward_01": 0.0,
               "status": "exception", "error": repr(e), "elapsed_s": time.time() - t0}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))
        return {"task": task["yuchen_name"], "rollout": rollout_idx, "status": "exception", "error": repr(e)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frontier-root", default="/fsx/xuanj/ttt-discover/src/frontier-cs")
    p.add_argument("--tasks-json", default="/fsx/xuanj/ttt-discover/bench/tasks_19.json")
    p.add_argument("--model-tag", default="dsr1q3_8b_base")
    p.add_argument("--num-rollouts", type=int, default=8)
    p.add_argument("--results-dir", default="/fsx/xuanj/ttt-discover/results/base/eval")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--timeout", type=int, default=2400)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--shard-id", type=int, default=0, help="0-indexed shard")
    p.add_argument("--num-shards", type=int, default=1, help="Total shards; this process handles jobs where (idx % num_shards) == shard_id")
    args = p.parse_args()

    data = json.loads(Path(args.tasks_json).read_text())
    tasks = []
    for bucket in (data["in_distribution"], data["out_of_distribution"]):
        for t in bucket:
            if t.get("yuchen_name", "").startswith("_"):
                continue
            tasks.append(t)

    all_jobs = [(t, i) for t in tasks for i in range(args.num_rollouts)]
    # Shard by index: shard k handles jobs where idx % num_shards == k
    jobs = [job for idx, job in enumerate(all_jobs) if (idx % args.num_shards) == args.shard_id]
    print(f"[eval] shard {args.shard_id}/{args.num_shards}: {len(jobs)}/{len(all_jobs)} evaluations, concurrency={args.concurrency}, timeout={args.timeout}s", flush=True)

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {ex.submit(run_eval, args, t, i): (t["yuchen_name"], i) for (t, i) in jobs}
        for j, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            results.append(r)
            elapsed = time.time() - t_start
            sc = r.get("score_100")
            print(f"[eval {j}/{len(jobs)}] [{elapsed:7.1f}s] {r['task']} r{r['rollout']} -> {r['status']} score={sc}", flush=True)

    summary_path = Path(args.results_dir) / "_summary.json"
    summary_path.write_text(json.dumps({
        "wall_time_s": time.time() - t_start,
        "num_jobs": len(jobs),
        "results": results,
    }, indent=2))
    print(f"[eval] done in {time.time() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
