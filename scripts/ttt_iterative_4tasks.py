"""
TTT-Discover with iterative tool-calling rollout, 4-task training pool.

Aligns with Yuchen STaR (kelp PR #11) setup:
- Train pool: 4 tasks selected via random.Random(seed=1).sample(yuchen_pool_66, 4)
  (Yuchen's iterations=1, batch_size=4, seed=iter_id=1)
- Each TTT step samples 4 tasks (= all 4 train_tasks since group_size//N_tasks=2),
  with group_size//4 = 2 rollouts per task per step.
- 16 steps × 8 rollouts/step = 128 total rollouts ↔ Yuchen's 4 tasks × 32 rollouts = 128.

Each step:
  for task in 4 train_tasks: sample 2 rollouts using current LoRA
  combine 8 rollouts → entropic adaptive β advantages → LoRA gradient step
  save step_NNN ckpt + reload into vLLM
"""
from __future__ import annotations
import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from iterative_rollout import (
    do_iterative_rollout, build_initial_user_message, SYSTEM_PROMPT,
)


# ----- Frontier-CS task helpers (from base_eval_iterative) -----

def find_task_in_pool(tasks_json: Path, task_id: str) -> dict:
    """tasks_json: tasks_66.json with {tasks: [{id, problem, variant}, ...]}"""
    data = json.loads(tasks_json.read_text())
    for t in data["tasks"]:
        if t["id"] == task_id:
            return t
    raise KeyError(f"task {task_id} not in {tasks_json}")


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


def read_starter_code(frontier_root: Path, problem: str, variant: str | None) -> str | None:
    bases = []
    base = frontier_root / "research" / "problems" / problem
    if variant:
        base = base / variant
    bases.append(base)
    if base.is_dir():
        for child in sorted(base.iterdir()):
            if child.is_dir() and child.name not in ("resources", "common", "__pycache__"):
                bases.append(child)
    for b in bases:
        progs = b / "resources" / "programs"
        if progs.is_dir():
            for name in ["initial_greedy.py", "initial.py"]:
                p = progs / name
                if p.exists():
                    return p.read_text()
            for p in sorted(progs.glob("*.py")):
                try:
                    return p.read_text()
                except Exception:
                    continue
    return None


# ----- Adaptive β bisection (Appendix A.1) -----

def entropic_adaptive_beta_advantages(rewards: torch.Tensor, target_kl_nats: float = math.log(2),
                                       lo: float = 1e-3, hi: float = 1e3, max_iter: int = 50,
                                       tol: float = 1e-4):
    """Solve for β such that KL(softmax(β*r) || uniform) = target_kl_nats."""
    N = rewards.numel()
    if N == 0:
        return torch.zeros_like(rewards), 0.0
    r = rewards.float()
    r_centered = r - r.mean()
    if r_centered.abs().max() < 1e-8:
        return torch.zeros_like(r), 0.0
    log_N = math.log(N)
    def kl_at(beta):
        z = beta * r_centered
        z = z - z.max()
        p = F.softmax(z, dim=0)
        h = -(p * torch.log(p + 1e-30)).sum().item()
        return log_N - h
    # bisection
    a, b = lo, hi
    for _ in range(max_iter):
        m = (a + b) / 2
        if kl_at(m) < target_kl_nats:
            a = m
        else:
            b = m
        if b - a < tol:
            break
    beta = (a + b) / 2
    z = beta * r_centered
    z = z - z.max()
    p = F.softmax(z, dim=0)
    advantages = (p * N - 1.0).detach()  # advantage relative to uniform
    return advantages, beta


# ----- LoRA trainer -----

class LoRATrainer:
    def __init__(self, model_name: str, lora_rank: int, lr: float, device: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/fsx/xuanj/ttt-discover/.hf-cache")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, attn_implementation="eager",
            cache_dir="/fsx/xuanj/ttt-discover/.hf-cache",
        ).to(device)
        self.base_model.gradient_checkpointing_enable()
        self.base_model.eval()
        for p in self.base_model.parameters():
            p.requires_grad = False
        cfg = LoraConfig(
            r=lora_rank, lora_alpha=lora_rank * 2, lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.base_model, cfg)
        self.model.train()
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable, lr=lr, betas=(0.9, 0.95), eps=1e-8)
        self.device = device

    def step(self, prompt: str, completions: list[str], advantages: torch.Tensor,
             kl_coef: float = 0.1) -> dict:
        if not completions:
            return {"loss": 0.0, "skipped": True}
        prompt_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        prompt_len = prompt_ids.shape[1]
        total_loss = torch.zeros((), device=self.device)
        avg_logp_diff = torch.zeros((), device=self.device)
        n_used = 0
        self.optimizer.zero_grad()
        for i, comp in enumerate(completions):
            comp_ids = self.tokenizer(comp, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
            full = torch.cat([prompt_ids, comp_ids], dim=1)
            with torch.no_grad():
                # Reference logprobs from base (LoRA off via disable_adapter)
                with self.model.disable_adapter():
                    ref_out = self.model(full, return_dict=True)
                    ref_logits = ref_out.logits[:, prompt_len-1:-1, :]
                    ref_lp = F.log_softmax(ref_logits.float(), dim=-1)
                    ref_token_lp = ref_lp.gather(2, comp_ids.unsqueeze(-1)).squeeze(-1)
            out = self.model(full, return_dict=True)
            logits = out.logits[:, prompt_len-1:-1, :]
            lp = F.log_softmax(logits.float(), dim=-1)
            token_lp = lp.gather(2, comp_ids.unsqueeze(-1)).squeeze(-1)
            mean_lp = token_lp.mean()
            mean_ref_lp = ref_token_lp.mean()
            adv_i = advantages[i].to(self.device)
            kl = (mean_lp - mean_ref_lp)
            loss_i = -(adv_i * mean_lp) + kl_coef * kl ** 2
            total_loss = total_loss + loss_i
            avg_logp_diff = avg_logp_diff + kl
            n_used += 1
        loss = total_loss / max(1, n_used)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], max_norm=1.0)
        self.optimizer.step()
        return {"loss": float(loss.item()), "avg_logp_diff": float(avg_logp_diff.item() / max(1, n_used))}

    def save_lora(self, path: str):
        self.model.save_pretrained(path)


# ----- vLLM helpers -----

def vllm_unload_lora(server: str, name: str, timeout: int = 30):
    try:
        r = requests.post(f"{server}/v1/unload_lora_adapter", json={"lora_name": name}, timeout=timeout)
        return r.status_code in (200, 404)
    except Exception:
        return False


def vllm_load_lora(server: str, name: str, path: str, timeout: int = 120):
    r = requests.post(f"{server}/v1/load_lora_adapter",
                      json={"lora_name": name, "lora_path": path}, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"load_lora_adapter failed: {r.status_code} {r.text[:200]}")


# ----- Main -----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True,
                    help="Comma-separated task IDs (e.g. 'task_a,task_b,task_c,task_d')")
    ap.add_argument("--tasks-json", default="/fsx/xuanj/ttt-discover/bench/tasks_66.json")
    ap.add_argument("--frontier-root", default="/fsx/xuanj/ttt-discover/src/frontier-cs")
    ap.add_argument("--vllm-url", required=True)
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
    ap.add_argument("--lora-name", default="ttt-iter")
    ap.add_argument("--lora-save-dir", default="/fsx/xuanj/ttt-discover/results/ttt_iter_lora")

    ap.add_argument("--num-steps", type=int, default=16)
    ap.add_argument("--group-size", type=int, default=8,
                    help="Total rollouts/step. group_size//len(tasks) rollouts per task.")
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--kl-coef", type=float, default=0.1)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max-tokens-per-turn", type=int, default=16384)
    ap.add_argument("--max-turns", type=int, default=3)
    ap.add_argument("--gen-timeout", type=int, default=1800)
    ap.add_argument("--eval-timeout", type=int, default=1800)
    ap.add_argument("--rollout-concurrency", type=int, default=8)

    ap.add_argument("--scratch-dir", default="/fsx/xuanj/ttt-discover/scratch_iter_train")
    ap.add_argument("--output-dir", default="/fsx/xuanj/ttt-discover/results/ttt_iter")
    ap.add_argument("--run-tag", default="4tasks_yuchen_seed1",
                    help="Subdir name under output-dir, lora-save-dir, scratch-dir")
    args = ap.parse_args()

    train_tasks_ids = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if not train_tasks_ids:
        raise ValueError("--tasks must be non-empty")
    n_tasks = len(train_tasks_ids)
    rollouts_per_task = args.group_size // n_tasks
    if rollouts_per_task * n_tasks != args.group_size:
        raise ValueError(f"group_size ({args.group_size}) must be divisible by len(tasks) ({n_tasks})")

    tasks_json = Path(args.tasks_json)
    frontier_root = Path(args.frontier_root)

    # Resolve task metadata + readme/starter for each
    train_tasks = []
    for tid in train_tasks_ids:
        meta = find_task_in_pool(tasks_json, tid)
        readme = read_readme(frontier_root, meta["problem"], meta.get("variant"))
        starter = read_starter_code(frontier_root, meta["problem"], meta.get("variant"))
        # Compatibility: do_iterative_rollout expects {problem, variant, language}
        meta = {"problem": meta["problem"], "variant": meta.get("variant"),
                "language": "python"}
        train_tasks.append({
            "id": tid, "meta": meta, "readme": readme, "starter": starter,
        })
        print(f"[ttt-iter] task: {tid} starter={'yes' if starter else 'no'}", flush=True)

    scratch = Path(args.scratch_dir) / args.run_tag
    scratch.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.output_dir) / args.run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    lora_dir = Path(args.lora_save_dir) / args.run_tag
    lora_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ttt-iter] tasks={n_tasks} rollouts/task={rollouts_per_task} "
          f"group_size={args.group_size} num_steps={args.num_steps}", flush=True)

    print(f"[ttt-iter] loading model {args.model}", flush=True)
    t_load = time.time()
    trainer = LoRATrainer(args.model, lora_rank=args.lora_rank, lr=args.lr, device="cuda:0")
    print(f"[ttt-iter] model loaded in {time.time()-t_load:.0f}s", flush=True)

    init_path = str(lora_dir / "init")
    trainer.save_lora(init_path)
    vllm_unload_lora(args.vllm_url, args.lora_name)
    vllm_load_lora(args.vllm_url, args.lora_name, init_path)
    print(f"[ttt-iter] vLLM loaded LoRA '{args.lora_name}' from {init_path}", flush=True)

    history = []
    best_overall = {"reward": 0.0, "code": "", "step": -1, "task_id": ""}
    t_start = time.time()

    for step_idx in range(args.num_steps):
        t_step = time.time()
        # Build rollout requests: rollouts_per_task per task
        rollout_specs = []
        for task in train_tasks:
            for _ in range(rollouts_per_task):
                rollout_specs.append(task)

        # Concurrent execution
        with ThreadPoolExecutor(max_workers=args.rollout_concurrency) as ex:
            futs = []
            for task in rollout_specs:
                task_scratch = scratch / task["id"]
                task_scratch.mkdir(parents=True, exist_ok=True)
                fut = ex.submit(do_iterative_rollout,
                                server=args.vllm_url, model=args.lora_name,
                                task_meta=task["meta"], readme=task["readme"],
                                starter_code=task["starter"],
                                frontier_root=frontier_root, scratch=task_scratch,
                                max_turns=args.max_turns,
                                max_tokens_per_turn=args.max_tokens_per_turn,
                                temperature=args.temperature,
                                gen_timeout=args.gen_timeout,
                                eval_timeout=args.eval_timeout)
                futs.append((task, fut))
            rollouts = []
            for task, f in futs:
                try:
                    r = f.result()
                except Exception as e:
                    r = {"final_score": 0.0, "best_code": "", "turns": [], "messages": [], "_err": repr(e)}
                r["_task_id"] = task["id"]
                r["_task_meta"] = task["meta"]
                r["_readme"] = task["readme"]
                r["_starter"] = task["starter"]
                rollouts.append(r)

        rewards = [r["final_score"] for r in rollouts]
        codes = [r["best_code"] for r in rollouts]

        r_t = torch.tensor(rewards, dtype=torch.float32)
        advantages, beta_used = entropic_adaptive_beta_advantages(r_t)

        # Gradient: build per-task pseudo prompt; use best turn's text as completion.
        # If best_code is empty (e.g. R1 truncated before code block), still use
        # the first turn's text so we get gradient signal even from "exploration"
        # rollouts that happened to score 0.
        completions_per_idx = []
        prompts_per_idx = []
        for i, r in enumerate(rollouts):
            if r["turns"]:
                if r.get("best_code"):
                    best_turn = max(r["turns"], key=lambda t: t.get("score", 0))
                    text = best_turn.get("text", "")
                else:
                    # use the longest turn we have (most informative)
                    candidates = [t for t in r["turns"] if t.get("text")]
                    text = max(candidates, key=lambda t: len(t.get("text", "")))["text"] if candidates else ""
                if text:
                    completions_per_idx.append(text)
                    prompts_per_idx.append(
                        f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n"
                        f"{build_initial_user_message(r['_task_meta'], r['_readme'], r['_starter'])}\n<|assistant|>\n"
                    )
                    continue
            completions_per_idx.append(None)
            prompts_per_idx.append(None)

        valid_idx = [i for i, c in enumerate(completions_per_idx) if c]
        grad_metrics = {"loss": 0.0, "skipped": True,
                         "n_valid": len(valid_idx),
                         "r_max": float(r_t[valid_idx].max()) if valid_idx else 0.0,
                         "r_min": float(r_t[valid_idx].min()) if valid_idx else 0.0}
        if len(valid_idx) < 2:
            grad_metrics["skip_reason"] = f"n_valid={len(valid_idx)}<2"
        elif r_t[valid_idx].max() <= r_t[valid_idx].min():
            grad_metrics["skip_reason"] = f"max==min={float(r_t[valid_idx].max()):.4f}"
        else:
            try:
                trainer.optimizer.zero_grad()
                total_loss = 0.0
                kl_sum = 0.0
                for i in valid_idx:
                    sub_loss = trainer._step_one(prompts_per_idx[i], completions_per_idx[i],
                                                  advantages[i], kl_coef=args.kl_coef)
                    total_loss += sub_loss["loss"]
                    kl_sum += sub_loss["kl"]
                torch.nn.utils.clip_grad_norm_(
                    [p for p in trainer.model.parameters() if p.requires_grad], max_norm=1.0)
                trainer.optimizer.step()
                grad_metrics = {"loss": total_loss / len(valid_idx), "skipped": False,
                                 "kl_avg": kl_sum / len(valid_idx),
                                 "n_valid": len(valid_idx)}
            except torch.cuda.OutOfMemoryError:
                grad_metrics = {"loss": 0.0, "skipped": True, "oom": True,
                                 "skip_reason": "OOM"}
                torch.cuda.empty_cache()
            except Exception as e:
                grad_metrics = {"loss": 0.0, "skipped": True,
                                 "skip_reason": f"exc:{type(e).__name__}:{str(e)[:120]}"}
                torch.cuda.empty_cache()

        # Save LoRA + reload into vLLM
        ckpt_path = str(lora_dir / f"step_{step_idx:03d}")
        trainer.save_lora(ckpt_path)
        vllm_unload_lora(args.vllm_url, args.lora_name)
        vllm_load_lora(args.vllm_url, args.lora_name, ckpt_path)

        avg_r = float(np.mean(rewards))
        max_r = float(np.max(rewards))
        # Per-task breakdown
        per_task = {}
        for r in rollouts:
            per_task.setdefault(r["_task_id"], []).append(r["final_score"])
        for r, code in zip(rollouts, codes):
            if r["final_score"] > best_overall["reward"]:
                best_overall = {"reward": r["final_score"], "code": code,
                                 "step": step_idx, "task_id": r["_task_id"]}

        elapsed = time.time() - t_start
        per_task_str = " | ".join(f"{k.split('__')[0][:14]}={sum(v)/len(v):.2f}/{max(v):.2f}"
                                   for k, v in per_task.items())
        print(f"[ttt-iter] step {step_idx+1}/{args.num_steps} | "
              f"avg={avg_r:.4f} max={max_r:.4f} | best_so_far={best_overall['reward']:.4f}@{best_overall['task_id'][:18]} | "
              f"loss={grad_metrics.get('loss', 0):.4f} skipped={grad_metrics.get('skipped', True)} "
              f"skip_reason={grad_metrics.get('skip_reason', '-')} n_valid={grad_metrics.get('n_valid', 0)} | "
              f"step_time={time.time()-t_step:.0f}s elapsed={elapsed:.0f}s | "
              f"per_task: {per_task_str}", flush=True)
        history.append({
            "step": step_idx, "rewards": rewards, "avg": avg_r, "max": max_r,
            "best_so_far": best_overall["reward"], "beta": beta_used,
            "grad": grad_metrics, "elapsed_s": elapsed,
            "per_task": per_task,
            "task_per_rollout": [r["_task_id"] for r in rollouts],
        })
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
        (out_dir / "best_solution.py").write_text(best_overall["code"])

    # Final LoRA
    final_lora = str(lora_dir / "final")
    trainer.save_lora(final_lora)
    print(f"[ttt-iter] DONE. final LoRA at {final_lora}, "
          f"best_reward={best_overall['reward']:.4f} at step={best_overall['step']} "
          f"task={best_overall['task_id']}", flush=True)


def _step_one(self, prompt: str, completion: str, advantage: torch.Tensor, kl_coef: float,
              max_completion_tokens: int = 4096, max_prompt_tokens: int = 8192):
    """Single prompt+completion gradient accumulation step (no zero_grad/optimizer.step)."""
    prompt_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
    if prompt_ids.shape[1] > max_prompt_tokens:
        # left-truncate prompt to fit
        prompt_ids = prompt_ids[:, -max_prompt_tokens:]
    prompt_len = prompt_ids.shape[1]
    comp_ids = self.tokenizer(completion, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
    if comp_ids.shape[1] > max_completion_tokens:
        # take the last N tokens (most useful for code generation tail)
        comp_ids = comp_ids[:, -max_completion_tokens:]
    full = torch.cat([prompt_ids, comp_ids], dim=1)
    with torch.no_grad():
        with self.model.disable_adapter():
            ref_out = self.model(full, return_dict=True)
            ref_logits = ref_out.logits[:, prompt_len-1:-1, :]
            ref_lp = F.log_softmax(ref_logits.float(), dim=-1)
            ref_token_lp = ref_lp.gather(2, comp_ids.unsqueeze(-1)).squeeze(-1)
    out = self.model(full, return_dict=True)
    logits = out.logits[:, prompt_len-1:-1, :]
    lp = F.log_softmax(logits.float(), dim=-1)
    token_lp = lp.gather(2, comp_ids.unsqueeze(-1)).squeeze(-1)
    mean_lp = token_lp.mean()
    mean_ref_lp = ref_token_lp.mean()
    adv = advantage.to(self.device)
    kl = (mean_lp - mean_ref_lp)
    loss_i = -(adv * mean_lp) + kl_coef * kl ** 2
    loss_i.backward()
    return {"loss": float(loss_i.item()), "kl": float(kl.item())}


# Bind helper as method
LoRATrainer._step_one = _step_one


if __name__ == "__main__":
    main()
