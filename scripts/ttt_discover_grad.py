"""
TTT-Discover with LoRA gradient step (REINFORCE-style with entropic advantage).

Faithful to Yuksekgonul et al. 2026 §3 algorithm:
  for step i in 0..N-1:
    1. sample G rollouts a_i ~ pi_theta_i(. | d, s_i, c_i)  via transformers.generate
    2. evaluate r_i = R(a_i)                                via Frontier-CS Docker
    3. compute entropic advantages w_i = e_i / Z_i; adv_i = w_i - 1
    4. take Adam gradient step on LoRA weights:
         loss = -mean_i [ adv_i * mean_token_log_prob(a_i | s_i, c_i) ]

Single H200 holds the 8B model + LoRA training. Eval runs in parallel via Docker.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPT_TEMPLATE = """You are an expert Python engineer. Solve the following problem from the Frontier-CS benchmark.

PROBLEM SPECIFICATION:
{readme}

{context_block}

Write a complete, self-contained Python solution. Output ONLY the Python source code inside a single ```python ... ``` fenced block. No explanations outside the block.
"""

CONTEXT_HEADER = """
KNOWN-GOOD SOLUTIONS (use as reference; you MAY borrow good ideas, but produce a NEW improved solution):
"""


def extract_python_block(text):
    m = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).rstrip() + "\n"
    m = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).rstrip() + "\n"
    return text.rstrip() + "\n"


def find_task(tasks_json, name):
    data = json.loads(Path(tasks_json).read_text())
    for bucket in (data["in_distribution"], data["out_of_distribution"]):
        for t in bucket:
            if t.get("yuchen_name") == name:
                return t
    raise KeyError(name)


def read_readme(frontier_root, problem, variant):
    candidates = []
    base = Path(frontier_root) / "research" / "problems" / problem
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


def read_seed(frontier_root, problem, variant):
    bases = []
    base = Path(frontier_root) / "research" / "problems" / problem
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
            for p in sorted(progs.glob("*.py")):
                try:
                    return p.read_text()
                except Exception:
                    continue
    return None


def make_context(buffer, top_k=3):
    if not buffer:
        return ""
    sorted_buf = sorted(buffer, key=lambda r: r["reward"], reverse=True)[:top_k]
    parts = [CONTEXT_HEADER]
    for i, item in enumerate(sorted_buf, 1):
        parts.append(f"\n[Solution {i}, reward={item['reward']:.4f}]\n```python\n{item['code']}\n```\n")
    return "".join(parts)


def evaluate_with_frontier(frontier_root, problem, variant, code, scratch_dir):
    sol = scratch_dir / f"sol_{int(time.time()*1e6)}_{os.getpid()}.py"
    sol.write_text(code)
    pid = f"{problem}/{variant}" if variant else problem
    cmd = ["frontier", "eval", "research", pid, str(sol), "--backend", "docker"]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        out = p.stdout
        m = re.search(r"Score:\s*([\-\d\.eE]+)", out)
        score = float(m.group(1)) if m else 0.0
        if score < 0:
            score = 0.0
        return score / 100.0, time.time() - t0
    except subprocess.TimeoutExpired:
        return 0.0, time.time() - t0
    finally:
        try:
            sol.unlink()
        except Exception:
            pass


def preload_base_rollouts(buffer, base_eval_dir, frontier_root, problem, variant, task, model_tag, top_k=2):
    """Preload buffer with the top-k non-zero base rollouts for this task,
    so step 0 already has a real positive reward to compound on."""
    base_eval_dir = Path(base_eval_dir)
    if not base_eval_dir.exists():
        return
    candidates = []
    for evf in sorted(base_eval_dir.glob(f"{task}.r*.eval.json")):
        try:
            j = json.loads(evf.read_text())
            r = j.get("reward_01", 0)
            if r <= 0:
                continue
            sol_path = Path(j.get("solution_path", ""))
            if not sol_path.exists():
                # Reconstruct path
                sol_dir = Path(frontier_root) / "research" / "solutions" / problem
                if variant:
                    sol_dir = sol_dir / variant
                sol_path = sol_dir / f"{model_tag}_{j['rollout']}.py"
            if sol_path.exists():
                candidates.append((r, sol_path.read_text(), j["rollout"]))
        except Exception:
            continue
    candidates.sort(reverse=True)
    for r, code, ridx in candidates[:top_k]:
        buffer.append({"code": code, "reward": r, "step": -2, "is_base_rollout": True, "rollout": ridx})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--tasks-json", default="/fsx/xuanj/ttt-discover/bench/tasks_19.json")
    ap.add_argument("--frontier-root", default="/fsx/xuanj/ttt-discover/src/frontier-cs")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
    ap.add_argument("--num-steps", type=int, default=6)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--eval-concurrency", type=int, default=4)
    ap.add_argument("--readme-max-chars", type=int, default=6000)
    ap.add_argument("--scratch-dir", default="/fsx/xuanj/ttt-discover/scratch")
    ap.add_argument("--output-dir", default="/fsx/xuanj/ttt-discover/results/ttt_grad")
    ap.add_argument("--base-eval-dir", default="/fsx/xuanj/ttt-discover/results/base/eval")
    ap.add_argument("--base-model-tag", default="dsr1q3_8b_base")
    ap.add_argument("--no-seed", action="store_true")
    ap.add_argument("--no-base-preload", action="store_true")
    args = ap.parse_args()

    task_meta = find_task(args.tasks_json, args.task)
    problem = task_meta["problem"]
    variant = task_meta.get("variant")
    readme = read_readme(args.frontier_root, problem, variant)

    scratch = Path(args.scratch_dir) / args.task / "grad"
    scratch.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.output_dir) / args.task
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ttt-grad] task={args.task} problem={problem} variant={variant}", flush=True)
    print(f"[ttt-grad] loading {args.model}", flush=True)
    t_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda:0",
    )
    model.gradient_checkpointing_enable()
    print(f"[ttt-grad] model loaded in {time.time()-t_load:.0f}s", flush=True)

    lora_config = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
    )

    buffer = []
    if not args.no_seed:
        seed = read_seed(args.frontier_root, problem, variant)
        if seed:
            buffer.append({"code": seed, "reward": 0.05, "step": -1, "is_seed": True})
            print(f"[ttt-grad] seeded buffer with reference solution", flush=True)
    if not args.no_base_preload:
        before = len(buffer)
        preload_base_rollouts(buffer, args.base_eval_dir, args.frontier_root, problem, variant, args.task, args.base_model_tag, top_k=2)
        print(f"[ttt-grad] preloaded {len(buffer)-before} base rollouts (top reward = {max((b['reward'] for b in buffer), default=0):.4f})", flush=True)

    history = []
    t_start = time.time()

    for step in range(args.num_steps):
        ctx = make_context(buffer, top_k=3)
        prompt = PROMPT_TEMPLATE.format(readme=readme[: args.readme_max_chars], context_block=ctx)

        # ===== 1. Sample G rollouts =====
        t_gen_start = time.time()
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        # Generate one rollout at a time so 24K-token outputs fit in H200 KV-cache
        # comfortably; batched gen would OOM at this length.
        model.eval()
        gen_seqs = []
        for _ in range(args.group_size):
            with torch.no_grad():
                out_i = model.generate(
                    input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True, temperature=args.temperature, top_p=args.top_p,
                    pad_token_id=tokenizer.pad_token_id,
                )
            gen_seqs.append(out_i[0])
        # Right-pad to common length so we can stack
        max_len = max(s.shape[0] for s in gen_seqs)
        padded = []
        for s in gen_seqs:
            if s.shape[0] < max_len:
                pad = torch.full((max_len - s.shape[0],), tokenizer.pad_token_id,
                                 dtype=s.dtype, device=s.device)
                s = torch.cat([s, pad], dim=0)
            padded.append(s)
        outputs = torch.stack(padded, dim=0)
        new_token_ids = outputs[:, prompt_len:]
        completions = [tokenizer.decode(seq, skip_special_tokens=True) for seq in new_token_ids]
        codes = [extract_python_block(c) for c in completions]
        t_gen = time.time() - t_gen_start

        # ===== 2. Evaluate rewards in parallel =====
        t_eval_start = time.time()
        rewards = [0.0] * args.group_size
        with ThreadPoolExecutor(max_workers=args.eval_concurrency) as ex:
            futures = {ex.submit(evaluate_with_frontier, args.frontier_root, problem, variant, codes[i], scratch): i
                       for i in range(args.group_size) if codes[i].strip()}
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    rewards[i], _ = fut.result()
                except Exception as e:
                    print(f"  rollout {i} eval error: {e}", flush=True)
        t_eval = time.time() - t_eval_start

        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        avg_r = rewards_t.mean().item()
        max_r = rewards_t.max().item()

        # ===== 3. Entropic advantages =====
        if rewards_t.max() > 0:
            r_max = rewards_t.max()
            e = torch.exp(args.beta * (rewards_t - r_max))
            G = e.shape[0]
            if G == 1:
                Z = e
            else:
                Z = (e.sum() - e) / (G - 1)
            w = e / (Z + 1e-12)
            advantages = w - 1.0
        else:
            advantages = torch.zeros_like(rewards_t)
        advantages = advantages.to(model.device)

        # ===== 4. Gradient step (only if signal) =====
        t_grad_start = time.time()
        loss_val = 0.0
        if (advantages.abs() > 1e-6).any():
            model.train()
            full_seqs = outputs  # [G, total_len]
            attn_full = (full_seqs != tokenizer.pad_token_id).long()
            # Process micro-batch of 1 to fit memory
            optimizer.zero_grad()
            total_loss_t = torch.tensor(0.0, device=model.device)
            num_used = 0
            for i in range(args.group_size):
                if abs(advantages[i].item()) < 1e-6:
                    continue
                seq_i = full_seqs[i:i+1]  # [1, T]
                attn_i = attn_full[i:i+1]
                inp = seq_i[:, :-1]
                tgt = seq_i[:, 1:]
                logits = model(input_ids=inp, attention_mask=attn_i[:, :-1]).logits
                logp = F.log_softmax(logits, dim=-1)
                tgt_lp = logp.gather(2, tgt.unsqueeze(-1)).squeeze(-1)  # [1, T-1]
                # mask: completion tokens only, not pad
                cmask = torch.zeros_like(tgt, dtype=torch.float32)
                cmask[:, prompt_len-1:] = 1.0
                cmask = cmask * attn_i[:, 1:].float()
                ntok = cmask.sum().clamp(min=1.0)
                logp_i = (tgt_lp * cmask).sum() / ntok
                # weighted sum of -advantage * logp
                loss_i = -(advantages[i] * logp_i) / max(1, (advantages.abs() > 1e-6).sum().item())
                loss_i.backward()
                total_loss_t += loss_i.detach()
                num_used += 1
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
            optimizer.step()
            loss_val = total_loss_t.item()
        t_grad = time.time() - t_grad_start

        # ===== 5. Buffer update =====
        for i, code in enumerate(codes):
            if code.strip() and rewards[i] > 0:
                buffer.append({"code": code, "reward": rewards[i], "step": step})
        buffer = sorted(buffer, key=lambda x: x["reward"], reverse=True)[:16]

        elapsed = time.time() - t_start
        print(f"[ttt-grad] step {step+1}/{args.num_steps} | rewards avg={avg_r:.4f} max={max_r:.4f} | "
              f"gen={t_gen:.1f}s eval={t_eval:.1f}s grad={t_grad:.1f}s loss={loss_val:.4f} | "
              f"total={elapsed:.0f}s | buffer top={buffer[0]['reward'] if buffer else 0:.4f}", flush=True)

        history.append({
            "step": step,
            "rewards": rewards,
            "avg_reward": avg_r,
            "max_reward": max_r,
            "loss": loss_val,
            "t_gen_s": t_gen,
            "t_eval_s": t_eval,
            "t_grad_s": t_grad,
            "elapsed_s": elapsed,
            "buffer_top_reward": buffer[0]["reward"] if buffer else 0.0,
        })
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
        if buffer:
            (out_dir / "best_solution.py").write_text(buffer[0]["code"])

    final = {
        "task": args.task, "problem": problem, "variant": variant,
        "wall_s": time.time() - t_start, "num_steps": args.num_steps,
        "group_size": args.group_size, "history": history,
        "best_reward": buffer[0]["reward"] if buffer else 0.0,
        "best_step": buffer[0]["step"] if buffer else None,
        "config": {"lr": args.lr, "lora_rank": args.lora_rank, "beta": args.beta, "max_new_tokens": args.max_new_tokens},
    }
    (out_dir / "final.json").write_text(json.dumps(final, indent=2))
    print(f"[ttt-grad] done {args.task}: best_reward={final['best_reward']:.4f} | wall={final['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
