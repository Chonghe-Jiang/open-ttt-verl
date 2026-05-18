"""
Supervised fine-tune the base model on the highest-reward base rollout(s) per task,
then resample to see if the SFT'd model produces non-zero rollouts on its own.

This is a much simpler hypothesis test than full TTT-Discover RL: just teach the
model to imitate what worked.

For each task:
  1. Pick top-1 base rollout with reward > threshold (default 0.1)
  2. Build (prompt, solution) pair
  3. SFT a fresh LoRA adapter for N epochs
  4. Save adapter to disk for later vLLM serving
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def find_top_base_solution(base_eval_dir: Path, frontier_root: Path, problem: str, variant: str | None,
                           task: str, model_tag: str, threshold: float = 0.1) -> tuple[float, str] | None:
    candidates = []
    for ev in sorted(base_eval_dir.glob(f"{task}.r*.eval.json")):
        try:
            j = json.loads(ev.read_text())
            r = j.get("reward_01", 0)
            if r < threshold:
                continue
            sp = Path(j.get("solution_path", ""))
            if not sp.exists():
                sd = Path(frontier_root) / "research" / "solutions" / problem
                if variant:
                    sd = sd / variant
                sp = sd / f"{model_tag}_{j['rollout']}.py"
            if sp.exists():
                candidates.append((r, sp.read_text()))
        except Exception:
            continue
    candidates.sort(reverse=True)
    return candidates[0] if candidates else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--tasks-json", default="/fsx/xuanj/ttt-discover/bench/tasks_19.json")
    ap.add_argument("--frontier-root", default="/fsx/xuanj/ttt-discover/src/frontier-cs")
    ap.add_argument("--base-eval-dir", default="/fsx/xuanj/ttt-discover/results/base/eval")
    ap.add_argument("--base-model-tag", default="dsr1q3_8b_base")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
    ap.add_argument("--output-dir", default="/fsx/xuanj/ttt-discover/results/sft_lora")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--readme-max-chars", type=int, default=8000)
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--reward-threshold", type=float, default=0.1)
    args = ap.parse_args()

    task_meta = find_task(Path(args.tasks_json), args.task)
    problem = task_meta["problem"]
    variant = task_meta.get("variant")
    print(f"[sft] task={args.task} problem={problem} variant={variant}", flush=True)

    top = find_top_base_solution(
        Path(args.base_eval_dir), Path(args.frontier_root), problem, variant,
        args.task, args.base_model_tag, threshold=args.reward_threshold,
    )
    if top is None:
        print(f"[sft] no base solution above threshold {args.reward_threshold}, skipping", flush=True)
        return
    base_reward, solution_code = top
    print(f"[sft] training on top base rollout reward={base_reward:.4f} ({len(solution_code)} chars)", flush=True)

    readme = read_readme(Path(args.frontier_root), problem, variant)
    prompt = PROMPT_TEMPLATE.format(readme=readme[: args.readme_max_chars])
    target = f"```python\n{solution_code}\n```"
    full_text = prompt + target

    print(f"[sft] loading {args.model}", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda:0",
    )
    model.gradient_checkpointing_enable()
    print(f"[sft] model loaded in {time.time()-t0:.0f}s", flush=True)

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

    enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=args.max_seq_len).to(model.device)
    prompt_enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_seq_len).to(model.device)
    prompt_len = prompt_enc["input_ids"].shape[1]
    seq = enc["input_ids"]
    attn = enc["attention_mask"]
    if seq.shape[1] <= prompt_len:
        print(f"[sft] target truncated; aborting", flush=True)
        return

    model.train()
    for ep in range(args.epochs):
        t_ep = time.time()
        optimizer.zero_grad()
        inp = seq[:, :-1]
        tgt = seq[:, 1:]
        out = model(input_ids=inp, attention_mask=attn[:, :-1])
        logits = out.logits  # [1, T-1, V]
        logp = F.log_softmax(logits, dim=-1)
        tgt_lp = logp.gather(2, tgt.unsqueeze(-1)).squeeze(-1)  # [1, T-1]
        # Mask: only score the target portion (positions where tgt is from the solution code)
        cmask = torch.zeros_like(tgt, dtype=torch.float32)
        cmask[:, prompt_len-1:] = 1.0
        cmask = cmask * attn[:, 1:].float()
        loss = -(tgt_lp * cmask).sum() / cmask.sum().clamp(min=1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
        optimizer.step()
        print(f"[sft] epoch {ep+1}/{args.epochs} loss={loss.item():.4f} dt={time.time()-t_ep:.1f}s", flush=True)

    out_dir = Path(args.output_dir) / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    print(f"[sft] saved LoRA adapter to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
