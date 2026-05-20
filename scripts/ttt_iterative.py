"""
TTT-Discover with iterative tool-calling rollout (Yuchen-aligned).

Differences from ttt_faithful.py (single-shot rollout):
- Each rollout = up to 8 turns of tool calling, score = max across turns
- Prompt includes starter code (Yuchen-aligned)
- For RL gradient: we use the *first turn only* token logprobs and final score
  as the (state, action, reward) tuple. Multi-turn reward shaping is left for
  future work — this matches the simplest faithful adaptation.
- temperature=0.9, max_tokens=2048/turn

Same algorithm core as ttt_faithful.py:
- PUCT prioritization on state archive (Appendix A.2)
- Entropic adaptive β (Appendix A.1)
- KL penalty against base, importance-sampling correction
- Adam(lr=4e-5, β1=.9, β2=.95, ε=1e-8), LoRA rank 32

Train task default: cbl__low_av_tight_dl_large_oh (Yuchen STaR delta highest).
After training, save LoRA → eval zero-shot on 18 other tasks.
"""
from __future__ import annotations
import argparse
import dataclasses
import json
import math
import sys
import time
import uuid
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
    do_iterative_rollout, build_initial_user_message,
    SYSTEM_PROMPT,
)


# ------------------- task discovery (same as base_eval_iterative) -------------------

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


# ------------------- PUCT (same as ttt_faithful.py) -------------------

@dataclasses.dataclass
class ArchiveState:
    sid: str
    reward: float
    code: str
    parent_sid: str | None
    ancestor_sids: list[str] = dataclasses.field(default_factory=list)


class PUCTSampler:
    def __init__(self, c: float = 1.0, max_size: int = 1000, topk_children: int = 2):
        self.c = c
        self.max_size = max_size
        self.topk_children = topk_children
        self.states: list[ArchiveState] = []
        self.initial_sids: set[str] = set()
        self.n: dict[str, int] = {}
        self.m: dict[str, float] = {}
        self.T: int = 0

    def add_initial(self, code: str, reward: float):
        sid = str(uuid.uuid4())
        s = ArchiveState(sid=sid, reward=reward, code=code, parent_sid=None, ancestor_sids=[])
        self.states.append(s)
        self.initial_sids.add(sid)

    def _build_children_map(self) -> dict[str, set[str]]:
        children: dict[str, set[str]] = {}
        for s in self.states:
            if s.parent_sid:
                children.setdefault(s.parent_sid, set()).add(s.sid)
        return children

    def _full_lineage(self, sid: str, children: dict[str, set[str]]) -> set[str]:
        lineage = {sid}
        stack = [sid]
        while stack:
            x = stack.pop()
            for c in children.get(x, set()):
                if c not in lineage:
                    lineage.add(c)
                    stack.append(c)
        return lineage

    def _compute_prior(self, vals: np.ndarray) -> np.ndarray:
        if vals.size == 0:
            return np.array([])
        ranks = np.argsort(np.argsort(-vals))
        weights = (len(vals) - ranks).astype(np.float64)
        return weights / max(1.0, weights.sum())

    def _compute_scale(self, vals: np.ndarray) -> float:
        if vals.size == 0:
            return 1.0
        return float(max(vals.max() - vals.min(), 1e-6))

    def sample(self, num: int) -> list[ArchiveState]:
        if not self.states:
            raise ValueError("PUCT archive is empty")
        vals = np.array([s.reward for s in self.states])
        scale = self._compute_scale(vals)
        P = self._compute_prior(vals)
        sqrtT = math.sqrt(1.0 + self.T)
        scored = []
        for i, s in enumerate(self.states):
            n_s = self.n.get(s.sid, 0)
            m_s = self.m.get(s.sid, vals[i])
            Q = m_s if n_s > 0 else vals[i]
            bonus = self.c * scale * P[i] * sqrtT / (1.0 + n_s)
            scored.append((Q + bonus, vals[i], s))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        if num > 1:
            children = self._build_children_map()
            picked, blocked = [], set()
            for _, _, s in scored:
                if s.sid in blocked:
                    continue
                picked.append(s)
                blocked.update(self._full_lineage(s.sid, children))
                if len(picked) >= num:
                    break
            return picked
        return [scored[0][2]]

    def update(self, parent: ArchiveState, children_codes: list[str], children_rewards: list[float]):
        if not children_codes:
            self.n[parent.sid] = self.n.get(parent.sid, 0) + 1
            for aid in parent.ancestor_sids:
                self.n[aid] = self.n.get(aid, 0) + 1
            self.T += 1
            return
        y = max(children_rewards)
        self.m[parent.sid] = max(self.m.get(parent.sid, y), y)
        for aid in [parent.sid] + parent.ancestor_sids:
            self.n[aid] = self.n.get(aid, 0) + 1
        self.T += 1
        ranked = sorted(zip(children_rewards, children_codes), key=lambda x: x[0], reverse=True)
        for r, code in ranked[:self.topk_children]:
            sid = str(uuid.uuid4())
            child = ArchiveState(sid=sid, reward=r, code=code, parent_sid=parent.sid,
                                  ancestor_sids=[parent.sid] + parent.ancestor_sids)
            self.states.append(child)
        if len(self.states) > self.max_size:
            keep = [s for s in self.states if s.sid in self.initial_sids]
            others = sorted([s for s in self.states if s.sid not in self.initial_sids],
                            key=lambda s: s.reward, reverse=True)
            self.states = keep + others[:self.max_size - len(keep)]


# ------------------- Entropic advantage (same as ttt_faithful.py) -------------------

def entropic_adaptive_beta_advantages(rewards: torch.Tensor, gamma: float = math.log(2),
                                        beta_max: float = 1e6, iters: int = 60,
                                        eps: float = 1e-12) -> tuple[torch.Tensor, float]:
    r = rewards.float()
    k = r.shape[0]
    if k < 2:
        return torch.zeros_like(r), 0.0
    logK = math.log(k)
    def kl_hat(beta_scalar: float) -> float:
        b = r.new_tensor(beta_scalar)
        logits = b * (r - r.max(dim=0, keepdim=True).values)
        logq = logits - torch.logsumexp(logits, dim=0, keepdim=True)
        q = torch.exp(logq)
        kl = (q * (logq + logK)).sum(dim=0)
        return float(kl.mean().item())
    lo, hi = 0.0, 1.0
    if kl_hat(hi) < gamma:
        while hi < beta_max and kl_hat(hi) < gamma:
            hi *= 2.0
        beta = hi if kl_hat(hi) < gamma else None
    else:
        beta = None
    if beta is None:
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if kl_hat(mid) < gamma:
                lo = mid
            else:
                hi = mid
        beta = hi
    e = torch.exp(beta * (r - r.max(dim=0, keepdim=True).values))
    if k == 1:
        Z = e
    else:
        Z = (e.sum(dim=0, keepdim=True) - e) / (k - 1)
    w = e / (Z + eps)
    return w - 1.0, float(beta)


# ------------------- vLLM LoRA hot-reload -------------------

def vllm_load_lora(server: str, name: str, path: str, timeout: int = 60):
    r = requests.post(f"{server}/v1/load_lora_adapter",
                       json={"lora_name": name, "lora_path": path}, timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"vLLM load_lora failed: {r.status_code} {r.text}")


def vllm_unload_lora(server: str, name: str, timeout: int = 60):
    try:
        requests.post(f"{server}/v1/unload_lora_adapter",
                       json={"lora_name": name}, timeout=timeout)
    except Exception:
        pass


# ------------------- LoRA Trainer (same shape as ttt_faithful.py) -------------------

class LoRATrainer:
    def __init__(self, model_name: str, lora_rank: int = 32, lr: float = 4e-5, device: str = "cuda:0"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.base = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map=device,
        )
        self.base.gradient_checkpointing_enable()
        lora_cfg = LoraConfig(
            r=lora_rank, lora_alpha=lora_rank * 2,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.base, lora_cfg)
        self.optimizer = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=lr, betas=(0.9, 0.95), eps=1e-8,
        )
        self.device = device

    @torch.no_grad()
    def base_logprobs(self, input_ids: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        with self.model.disable_adapter():
            out = self.model(input_ids=input_ids[:, :-1], attention_mask=attn[:, :-1]).logits
        logp = F.log_softmax(out, dim=-1)
        tgt = input_ids[:, 1:]
        return logp.gather(2, tgt.unsqueeze(-1)).squeeze(-1)

    def step(self, prompt: str, completions: list[str], advantages: torch.Tensor,
             kl_coef: float, max_seq_len: int = 4096) -> dict:
        """Simplified IS-corrected entropic loss (no sampler logprobs since iterative
        rollouts make multi-turn IS expensive — we use on-policy approximation:
        loss = -mean_i [ A_i * mean_logp(a_i | prompt + first_user) ] + KL term.
        """
        self.model.train()
        self.optimizer.zero_grad()
        device = self.device

        prompt_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                     max_length=max_seq_len).input_ids.to(device)
        prompt_len = prompt_ids.shape[1]

        per_lp_cur, per_lp_base, per_mask, per_seq = [], [], [], []
        for c in completions:
            full = prompt + c
            ids = self.tokenizer(full, return_tensors="pt", truncation=True,
                                  max_length=max_seq_len).input_ids.to(device)
            attn = torch.ones_like(ids)
            out = self.model(input_ids=ids[:, :-1], attention_mask=attn[:, :-1]).logits
            logp_full = F.log_softmax(out, dim=-1)
            tgt = ids[:, 1:]
            logp_cur = logp_full.gather(2, tgt.unsqueeze(-1)).squeeze(-1)
            with torch.no_grad():
                logp_base = self.base_logprobs(ids, attn)
            mask = torch.zeros_like(tgt, dtype=torch.float32)
            mask[:, prompt_len - 1:] = 1.0
            per_lp_cur.append(logp_cur)
            per_lp_base.append(logp_base)
            per_mask.append(mask)
            per_seq.append(ids)

        total_diff = torch.tensor(0.0, device=device)
        total_mask = torch.tensor(0.0, device=device)
        for lp_c, lp_b, m in zip(per_lp_cur, per_lp_base, per_mask):
            d = (lp_c.detach() - lp_b) * m
            total_diff += d.sum()
            total_mask += m.sum()
        avg_logp_diff = total_diff / total_mask.clamp(min=1)

        total_loss = torch.tensor(0.0, device=device)
        n_used = 0
        for lp_c, lp_b, m, adv in zip(per_lp_cur, per_lp_base, per_mask, advantages):
            logp_diff = (lp_c.detach() - lp_b) * m
            kl_term = kl_coef * m * (avg_logp_diff - logp_diff)
            adv_eff = adv.to(device) + kl_term
            per_tok = adv_eff * lp_c * m
            denom = m.sum().clamp(min=1)
            loss_i = -per_tok.sum() / denom
            total_loss += loss_i
            n_used += 1
        loss = total_loss / max(1, n_used)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad],
                                        max_norm=1.0)
        self.optimizer.step()
        return {"loss": float(loss.item()), "avg_logp_diff": float(avg_logp_diff.item())}

    def save_lora(self, path: str):
        self.model.save_pretrained(path)


# ------------------- Main TTT loop -------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="cbl__low_av_tight_dl_large_oh",
                    help="Single task to train on (Yuchen trains on 1 task).")
    ap.add_argument("--tasks-json", default="/fsx/xuanj/ttt-discover/bench/tasks_19.json")
    ap.add_argument("--frontier-root", default="/fsx/xuanj/ttt-discover/src/frontier-cs")
    ap.add_argument("--vllm-url", required=True)
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
    ap.add_argument("--lora-name", default="ttt-iter")
    ap.add_argument("--lora-save-dir", default="/fsx/xuanj/ttt-discover/results/ttt_iter_lora")

    ap.add_argument("--num-steps", type=int, default=20)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--kl-coef", type=float, default=0.1)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max-tokens-per-turn", type=int, default=2048)
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--gen-timeout", type=int, default=600)
    ap.add_argument("--eval-timeout", type=int, default=1200)
    ap.add_argument("--rollout-concurrency", type=int, default=4)

    ap.add_argument("--scratch-dir", default="/fsx/xuanj/ttt-discover/scratch_iter_train")
    ap.add_argument("--output-dir", default="/fsx/xuanj/ttt-discover/results/ttt_iter")
    args = ap.parse_args()

    task_meta = find_task(Path(args.tasks_json), args.task)
    problem = task_meta["problem"]
    variant = task_meta.get("variant")
    readme = read_readme(Path(args.frontier_root), problem, variant)
    starter = read_starter_code(Path(args.frontier_root), problem, variant)

    scratch = Path(args.scratch_dir) / args.task
    scratch.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.output_dir) / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    lora_dir = Path(args.lora_save_dir) / args.task
    lora_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ttt-iter] task={args.task} starter_code={'yes' if starter else 'no'}", flush=True)

    # PUCT archive: initial state = empty
    sampler = PUCTSampler(c=1.0)
    sampler.add_initial(code="", reward=0.0)

    # Init trainer
    print(f"[ttt-iter] loading model {args.model}", flush=True)
    t_load = time.time()
    trainer = LoRATrainer(args.model, lora_rank=args.lora_rank, lr=args.lr, device="cuda:0")
    print(f"[ttt-iter] model loaded in {time.time()-t_load:.0f}s", flush=True)

    # Initial LoRA save + load into vLLM
    init_path = str(lora_dir / "init")
    trainer.save_lora(init_path)
    vllm_unload_lora(args.vllm_url, args.lora_name)
    vllm_load_lora(args.vllm_url, args.lora_name, init_path)
    print(f"[ttt-iter] vLLM loaded LoRA '{args.lora_name}' from {init_path}", flush=True)

    history = []
    best_overall = {"reward": 0.0, "code": "", "step": -1}
    t_start = time.time()

    for step_idx in range(args.num_steps):
        t_step = time.time()

        # Sample group_size rollouts using iterative tool calling with current LoRA
        # Each rollout: 8 turns of (gen, eval, feedback). final_score = max across turns.
        with ThreadPoolExecutor(max_workers=args.rollout_concurrency) as ex:
            futs = [ex.submit(do_iterative_rollout,
                               server=args.vllm_url, model=args.lora_name,
                               task_meta=task_meta, readme=readme, starter_code=starter,
                               frontier_root=Path(args.frontier_root), scratch=scratch,
                               max_turns=args.max_turns,
                               max_tokens_per_turn=args.max_tokens_per_turn,
                               temperature=args.temperature,
                               gen_timeout=args.gen_timeout,
                               eval_timeout=args.eval_timeout)
                     for _ in range(args.group_size)]
            rollouts = []
            for f in as_completed(futs):
                try:
                    rollouts.append(f.result())
                except Exception as e:
                    rollouts.append({"final_score": 0.0, "best_code": "",
                                      "turns": [], "messages": [], "_err": repr(e)})

        rewards = [r["final_score"] for r in rollouts]
        codes = [r["best_code"] for r in rollouts]

        r_t = torch.tensor(rewards, dtype=torch.float32)
        advantages, beta_used = entropic_adaptive_beta_advantages(r_t)

        # Gradient step: use first turn assistant text as the action
        # (simplification — paper does single-turn; for multi-turn we attribute reward to turn-1 action)
        # Build pseudo-prompt = system + initial user message
        pseudo_prompt = (
            f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n"
            f"{build_initial_user_message(task_meta, readme, starter)}\n<|assistant|>\n"
        )
        # completion = best turn's text (most informative)
        completions = []
        valid_idx = []
        for i, r in enumerate(rollouts):
            if r["turns"] and r["best_code"]:
                # use the turn that achieved best_code
                best_turn = max(r["turns"], key=lambda t: t.get("score", 0))
                completions.append(best_turn.get("text", ""))
                valid_idx.append(i)

        grad_metrics = {"loss": 0.0, "skipped": True}
        if len(valid_idx) >= 2 and r_t[valid_idx].max() > r_t[valid_idx].min():
            adv_v = advantages[valid_idx]
            try:
                grad_metrics = trainer.step(prompt=pseudo_prompt, completions=completions,
                                             advantages=adv_v, kl_coef=args.kl_coef)
                grad_metrics["skipped"] = False
            except torch.cuda.OutOfMemoryError as e:
                grad_metrics = {"loss": 0.0, "skipped": True, "oom": True}
                torch.cuda.empty_cache()

        # PUCT update — pick a parent (top of archive) and update with these children
        parent = sampler.sample(1)[0]
        sampler.update(parent=parent, children_codes=codes, children_rewards=rewards)

        # Save + reload LoRA into vLLM for next step
        ckpt_path = str(lora_dir / f"step_{step_idx:03d}")
        trainer.save_lora(ckpt_path)
        vllm_unload_lora(args.vllm_url, args.lora_name)
        vllm_load_lora(args.vllm_url, args.lora_name, ckpt_path)

        avg_r = float(np.mean(rewards))
        max_r = float(np.max(rewards))
        for code, r in zip(codes, rewards):
            if r > best_overall["reward"]:
                best_overall = {"reward": r, "code": code, "step": step_idx}

        elapsed = time.time() - t_start
        print(f"[ttt-iter] step {step_idx+1}/{args.num_steps} | "
              f"avg={avg_r:.4f} max={max_r:.4f} | best_so_far={best_overall['reward']:.4f} | "
              f"loss={grad_metrics.get('loss', 0):.4f} skipped={grad_metrics.get('skipped', True)} | "
              f"step_time={time.time()-t_step:.0f}s elapsed={elapsed:.0f}s", flush=True)
        history.append({
            "step": step_idx, "rewards": rewards, "avg": avg_r, "max": max_r,
            "best_so_far": best_overall["reward"],
            "beta": beta_used, "grad": grad_metrics,
            "elapsed_s": elapsed,
        })
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
        (out_dir / "best_solution.py").write_text(best_overall["code"])

    # Final LoRA save (use best step's LoRA actually — but for now save final)
    final_lora = str(lora_dir / "final")
    trainer.save_lora(final_lora)
    final = {
        "task": args.task, "best": {k: v for k, v in best_overall.items() if k != "code"},
        "wall_s": time.time() - t_start, "num_steps": args.num_steps,
        "group_size": args.group_size, "history": history,
        "lora_path": final_lora,
        "config": {"lr": args.lr, "lora_rank": args.lora_rank, "kl_coef": args.kl_coef,
                    "temperature": args.temperature, "max_turns": args.max_turns,
                    "max_tokens_per_turn": args.max_tokens_per_turn},
    }
    (out_dir / "final.json").write_text(json.dumps(final, indent=2))
    print(f"[ttt-iter] DONE task={args.task} best_reward={best_overall['reward']:.4f} "
          f"lora_saved_at={final_lora}", flush=True)


if __name__ == "__main__":
    main()
