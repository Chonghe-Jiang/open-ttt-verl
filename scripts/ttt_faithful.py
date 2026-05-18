"""
Faithful port of TTT-Discover (Yuksekgonul 2026, arxiv 2601.16175) to our infra.

Replaces Tinker with: vLLM (sampling) + HuggingFace+PEFT (gradient) + LoRA hot-reload.
Algorithm preserved exactly:
  - PUCT reuse over a state archive (paper §3.2 / Appendix A.2)
  - Entropic objective with adaptive β solved by bisection on KL(qβ||uniform)=ln(2) (Appendix A.1)
  - LOO entropic advantages
  - KL penalty against base policy
  - Importance-sampling correction for sampler/learner mismatch
  - Adam (lr=4e-5, β1=0.9, β2=0.95, ε=1e-8), LoRA rank 32

We compromise only on batch dimensions to fit p5en wall budget:
  - paper: 50 steps × (8 groups × 64 rollouts) = 25600 rollouts
  - us:    20 steps × (1 group  × 8  rollouts) = 160   rollouts per task
The algorithm itself is unchanged.
"""
from __future__ import annotations
import argparse
import asyncio
import dataclasses
import json
import math
import os
import re
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


# -------------------- Prompt + extraction (same as base eval) --------------------

PROMPT_TEMPLATE = """You are an expert Python engineer. Solve the following problem from the Frontier-CS benchmark.

PROBLEM SPECIFICATION:
{readme}

{context_block}

Write a complete, self-contained Python solution. Output ONLY the Python source code inside a single ```python ... ``` fenced block. No explanations outside the block.
"""

CONTEXT_HEADER = """
KNOWN-GOOD SOLUTIONS (use as reference; you MAY borrow good ideas, but produce a NEW improved solution):
"""


def extract_python_block(text: str) -> str:
    m = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).rstrip() + "\n"
    m = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).rstrip() + "\n"
    return text.rstrip() + "\n"


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


def make_context_block(buffer_top: list[dict], top_k: int = 2) -> str:
    if not buffer_top:
        return ""
    items = sorted(buffer_top, key=lambda r: r["reward"], reverse=True)[:top_k]
    parts = [CONTEXT_HEADER]
    for i, item in enumerate(items, 1):
        parts.append(f"\n[Solution {i}, reward={item['reward']:.4f}]\n```python\n{item['code']}\n```\n")
    return "".join(parts)


# -------------------- PUCT sampler (paper Appendix A.2) --------------------

@dataclasses.dataclass
class ArchiveState:
    """A state s in the PUCT archive. Holds code (the action that led here) and reward."""
    sid: str
    reward: float
    code: str            # solution code at this state (used as context for children)
    parent_sid: Optional[str]
    ancestor_sids: list[str] = dataclasses.field(default_factory=list)


class PUCTSampler:
    """Faithful implementation of paper's PUCT prioritization (Appendix A.2)."""
    def __init__(self, c: float = 1.0, max_size: int = 1000, topk_children: int = 2):
        self.c = c
        self.max_size = max_size
        self.topk_children = topk_children
        self.states: list[ArchiveState] = []
        self.initial_sids: set[str] = set()
        self.n: dict[str, int] = {}     # visit count
        self.m: dict[str, float] = {}   # max child reward
        self.T: int = 0                 # total expansions

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
        """Linear rank prior: rank 0 (best) gets weight N, last rank gets 1."""
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
        # update m[parent] = max child reward
        y = max(children_rewards)
        self.m[parent.sid] = max(self.m.get(parent.sid, y), y)
        # backprop visitation (parent + ancestors)
        for aid in [parent.sid] + parent.ancestor_sids:
            self.n[aid] = self.n.get(aid, 0) + 1
        self.T += 1
        # add top-k children to archive
        ranked = sorted(zip(children_rewards, children_codes), key=lambda x: x[0], reverse=True)
        for r, code in ranked[: self.topk_children]:
            sid = str(uuid.uuid4())
            child = ArchiveState(
                sid=sid, reward=r, code=code, parent_sid=parent.sid,
                ancestor_sids=[parent.sid] + parent.ancestor_sids,
            )
            self.states.append(child)
        # enforce size cap (keep initials + top-by-reward)
        if len(self.states) > self.max_size:
            keep = [s for s in self.states if s.sid in self.initial_sids]
            others = sorted([s for s in self.states if s.sid not in self.initial_sids],
                            key=lambda s: s.reward, reverse=True)
            self.states = keep + others[: self.max_size - len(keep)]


# -------------------- Entropic adaptive β (paper Appendix A.1) --------------------

def entropic_adaptive_beta_advantages(rewards: torch.Tensor, gamma: float = math.log(2),
                                       beta_max: float = 1e6, iters: int = 60, eps: float = 1e-12) -> tuple[torch.Tensor, float]:
    """Match official ttt_discover/rl/train.py compute_advantages exactly (entropic_adaptive_beta).
    Returns (advantages, beta)."""
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
        if kl_hat(hi) < gamma:
            beta = hi
        else:
            beta = None
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
    advantages = w - 1.0
    return advantages, float(beta)


# -------------------- vLLM client (sampling + logprobs) --------------------

def vllm_complete(server: str, model: str, prompt: str, max_tokens: int, temperature: float,
                  top_p: float, timeout: int) -> dict:
    """Returns {"text", "tokens", "logprobs"} where logprobs are sampler's logprobs of chosen tokens."""
    payload = {
        "model": model, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": temperature, "top_p": top_p, "stream": False,
        "logprobs": 1,  # ask vLLM for logprob of chosen token
    }
    r = requests.post(f"{server}/v1/completions", json=payload, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    choice = j["choices"][0]
    text = choice["text"]
    lp_obj = choice.get("logprobs") or {}
    token_logprobs = lp_obj.get("token_logprobs") or []
    tokens_str = lp_obj.get("tokens") or []
    return {"text": text, "tokens_str": tokens_str, "token_logprobs": token_logprobs,
            "completion_tokens": j.get("usage", {}).get("completion_tokens", 0)}


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


# -------------------- Frontier-CS evaluator --------------------

def evaluate_with_frontier(frontier_root: Path, problem: str, variant: str | None,
                           code: str, scratch: Path, timeout: int = 1800) -> tuple[float, dict]:
    sol = scratch / f"sol_{int(time.time()*1e6)}_{uuid.uuid4().hex[:6]}.py"
    sol.write_text(code)
    pid = f"{problem}/{variant}" if variant else problem
    cmd = ["frontier", "eval", "research", pid, str(sol), "--backend", "docker"]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        out = proc.stdout
        m = re.search(r"Score:\s*([\-\d\.eE]+)", out)
        score = float(m.group(1)) if m else 0.0
        if score < 0:
            score = 0.0
        return score / 100.0, {"score_100": score, "elapsed_s": time.time() - t0,
                                "rc": proc.returncode}
    except subprocess.TimeoutExpired:
        return 0.0, {"score_100": 0.0, "rc": -1, "elapsed_s": time.time() - t0}
    finally:
        try:
            sol.unlink()
        except Exception:
            pass


# -------------------- HF + PEFT trainer --------------------

class LoRATrainer:
    """HF model with PEFT LoRA + Adam, matched to paper hyperparams.
    Supports importance-sampling-corrected entropic loss with KL penalty against base."""

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
        """Compute logp under base model (LoRA disabled). Returns [B, T-1] of tgt logprobs."""
        with self.model.disable_adapter():
            out = self.model(input_ids=input_ids[:, :-1], attention_mask=attn[:, :-1]).logits
        logp = F.log_softmax(out, dim=-1)
        tgt = input_ids[:, 1:]
        return logp.gather(2, tgt.unsqueeze(-1)).squeeze(-1)

    def step(self, prompt: str, completions: list[str], sampler_token_logprobs: list[list[float]],
             advantages: torch.Tensor, kl_coef: float, max_seq_len: int = 16384) -> dict:
        """One importance-sampling-corrected gradient step.

        For each rollout i:
          - tokenize prompt + completion_i
          - compute current-policy logp (LoRA on)
          - compute base-policy logp (LoRA off) for KL penalty
          - importance ratio: rho_i = exp(logp_current - logp_sampler) (per-token)
          - effective per-token advantage: A_i + kl_coef * (avg_logp_diff - logp_diff_i)
          - loss_i = - mean over completion tokens of [A_eff_i * rho_i * logp_current_i]
        Sum loss across rollouts (advantages already centered/LOO).
        """
        self.model.train()
        self.optimizer.zero_grad()
        device = self.device

        # Tokenize prompt once (used for prompt_len)
        prompt_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                     max_length=max_seq_len).input_ids.to(device)
        prompt_len = prompt_ids.shape[1]

        # Tokenize each (prompt + completion) pair
        rollout_inputs = []
        for c in completions:
            full = prompt + c
            ids = self.tokenizer(full, return_tensors="pt", truncation=True,
                                  max_length=max_seq_len).input_ids.to(device)
            rollout_inputs.append(ids)

        # First pass: compute base logprobs (LoRA disabled) for ALL rollouts to compute avg_logp_diff
        # Process one at a time to control memory.
        per_rollout_logp_current = []
        per_rollout_logp_base = []
        per_rollout_logp_sampler = []
        per_rollout_mask = []
        per_rollout_seq = []
        for i, seq in enumerate(rollout_inputs):
            attn = torch.ones_like(seq)
            # current-policy logp (LoRA on, with grad)
            out = self.model(input_ids=seq[:, :-1], attention_mask=attn[:, :-1]).logits
            logp_full = F.log_softmax(out, dim=-1)
            tgt = seq[:, 1:]
            logp_cur = logp_full.gather(2, tgt.unsqueeze(-1)).squeeze(-1)  # [1, T-1]
            # base-policy logp (LoRA disabled, no grad)
            with torch.no_grad():
                logp_base = self.base_logprobs(seq, attn)
            # mask for completion tokens only
            mask = torch.zeros_like(tgt, dtype=torch.float32)
            mask[:, prompt_len - 1:] = 1.0
            # sampler-time logp (provided by vLLM at sampling). Pad to T-1.
            sampler_lp = sampler_token_logprobs[i]
            # Sampler logp covers only completion tokens (one per completion token).
            # Place them at positions [prompt_len-1 ... prompt_len-1 + len(sampler_lp)].
            sampler_padded = torch.zeros_like(tgt, dtype=torch.float32)
            n_comp_tokens = min(len(sampler_lp), tgt.shape[1] - (prompt_len - 1))
            if n_comp_tokens > 0:
                sampler_padded[0, prompt_len - 1: prompt_len - 1 + n_comp_tokens] = torch.tensor(
                    sampler_lp[:n_comp_tokens], dtype=torch.float32, device=device)

            per_rollout_logp_current.append(logp_cur)
            per_rollout_logp_base.append(logp_base)
            per_rollout_logp_sampler.append(sampler_padded)
            per_rollout_mask.append(mask)
            per_rollout_seq.append(seq)

        # avg logp diff (current vs base) over all completion tokens, used as KL baseline
        total_diff = torch.tensor(0.0, device=device)
        total_mask = torch.tensor(0.0, device=device)
        for lp_cur, lp_base, m in zip(per_rollout_logp_current, per_rollout_logp_base, per_rollout_mask):
            d = (lp_cur.detach() - lp_base) * m
            total_diff = total_diff + d.sum()
            total_mask = total_mask + m.sum()
        avg_logp_diff = total_diff / total_mask.clamp(min=1)

        # Compute loss
        total_loss = torch.tensor(0.0, device=device)
        n_used = 0
        for i, (lp_cur, lp_base, lp_sampler, m, seq, adv) in enumerate(
                zip(per_rollout_logp_current, per_rollout_logp_base,
                    per_rollout_logp_sampler, per_rollout_mask, per_rollout_seq, advantages)):
            logp_diff = (lp_cur.detach() - lp_base) * m  # detached, used to compute KL term
            # KL-augmented advantage: A_eff = A + kl_coef * mask * (avg_logp_diff - logp_diff)
            kl_term = kl_coef * m * (avg_logp_diff - logp_diff)
            adv_eff = adv.to(device) + kl_term  # scalar adv broadcast over tokens via kl_term shape

            # importance ratio per token: exp(logp_current - logp_sampler)
            with torch.no_grad():
                rho = torch.exp(lp_cur.detach() - lp_sampler)
                rho = rho * m  # zero outside completion

            # Loss: - mean_t [A_eff_t * rho_t * logp_current_t]
            per_tok = adv_eff * rho * lp_cur * m
            denom = m.sum().clamp(min=1)
            loss_i = -per_tok.sum() / denom
            total_loss = total_loss + loss_i
            n_used += 1

        loss = total_loss / max(1, n_used)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], max_norm=1.0)
        self.optimizer.step()
        return {"loss": float(loss.item()), "avg_logp_diff": float(avg_logp_diff.item())}

    def save_lora(self, path: str):
        self.model.save_pretrained(path)


# -------------------- Main TTT-Discover loop --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--tasks-json", default="/fsx/xuanj/ttt-discover/bench/tasks_19.json")
    ap.add_argument("--frontier-root", default="/fsx/xuanj/ttt-discover/src/frontier-cs")
    ap.add_argument("--vllm-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
    ap.add_argument("--lora-name", default="ttt-active",
                    help="Name vLLM uses to refer to current TTT LoRA. Will be reloaded each step.")
    ap.add_argument("--lora-save-dir", default="/fsx/xuanj/ttt-discover/results/ttt_faithful_lora")

    # Paper hyperparams (Table 9)
    ap.add_argument("--num-steps", type=int, default=20)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--groups-per-batch", type=int, default=1)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--kl-coef", type=float, default=0.1)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--gen-timeout", type=int, default=1800)
    ap.add_argument("--readme-max-chars", type=int, default=8000)
    ap.add_argument("--eval-concurrency", type=int, default=8)
    ap.add_argument("--eval-timeout", type=int, default=1200)
    ap.add_argument("--max-seq-len", type=int, default=16384)
    ap.add_argument("--puct-c", type=float, default=1.0)

    # Preload base eval rollouts as initial archive states
    ap.add_argument("--preload-base-eval-dir", default="/fsx/xuanj/ttt-discover/results/base/eval")
    ap.add_argument("--preload-base-solutions-root", default="/fsx/xuanj/ttt-discover/src/frontier-cs/research/solutions")
    ap.add_argument("--preload-base-tag", default="dsr1q3_8b_base")

    ap.add_argument("--scratch-dir", default="/fsx/xuanj/ttt-discover/scratch")
    ap.add_argument("--output-dir", default="/fsx/xuanj/ttt-discover/results/ttt_faithful")
    args = ap.parse_args()

    task_meta = find_task(Path(args.tasks_json), args.task)
    problem = task_meta["problem"]
    variant = task_meta.get("variant")
    readme = read_readme(Path(args.frontier_root), problem, variant)

    scratch = Path(args.scratch_dir) / args.task / "ttt_faithful"
    scratch.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.output_dir) / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    lora_dir = Path(args.lora_save_dir) / args.task
    lora_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ttt-faithful] task={args.task} problem={problem} variant={variant}", flush=True)
    print(f"[ttt-faithful] hyperparams: steps={args.num_steps} group={args.group_size} "
          f"lr={args.lr} lora_rank={args.lora_rank} kl_coef={args.kl_coef}", flush=True)

    # 1) Initialize archive: empty + each base rollout that scored > 0
    sampler = PUCTSampler(c=args.puct_c)
    sampler.add_initial(code="", reward=0.0)  # paper line 3: H0 = {(<empty>, R(<empty>), {})}
    base_eval_dir = Path(args.preload_base_eval_dir)
    if base_eval_dir.exists():
        for ev in sorted(base_eval_dir.glob(f"{args.task}.r*.eval.json")):
            try:
                j = json.loads(ev.read_text())
                r = j.get("reward_01", 0)
                if r <= 0:
                    continue
                sp = Path(j.get("solution_path", ""))
                if not sp.exists():
                    sd = Path(args.preload_base_solutions_root) / problem
                    if variant:
                        sd = sd / variant
                    sp = sd / f"{args.preload_base_tag}_{j['rollout']}.py"
                if sp.exists():
                    sampler.add_initial(code=sp.read_text(), reward=r)
            except Exception:
                continue
    print(f"[ttt-faithful] archive seeded with {len(sampler.states)} states "
          f"(top reward = {max(s.reward for s in sampler.states):.4f})", flush=True)

    # 2) Init LoRATrainer
    print(f"[ttt-faithful] loading {args.model} with LoRA rank {args.lora_rank}", flush=True)
    t0 = time.time()
    trainer = LoRATrainer(args.model, lora_rank=args.lora_rank, lr=args.lr, device="cuda:0")
    print(f"[ttt-faithful] model loaded in {time.time()-t0:.0f}s", flush=True)

    # 3) Save initial LoRA (random init) and load into vLLM
    trainer.save_lora(str(lora_dir / "init"))
    vllm_unload_lora(args.vllm_url, args.lora_name)
    vllm_load_lora(args.vllm_url, args.lora_name, str(lora_dir / "init"))
    print(f"[ttt-faithful] vLLM loaded LoRA '{args.lora_name}' from {lora_dir / 'init'}", flush=True)

    history = []
    best_overall = {"reward": max(s.reward for s in sampler.states),
                    "code": max(sampler.states, key=lambda s: s.reward).code,
                    "step": -1}

    t_start = time.time()
    for step_idx in range(args.num_steps):
        t_step = time.time()

        # 4a) PUCT pick a state to expand from (groups_per_batch states)
        picked = sampler.sample(args.groups_per_batch)
        # We use 1 group/batch in our budget. Still iterate to keep code general.
        all_step_rewards = []
        all_step_metrics = []
        for grp_idx, s in enumerate(picked):
            # Build prompt with the picked state's code as buffer context
            buf_top = []
            if s.code:
                buf_top.append({"reward": s.reward, "code": s.code})
            ctx = make_context_block(buf_top, top_k=2)
            prompt = PROMPT_TEMPLATE.format(readme=readme[: args.readme_max_chars], context_block=ctx)

            # 4b) Sample group_size rollouts via vLLM with current LoRA
            t_sample = time.time()
            with ThreadPoolExecutor(max_workers=args.group_size) as ex:
                futs = [ex.submit(vllm_complete, args.vllm_url, args.lora_name, prompt,
                                   args.max_tokens, args.temperature, args.top_p, args.gen_timeout)
                        for _ in range(args.group_size)]
                rollouts = []
                for f in futs:
                    try:
                        rollouts.append(f.result())
                    except Exception as e:
                        rollouts.append({"text": "", "tokens_str": [], "token_logprobs": [],
                                          "completion_tokens": 0, "_err": repr(e)})
            t_sample = time.time() - t_sample

            # 4c) Extract code, evaluate via Docker
            codes = [extract_python_block(r["text"]) for r in rollouts]
            t_eval = time.time()
            rewards = [0.0] * len(codes)
            metas = [None] * len(codes)
            with ThreadPoolExecutor(max_workers=args.eval_concurrency) as ex:
                fut_to_i = {ex.submit(evaluate_with_frontier, Path(args.frontier_root),
                                       problem, variant, codes[i], scratch, args.eval_timeout): i
                             for i in range(len(codes)) if codes[i].strip()}
                for f in as_completed(fut_to_i):
                    i = fut_to_i[f]
                    try:
                        rewards[i], metas[i] = f.result()
                    except Exception as e:
                        rewards[i], metas[i] = 0.0, {"err": repr(e)}
            t_eval = time.time() - t_eval

            r_t = torch.tensor(rewards, dtype=torch.float32)

            # 4d) Compute entropic-adaptive-β advantages
            advantages, beta_used = entropic_adaptive_beta_advantages(r_t)

            # 4e) Gradient step (importance-sampling corrected, with KL penalty)
            t_grad = time.time()
            # Filter out failed rollouts (empty completions with no logprobs)
            valid_idx = [i for i, r in enumerate(rollouts) if r["token_logprobs"] and codes[i].strip()]
            grad_metrics = {"loss": 0.0, "skipped": True}
            if len(valid_idx) >= 2 and r_t[valid_idx].max() > r_t[valid_idx].min():
                completions_v = [rollouts[i]["text"] for i in valid_idx]
                sampler_lp_v = [rollouts[i]["token_logprobs"] for i in valid_idx]
                adv_v = advantages[valid_idx]
                grad_metrics = trainer.step(prompt=prompt, completions=completions_v,
                                             sampler_token_logprobs=sampler_lp_v,
                                             advantages=adv_v, kl_coef=args.kl_coef,
                                             max_seq_len=args.max_seq_len)
                grad_metrics["skipped"] = False
            t_grad = time.time() - t_grad

            # 4f) Update PUCT archive
            sampler.update(parent=s, children_codes=codes, children_rewards=rewards)

            all_step_rewards.extend(rewards)
            all_step_metrics.append({
                "group_idx": grp_idx, "rewards": rewards, "beta": beta_used,
                "t_sample_s": t_sample, "t_eval_s": t_eval, "t_grad_s": t_grad,
                "grad_metrics": grad_metrics,
                "parent_reward": s.reward,
            })

            # Track best overall
            for code, r in zip(codes, rewards):
                if r > best_overall["reward"]:
                    best_overall = {"reward": r, "code": code, "step": step_idx}

        # 4g) Save updated LoRA + hot-reload into vLLM
        ckpt_path = str(lora_dir / f"step_{step_idx:03d}")
        trainer.save_lora(ckpt_path)
        vllm_unload_lora(args.vllm_url, args.lora_name)
        vllm_load_lora(args.vllm_url, args.lora_name, ckpt_path)

        avg_r = float(np.mean(all_step_rewards)) if all_step_rewards else 0.0
        max_r = float(np.max(all_step_rewards)) if all_step_rewards else 0.0
        elapsed = time.time() - t_start
        print(f"[ttt-faithful] step {step_idx+1}/{args.num_steps} | "
              f"avg={avg_r:.4f} max={max_r:.4f} | best_so_far={best_overall['reward']:.4f} | "
              f"step_time={time.time()-t_step:.0f}s elapsed={elapsed:.0f}s", flush=True)
        history.append({
            "step": step_idx, "rewards": all_step_rewards, "avg": avg_r, "max": max_r,
            "best_so_far": best_overall["reward"],
            "groups": all_step_metrics, "elapsed_s": elapsed,
        })
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
        # Always persist best-so-far code
        (out_dir / "best_solution.py").write_text(best_overall["code"])
        (out_dir / "best_meta.json").write_text(json.dumps(
            {k: v for k, v in best_overall.items() if k != "code"}, indent=2))

    final = {
        "task": args.task, "problem": problem, "variant": variant,
        "wall_s": time.time() - t_start, "num_steps": args.num_steps,
        "group_size": args.group_size, "history": history,
        "best": {k: v for k, v in best_overall.items() if k != "code"},
        "config": {"lr": args.lr, "lora_rank": args.lora_rank, "kl_coef": args.kl_coef,
                    "temperature": args.temperature, "max_tokens": args.max_tokens},
    }
    (out_dir / "final.json").write_text(json.dumps(final, indent=2))
    print(f"[ttt-faithful] DONE task={args.task} best_reward={best_overall['reward']:.4f}", flush=True)


if __name__ == "__main__":
    main()
