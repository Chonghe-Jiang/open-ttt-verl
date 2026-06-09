"""Paper-style entropic advantages for Slime TTT-Discover runs."""
from __future__ import annotations

import math
from collections.abc import Iterable

import torch


def _as_float_list(values: Iterable[float]) -> list[float]:
    return [float(v.item() if hasattr(v, "item") else v) for v in values]


def _kl_to_uniform(beta: float, rewards: list[float]) -> float:
    n = len(rewards)
    if n <= 1:
        return 0.0
    r = torch.tensor(rewards, dtype=torch.float64)
    r = r - r.max()
    q = torch.softmax(beta * r, dim=0)
    return float((q * torch.log(q * n + 1e-300)).sum().item())


def _solve_beta(rewards: list[float], gamma: float) -> float:
    if len(rewards) <= 1 or max(rewards) - min(rewards) < 1e-12 or gamma <= 0:
        return 0.0

    max_kl = math.log(len(rewards))
    target = min(float(gamma), max_kl - 1e-8)
    lo, hi = 0.0, 1.0
    while _kl_to_uniform(hi, rewards) < target and hi < 1e8:
        hi *= 2.0
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        if _kl_to_uniform(mid, rewards) < target:
            lo = mid
        else:
            hi = mid
    return hi


def _loo_entropic_scalars(rewards: list[float], gamma: float, eps: float) -> list[float]:
    n = len(rewards)
    if n <= 1:
        return [0.0] * n
    beta = _solve_beta(rewards, gamma)
    if beta == 0.0:
        return [0.0] * n
    rmax = max(rewards)
    exps = [math.exp(max(-80.0, min(80.0, beta * (r - rmax)))) for r in rewards]
    total = sum(exps)
    out: list[float] = []
    for e in exps:
        loo_z = (total - e) / max(1, n - 1)
        out.append(float(e / (loo_z + eps) - 1.0))
    return out


def _local_scalars(scalars: list[float], local_count: int) -> list[float]:
    if local_count <= 0:
        return []
    if len(scalars) <= local_count:
        return scalars + [0.0] * (local_count - len(scalars))

    try:
        from megatron.core import mpu

        dp_rank = mpu.get_data_parallel_rank(with_context_parallel=False)
        dp_size = mpu.get_data_parallel_world_size(with_context_parallel=False)
        if dp_size > 1 and len(scalars) % dp_size == 0:
            per_rank = len(scalars) // dp_size
            start = dp_rank * per_rank
            return scalars[start : start + local_count]
    except Exception:
        pass

    return scalars[:local_count]


def _local_sample_count(rollout_data) -> int:
    for key in ("tokens", "loss_masks", "response_lengths", "total_lengths", "log_probs", "ref_log_probs"):
        values = rollout_data.get(key)
        if isinstance(values, (list, tuple)):
            return len(values)
    return 0


def _build_entropic_scalars(rewards: list[float], group_size: int, gamma: float, eps: float) -> list[float]:
    scalars: list[float] = []
    for start in range(0, len(rewards), group_size):
        group = rewards[start : start + group_size]
        scalars.extend(_loo_entropic_scalars(group, gamma, eps))
    return scalars


def _full_like_sample(i: int, value: float, log_probs, ref_log_probs, loss_masks, response_lengths, kl_coef: float):
    if log_probs is not None and i < len(log_probs):
        base = torch.full_like(log_probs[i], float(value), dtype=torch.float32)
        if ref_log_probs is not None and i < len(ref_log_probs) and kl_coef and ref_log_probs[i].numel() == log_probs[i].numel():
            base = base - kl_coef * (log_probs[i].float() - ref_log_probs[i].float())
        return base
    if loss_masks is not None and i < len(loss_masks):
        return torch.full_like(loss_masks[i], float(value), dtype=torch.float32)
    if ref_log_probs is not None and i < len(ref_log_probs):
        return torch.full_like(ref_log_probs[i], float(value), dtype=torch.float32)
    length = int(response_lengths[i]) if response_lengths is not None and i < len(response_lengths) else 1
    return torch.full((length,), float(value), dtype=torch.float32)


def compute(args, rollout_data) -> None:
    log_probs = rollout_data.get("rollout_log_probs") if getattr(args, "use_rollout_logprobs", False) else rollout_data.get("log_probs")
    ref_log_probs = rollout_data.get("ref_log_probs")
    loss_masks = rollout_data.get("loss_masks")
    response_lengths = rollout_data.get("response_lengths")

    local_count = _local_sample_count(rollout_data)
    if local_count <= 0:
        rollout_data["advantages"] = []
        rollout_data["returns"] = []
        return

    group_size = int(getattr(args, "n_samples_per_prompt", 1) or 1)
    gamma = float(getattr(args, "entropic_kl_budget", getattr(args, "ttt_entropic_target_kl", math.log(2.0))))
    eps = float(getattr(args, "entropic_eps", 1e-8))
    kl_coef = float(getattr(args, "kl_coef", 0.0) or 0.0)

    raw_rewards = rollout_data.get("raw_reward")
    local_rewards = rollout_data.get("rewards")
    partition = rollout_data.get("_global_partition")

    if raw_rewards is not None and isinstance(raw_rewards, (list, tuple)) and len(raw_rewards) != local_count:
        full_scalars = _build_entropic_scalars(_as_float_list(raw_rewards), group_size, gamma, eps)
        if partition is not None and len(partition) == local_count and (not partition or max(partition) < len(full_scalars)):
            scalars = [full_scalars[i] for i in partition]
        else:
            scalars = _local_scalars(full_scalars, local_count)
    else:
        reward_source = raw_rewards if raw_rewards is not None else local_rewards
        rewards = _as_float_list(reward_source) if reward_source is not None else [0.0] * local_count
        if len(rewards) < local_count:
            rewards = rewards + [0.0] * (local_count - len(rewards))
        scalars = _build_entropic_scalars(rewards[:local_count], group_size, gamma, eps)

    if len(scalars) < local_count:
        scalars = scalars + [0.0] * (local_count - len(scalars))
    elif len(scalars) > local_count:
        scalars = scalars[:local_count]

    advantages = []
    returns = []
    for i, base_value in enumerate(scalars):
        base = _full_like_sample(i, base_value, log_probs, ref_log_probs, loss_masks, response_lengths, kl_coef)
        advantages.append(base)
        returns.append(base.clone())

    rollout_data["advantages"] = advantages
    rollout_data["returns"] = returns
