from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Callable

import numpy as np
import torch


def _group_indices(index: np.ndarray | list[Any], size: int) -> dict[Any, list[int]]:
    if index is None:
        index = np.arange(size)
    groups: dict[Any, list[int]] = defaultdict(list)
    for i, group_id in enumerate(index):
        groups[group_id].append(i)
    return groups


def _solve_adaptive_beta(rewards: torch.Tensor, target_kl: float = math.log(2.0)) -> torch.Tensor:
    if rewards.numel() < 2 or torch.allclose(rewards, rewards[0]):
        return rewards.new_tensor(0.0)

    log_k = math.log(rewards.numel())

    def kl_for(beta_value: float) -> float:
        beta = rewards.new_tensor(beta_value)
        logits = beta * (rewards - rewards.max())
        log_q = logits - torch.logsumexp(logits, dim=0)
        q = torch.exp(log_q)
        return float((q * (log_q + log_k)).sum().item())

    lo, hi = 0.0, 1.0
    while hi < 1e6 and kl_for(hi) < target_kl:
        hi *= 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if kl_for(mid) < target_kl:
            lo = mid
        else:
            hi = mid
    return rewards.new_tensor(hi)


def compute_entropic_adaptive_beta(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray | list[Any] | None = None,
    config: Any | None = None,
    **_: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """TTT-Discover leave-one-out entropic advantages grouped by verl uid."""
    with torch.no_grad():
        scores = token_level_rewards.sum(dim=-1).float()
        advantages = torch.zeros_like(token_level_rewards, dtype=torch.float32)
        groups = _group_indices(index, scores.shape[0])

        for indices in groups.values():
            group_scores = scores[indices]
            if len(indices) < 2 or torch.allclose(group_scores, group_scores[0]):
                continue
            beta = _solve_adaptive_beta(group_scores)
            exp_scores = torch.exp(beta * (group_scores - group_scores.max()))
            loo_denominator = (exp_scores.sum() - exp_scores) / max(1, len(indices) - 1)
            scalar_advantages = exp_scores / (loo_denominator + 1e-12) - 1.0
            for local_idx, batch_idx in enumerate(indices):
                advantages[batch_idx] = scalar_advantages[local_idx] * response_mask[batch_idx]

    return advantages, advantages


def _masked_token_mean(loss_mat: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    return (loss_mat * response_mask).sum() / response_mask.sum().clamp_min(1.0)


def _aggregate_loss(loss_mat: torch.Tensor, response_mask: torch.Tensor, loss_agg_mode: str) -> torch.Tensor:
    if loss_agg_mode in {"token-mean", "seq-mean-token-sum-norm"}:
        return _masked_token_mean(loss_mat, response_mask)
    if loss_agg_mode == "seq-mean-token-sum":
        return (loss_mat * response_mask).sum(dim=-1).mean()
    if loss_agg_mode == "seq-mean-token-mean":
        seq_loss = (loss_mat * response_mask).sum(dim=-1) / response_mask.sum(dim=-1).clamp_min(1.0)
        return seq_loss.mean()
    return _masked_token_mean(loss_mat, response_mask)


def compute_ttt_reinforce_is(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Any | None = None,
    rollout_is_weights: torch.Tensor | None = None,
    rollout_log_probs: torch.Tensor | None = None,
    **_: Any,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """REINFORCE loss with rollout-policy importance weights."""
    if rollout_is_weights is not None:
        is_weights = rollout_is_weights.detach()
    elif rollout_log_probs is not None:
        is_weights = torch.exp(log_prob - rollout_log_probs).detach()
    else:
        is_weights = torch.ones_like(log_prob)
    losses = -is_weights * log_prob * advantages
    if config is not None:
        from verl.trainer.ppo.core_algos import agg_loss

        global_batch_info = getattr(config, "global_batch_info", {})
        loss = agg_loss(loss_mat=losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **global_batch_info)
    else:
        loss = _aggregate_loss(losses, response_mask, loss_agg_mode)
    valid_weights = is_weights[response_mask.bool()]
    approx_kl = _masked_token_mean(old_log_prob - log_prob, response_mask)
    metrics = {
        "actor/ppo_kl": float(approx_kl.detach().item()),
        "policy/ttt_is_mean": float(valid_weights.mean().item()) if valid_weights.numel() else 0.0,
        "policy/ttt_is_max": float(valid_weights.max().item()) if valid_weights.numel() else 0.0,
    }
    return loss, metrics


def register_ttt_algorithms(
    *,
    register_adv_est: Callable[[str], Callable[[Callable[..., Any]], Callable[..., Any]]] | None = None,
    register_policy_loss: Callable[[str], Callable[[Callable[..., Any]], Callable[..., Any]]] | None = None,
) -> None:
    if register_adv_est is None or register_policy_loss is None:
        from verl.trainer.ppo.core_algos import register_adv_est as verl_register_adv_est
        from verl.trainer.ppo.core_algos import register_policy_loss as verl_register_policy_loss

        register_adv_est = register_adv_est or verl_register_adv_est
        register_policy_loss = register_policy_loss or verl_register_policy_loss

    register_adv_est("entropic_adaptive_beta")(compute_entropic_adaptive_beta)
    register_policy_loss("ttt_reinforce_is")(compute_ttt_reinforce_is)


try:
    register_ttt_algorithms()
except ModuleNotFoundError:
    pass
