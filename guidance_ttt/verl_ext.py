from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Callable

import numpy as np
import torch


def _config_get(config: Any | None, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)


def _group_indices(index: np.ndarray | list[Any] | None, size: int) -> dict[Any, list[int]]:
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
    # ray_trainer passes its optimization_mask clone through the standard
    # response_mask estimator argument; mutations here must never reach the
    # original token-validity mask.
    with torch.no_grad():
        scores = token_level_rewards.sum(dim=-1).float()
        advantages = torch.zeros_like(token_level_rewards, dtype=torch.float32)
        groups = _group_indices(index, scores.shape[0])
        remove_constant_groups = bool(_config_get(config, "remove_constant_reward_groups", False))
        grouped_indices = list(groups.values())
        has_variable_group = any(
            len(indices) >= 2 and not torch.allclose(scores[indices], scores[indices][0])
            for indices in grouped_indices
        )
        for group_idx, indices in enumerate(grouped_indices):
            group_scores = scores[indices]
            if len(indices) < 2 or torch.allclose(group_scores, group_scores[0]):
                keep_official_empty_batch_fallback = not has_variable_group and group_idx == 0
                if remove_constant_groups and not keep_official_empty_batch_fallback:
                    response_mask[indices] = 0
                continue
            beta = _solve_adaptive_beta(group_scores)
            exp_scores = torch.exp(beta * (group_scores - group_scores.max()))
            loo_denominator = (exp_scores.sum() - exp_scores) / max(1, len(indices) - 1)
            scalar_advantages = exp_scores / (loo_denominator + 1e-12) - 1.0
            for local_idx, batch_idx in enumerate(indices):
                advantages[batch_idx] = scalar_advantages[local_idx] * response_mask[batch_idx]
    return advantages, advantages


def add_discover_centered_kl_to_advantages(
    *,
    advantages: torch.Tensor,
    rollout_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    coef: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply Discover's centered token-level base-policy term to advantages.

    The official implementation computes ``d = log pi_rollout - log pi_base``
    over active response tokens and adds ``coef * (mean(d) - d)``.  Centering
    preserves zero mean while discouraging tokens that moved too far from the
    frozen base policy.
    """
    if advantages.shape != rollout_log_probs.shape or advantages.shape != ref_log_probs.shape:
        raise ValueError(
            "Discover KL tensors must have identical shapes: "
            f"advantages={tuple(advantages.shape)}, rollout={tuple(rollout_log_probs.shape)}, "
            f"ref={tuple(ref_log_probs.shape)}"
        )
    if response_mask.shape != advantages.shape:
        raise ValueError(
            f"response_mask shape {tuple(response_mask.shape)} does not match {tuple(advantages.shape)}"
        )
    mask = response_mask.to(dtype=torch.float32)
    active_tokens = mask.sum()
    if float(active_tokens.item()) <= 0.0 or float(coef) == 0.0:
        return advantages, {
            "discover/kl_logprob_diff_mean": 0.0,
            "discover/kl_advantage_abs_mean": 0.0,
        }
    logprob_diff = (rollout_log_probs.float() - ref_log_probs.float()) * mask
    mean_diff = logprob_diff.sum() / active_tokens
    kl_advantage = float(coef) * mask * (mean_diff - logprob_diff)
    updated = advantages.float() + kl_advantage
    metrics = {
        "discover/kl_logprob_diff_mean": float(mean_diff.detach().item()),
        "discover/kl_advantage_abs_mean": float(
            (kl_advantage.abs().sum() / active_tokens).detach().item()
        ),
    }
    return updated.to(dtype=advantages.dtype), metrics


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
    # A complete constant-reward group is intentionally excluded from the
    # Discover update. Keep a differentiable zero so distributed workers still
    # execute the same backward/collective sequence without dividing by zero.
    if not response_mask.any():
        loss = log_prob.sum() * 0.0
        return loss, {
            "actor/ppo_kl": 0.0,
            "policy/ttt_is_mean": 0.0,
            "policy/ttt_is_max": 0.0,
            "policy/empty_optimization_mask": 1.0,
        }

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
        loss = (losses * response_mask).sum() / response_mask.sum().clamp_min(1.0)
    valid_weights = is_weights[response_mask.bool()]
    active_tokens = response_mask.sum().clamp_min(1.0)
    metrics = {
        "actor/ppo_kl": float(
            (((old_log_prob - log_prob) * response_mask).sum() / active_tokens).detach().item()
        ),
        "policy/ttt_is_mean": float(valid_weights.mean().item()) if valid_weights.numel() else 0.0,
        "policy/ttt_is_max": float(valid_weights.max().item()) if valid_weights.numel() else 0.0,
        "policy/empty_optimization_mask": 0.0,
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
