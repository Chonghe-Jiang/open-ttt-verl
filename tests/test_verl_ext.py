import math

import numpy as np
import torch

from guidance_ttt.verl_ext import (
    add_discover_centered_kl_to_advantages,
    compute_entropic_adaptive_beta,
    compute_ttt_reinforce_is,
)
from verl.protocol import DataProto
from verl.trainer.ppo.ray_trainer import compute_advantage
from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs


def test_entropic_adaptive_beta_removes_constant_reward_group_from_optimization_mask():
    rewards = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ]
    )
    optimization_mask = torch.ones_like(rewards)
    group_ids = np.array(["constant", "constant", "variable", "variable"])

    advantages, _ = compute_entropic_adaptive_beta(
        rewards,
        optimization_mask,
        index=group_ids,
        config={"remove_constant_reward_groups": True},
    )

    assert torch.equal(optimization_mask[:2], torch.zeros_like(optimization_mask[:2]))
    assert torch.equal(advantages[:2], torch.zeros_like(advantages[:2]))
    assert torch.equal(optimization_mask[2:], torch.ones_like(optimization_mask[2:]))
    assert not torch.equal(advantages[2:], torch.zeros_like(advantages[2:]))


def test_entropic_adaptive_beta_keeps_one_group_when_all_rewards_are_constant():
    rewards = torch.ones((4, 2))
    optimization_mask = torch.ones_like(rewards)
    group_ids = np.array(["first", "first", "second", "second"])

    advantages, _ = compute_entropic_adaptive_beta(
        rewards,
        optimization_mask,
        index=group_ids,
        config={"remove_constant_reward_groups": True},
    )

    assert torch.equal(optimization_mask[:2], torch.ones_like(optimization_mask[:2]))
    assert torch.equal(optimization_mask[2:], torch.zeros_like(optimization_mask[2:]))
    assert torch.equal(advantages, torch.zeros_like(advantages))


def test_compute_advantage_separates_response_and_optimization_masks():
    response_mask = torch.ones((4, 2))
    rewards = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ]
    )
    data = DataProto.from_dict(
        tensors={
            "response_mask": response_mask.clone(),
            "token_level_rewards": rewards,
        },
        non_tensors={
            "uid": np.array(["constant", "constant", "variable", "variable"], dtype=object)
        },
    )

    result = compute_advantage(
        data,
        adv_estimator="entropic_adaptive_beta",
        config={"remove_constant_reward_groups": True},
    )

    assert torch.equal(result.batch["response_mask"], response_mask)
    assert torch.equal(result.batch["optimization_mask"][:2], torch.zeros((2, 2)))
    assert torch.equal(result.batch["optimization_mask"][2:], torch.ones((2, 2)))

    # Diagnostics continue to use token validity and therefore remain defined
    # even when an optimization microbatch is completely excluded.
    metrics = compute_rollout_corr_metrics_from_logprobs(
        log_prob=torch.zeros((2, 2)),
        rollout_log_prob=torch.zeros((2, 2)),
        response_mask=result.batch["response_mask"][:2],
    )
    assert metrics["rollout_corr/training_ppl"] == 1.0


def test_discover_centered_kl_matches_official_formula_and_is_zero_mean():
    advantages = torch.tensor([[1.0, 1.0], [-1.0, -1.0]])
    rollout_log_probs = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]])
    ref_log_probs = torch.tensor([[-1.5, -1.5], [-2.5, -5.0]])
    response_mask = torch.ones_like(advantages)
    coef = 0.1

    updated, metrics = add_discover_centered_kl_to_advantages(
        advantages=advantages,
        rollout_log_probs=rollout_log_probs,
        ref_log_probs=ref_log_probs,
        response_mask=response_mask,
        coef=coef,
    )

    diff = rollout_log_probs - ref_log_probs
    expected_adjustment = coef * (diff.mean() - diff)
    assert torch.allclose(updated, advantages + expected_adjustment)
    assert math.isclose(float(expected_adjustment.mean()), 0.0, abs_tol=1e-7)
    assert math.isclose(metrics["discover/kl_logprob_diff_mean"], float(diff.mean()), abs_tol=1e-7)


def test_ttt_reinforce_is_uses_untruncated_rollout_ratio_when_weights_are_not_precomputed():
    old_log_prob = torch.zeros((1, 2))
    log_prob = torch.tensor([[1.0, -1.0]], requires_grad=True)
    rollout_log_probs = torch.tensor([[-1.0, -1.0]])
    advantages = torch.ones((1, 2))
    response_mask = torch.ones((1, 2))

    loss, metrics = compute_ttt_reinforce_is(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        rollout_log_probs=rollout_log_probs,
    )

    expected_weights = torch.exp(log_prob.detach() - rollout_log_probs)
    expected_loss = (-expected_weights * log_prob * advantages).mean()
    assert torch.allclose(loss, expected_loss)
    assert metrics["policy/ttt_is_max"] > 2.0


def test_ttt_reinforce_is_all_zero_optimization_mask_is_finite_and_differentiable():
    log_prob = torch.tensor([[1.0, -1.0]], requires_grad=True)
    zero_mask = torch.zeros((1, 2))

    loss, metrics = compute_ttt_reinforce_is(
        old_log_prob=torch.zeros((1, 2)),
        log_prob=log_prob,
        advantages=torch.zeros((1, 2)),
        response_mask=zero_mask,
        rollout_log_probs=torch.zeros((1, 2)),
    )

    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    assert metrics["policy/empty_optimization_mask"] == 1.0
    loss.backward()
    assert torch.equal(log_prob.grad, torch.zeros_like(log_prob))
