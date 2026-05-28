import numpy as np
import torch

from verl_ttt_discover import verl_ext


def test_importing_verl_ext_registers_ttt_algorithms():
    registered_adv = {}
    registered_loss = {}

    def register_adv_est(name):
        def decorator(fn):
            registered_adv[name] = fn
            return fn

        return decorator

    def register_policy_loss(name):
        def decorator(fn):
            registered_loss[name] = fn
            return fn

        return decorator

    verl_ext.register_ttt_algorithms(
        register_adv_est=register_adv_est,
        register_policy_loss=register_policy_loss,
    )

    assert registered_adv["entropic_adaptive_beta"] is verl_ext.compute_entropic_adaptive_beta
    assert registered_loss["ttt_reinforce_is"] is verl_ext.compute_ttt_reinforce_is


def test_entropic_adaptive_beta_groups_by_uid_and_prefers_high_reward():
    token_level_rewards = torch.tensor(
        [
            [0.0, 1.0],
            [0.0, 3.0],
            [0.0, 2.0],
            [0.0, 0.5],
        ]
    )
    response_mask = torch.ones_like(token_level_rewards)
    index = np.array(["group-a", "group-a", "group-b", "group-b"], dtype=object)

    advantages, returns = verl_ext.compute_entropic_adaptive_beta(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )

    assert advantages[1, -1] > advantages[0, -1]
    assert advantages[2, -1] > advantages[3, -1]
    assert torch.allclose(advantages[:, 0], advantages[:, -1])
    assert torch.allclose(advantages, returns)


def test_ttt_reinforce_is_uses_rollout_log_probs_as_importance_weights():
    old_log_prob = torch.zeros(2, 2)
    log_prob = torch.tensor([[-0.1, -0.2], [-0.5, -0.5]])
    rollout_log_probs = torch.tensor([[-0.2, -0.2], [-0.5, -0.7]])
    rollout_is_weights = torch.exp(log_prob - rollout_log_probs).detach()
    advantages = torch.tensor([[1.0, 1.0], [-1.0, -1.0]])
    response_mask = torch.ones(2, 2)

    loss, metrics = verl_ext.compute_ttt_reinforce_is(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode="token-mean",
        config=None,
        rollout_is_weights=rollout_is_weights,
        unexpected_verl_kwarg=True,
    )

    expected = -(rollout_is_weights * log_prob * advantages * response_mask).sum() / response_mask.sum()
    assert loss == expected
    assert metrics["policy/ttt_is_mean"] > 0
