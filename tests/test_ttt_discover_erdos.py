from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from erdos_slime.ttt_discover.archive import PUCTArchive
from erdos_slime.ttt_discover.erdos_env import create_random_initial_state, score_erdos_result, verify_c5_solution
from erdos_slime.ttt_discover.sandbox import evaluate_python_code
from erdos_slime.ttt_slime import ttt_advantages, ttt_reward_post_process


class FakeSample:
    def __init__(self, reward: float, group_uid: str, raw_score: float | None = None) -> None:
        self.reward = reward
        self.group_index = 0
        self.index = 0
        self.metadata = {"ttt_group_uid": group_uid}
        if raw_score is not None:
            self.metadata["ttt_raw_score"] = raw_score
            self.metadata["raw_reward"] = raw_score
        self.train_metadata = dict(self.metadata)

    def get_reward_value(self, args):
        return self.reward


def test_verify_c5_solution_normalizes_reported_candidate() -> None:
    h = np.array([0.5, 0.5, 0.5, 0.5])
    c5 = float(np.max(np.correlate(h, 1.0 - h, mode="full") * 0.5))
    assert verify_c5_solution(h, c5, 4) == pytest.approx(c5)


def test_sandbox_executes_run_with_initial_values_and_budget(tmp_path) -> None:
    state = create_random_initial_state(seed=0, n_points_min=6, n_points_max=7)
    code = """
def run(seed=42, budget_s=1000, **kwargs):
    assert seed == 42
    assert budget_s == 5
    h = initial_h_values.copy()
    c5 = evaluate_erdos_solution((h, max(__import__("numpy").correlate(h, 1.0 - h, mode="full") * (2.0 / len(h))), len(h)))
    print("sandbox-ok")
    return h, c5, len(h)
"""
    result = evaluate_python_code(code, state=state, timeout_s=5, work_dir=tmp_path)
    assert result.error is None
    assert "sandbox-ok" in result.stdout
    assert score_erdos_result(result.output, code=code, timestep=0).raw_score > 0


def test_archive_keeps_best_lowest_raw_score_child(tmp_path) -> None:
    archive_path = tmp_path / "archive.json"
    parent = create_random_initial_state(seed=1, n_points_min=8, n_points_max=9)
    archive = PUCTArchive(archive_path, initial_states=[parent], rollout_n=2, topk_children=1)
    group_uid = "run:0:group:0"
    selected = archive.acquire_group(group_uid)
    assert selected.id == parent.id

    child_worse = create_random_initial_state(seed=2, n_points_min=8, n_points_max=9)
    child_better = create_random_initial_state(seed=3, n_points_min=8, n_points_max=9)
    child_worse.raw_score = 0.9
    child_worse.value = -0.9
    child_better.raw_score = 0.1
    child_better.value = -0.1

    assert archive.submit_child(group_uid, child_worse) is False
    assert archive.submit_child(group_uid, child_better) is True
    snapshot = archive.snapshot()
    assert snapshot["best_raw_score"] == pytest.approx(0.1)
    assert (tmp_path / "best_state.json").exists()
    assert (tmp_path / "puct_stats.json").exists()


def test_reward_post_process_computes_grouped_entropic_advantages() -> None:
    args = SimpleNamespace(reward_key=None, ttt_entropic_target_kl=math.log(2.0), ttt_advantage_clip=20.0)
    samples = [
        FakeSample(1.0, "g0", raw_score=0.4),
        FakeSample(2.0, "g0", raw_score=0.3),
        FakeSample(5.0, "g1", raw_score=0.2),
    ]
    raw_scores, advantages = ttt_reward_post_process(args, samples)
    assert raw_scores == pytest.approx([0.4, 0.3, 0.2])
    assert advantages[0] < 0
    assert advantages[1] > 0
    assert advantages[2] == 0.0
    assert samples[1].metadata["ttt_advantage"] == pytest.approx(advantages[1])


def test_ttt_advantages_broadcast_scalar_rewards_to_token_tensors() -> None:
    import torch

    rollout_data = {
        "rewards": [1.5, -0.5],
        "kl": [torch.zeros(3), torch.zeros(2)],
    }
    ttt_advantages(SimpleNamespace(), rollout_data)
    assert torch.equal(rollout_data["advantages"][0], torch.full((3,), 1.5))
    assert torch.equal(rollout_data["advantages"][1], torch.full((2,), -0.5))
    assert torch.equal(rollout_data["returns"][0], rollout_data["advantages"][0])
