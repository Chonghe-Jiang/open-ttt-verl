import numpy as np
import pytest

from verl_ttt_discover.erdos_env import (
    ErdosEvaluationError,
    build_erdos_prompt,
    create_random_initial_state,
    evaluate_erdos_solution,
    score_erdos_result,
    verify_c5_solution,
)
from verl_ttt_discover.state import DiscoveryState


def test_verify_c5_solution_accepts_valid_step_function():
    h_values = np.full(8, 0.5)
    dx = 2.0 / len(h_values)
    c5_bound = float(np.max(np.correlate(h_values, 1.0 - h_values, mode="full") * dx))

    assert verify_c5_solution(h_values, c5_bound, len(h_values)) == pytest.approx(c5_bound)
    assert evaluate_erdos_solution((h_values, c5_bound, len(h_values))) == pytest.approx(c5_bound)


def test_verify_c5_solution_rejects_invalid_values():
    with pytest.raises(ErdosEvaluationError, match="not in \\[0, 1\\]"):
        verify_c5_solution(np.array([1.2, -0.2]), 0.1, 2)

    with pytest.raises(ErdosEvaluationError, match="C5 mismatch"):
        verify_c5_solution(np.full(4, 0.5), 0.123, 4)


def test_score_erdos_result_returns_reward_and_child_state_fields():
    h_values = np.full(6, 0.5)
    c5_bound = float(np.max(np.correlate(h_values, 1.0 - h_values, mode="full") * (2.0 / len(h_values))))

    scored = score_erdos_result((h_values, c5_bound, len(h_values)), code="print('ok')", timestep=5, stdout="ok")

    assert scored.reward == pytest.approx(1.0 / (1e-8 + c5_bound))
    assert scored.state.value == pytest.approx(-c5_bound)
    assert scored.state.raw_score == pytest.approx(c5_bound)
    assert scored.state.construction == h_values.tolist()
    assert scored.state.observation == "ok"


def test_build_erdos_prompt_includes_bound_and_initial_construction():
    state = DiscoveryState(
        timestep=3,
        value=-0.381,
        raw_score=0.381,
        code="def run(): pass",
        construction=[0.5, 0.5],
        id="state",
    )

    prompt = build_erdos_prompt(state, budget_s=60, cpus=2, target_c5=0.3808)

    assert "Erdos minimum overlap" in prompt
    assert "initial_h_values" in prompt
    assert "Current record: C5 <= 0.38092" in prompt
    assert "Smaller sequences with less than 1k samples are preferred" in prompt
    assert "Make all helper functions top level, no closures or lambdas" in prompt
    assert "0.381000" in prompt
    assert "def run(): pass" in prompt


def test_create_random_initial_state_matches_official_shape_and_bounds():
    state = create_random_initial_state(seed=0)

    assert state.timestep == -1
    assert state.code == ""
    assert state.construction is not None
    assert 40 <= len(state.construction) < 100
    values = np.asarray(state.construction)
    assert np.all(np.isfinite(values))
    assert np.sum(values) == pytest.approx(len(values) / 2.0)
    assert state.value == pytest.approx(-state.raw_score)
