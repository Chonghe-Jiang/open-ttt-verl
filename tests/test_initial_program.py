import initial_program

from erdos_evolve.verifier import verify_program_result


def test_initial_program_exposes_valid_run_interface():
    result = initial_program.run(seed=0, budget_s=1, n_points=64)
    verification = verify_program_result(result)

    assert verification.valid is True
    assert verification.raw_score is not None
    assert verification.raw_score > 0
    assert verification.reward > 0


def test_initial_program_returns_requested_number_of_points():
    h_values, _c5_bound, n_points = initial_program.run(seed=0, budget_s=1, n_points=128)

    assert n_points == 128
    assert len(h_values) == 128
