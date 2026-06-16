import textwrap

from erdos_evolve.verifier import verify_c5_solution, verify_erdos_program_text


VALID_CODE = """
def run(seed=42, budget_s=1, **kwargs):
    return ([0.5, 0.5], 0.5, 2)
"""


def test_verify_c5_solution_accepts_valid_baseline():
    raw_score = verify_c5_solution([0.5, 0.5], 0.5, 2)

    assert raw_score == 0.5


def test_program_text_receives_positive_reward_for_valid_solution():
    result = verify_erdos_program_text(VALID_CODE, timeout_s=5)

    assert result.valid is True
    assert result.status == "valid"
    assert result.reward > 0
    assert result.raw_score == 0.5


def test_program_text_rejects_nan_values():
    result = verify_erdos_program_text(
        """
def run(seed=42, budget_s=1, **kwargs):
    return ([float("nan")], 0.5, 1)
""",
        timeout_s=5,
    )

    assert result.valid is False
    assert result.reward == 0.0
    assert result.status == "invalid"
    assert "NaN or inf" in result.message


def test_program_text_rejects_wrong_c5_report():
    result = verify_erdos_program_text(
        """
def run(seed=42, budget_s=1, **kwargs):
    return ([0.5, 0.5], 0.25, 2)
""",
        timeout_s=5,
    )

    assert result.valid is False
    assert result.reward == 0.0
    assert result.status == "invalid"
    assert "C5 mismatch" in result.message


def test_program_text_rejects_missing_run_function():
    result = verify_erdos_program_text("x = 1", timeout_s=5)

    assert result.valid is False
    assert result.reward == 0.0
    assert result.status == "invalid"
    assert "Program must define run" in result.message


def test_program_text_times_out():
    result = verify_erdos_program_text(
        textwrap.dedent(
            """
            def run(seed=42, budget_s=1, **kwargs):
                while True:
                    pass
            """
        ),
        timeout_s=1,
    )

    assert result.valid is False
    assert result.reward == 0.0
    assert result.status == "timeout"
