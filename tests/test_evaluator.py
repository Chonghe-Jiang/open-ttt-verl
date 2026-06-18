from pathlib import Path

import evaluator


def test_evaluator_scores_initial_program_positive():
    result = evaluator.evaluate(Path("initial_program.py"))

    assert result.metrics["combined_score"] > 0
    assert result.metrics["c5_score"] > 0
    assert result.metrics["raw_c5"] == result.metrics["c5_score"]
    assert result.metrics["valid"] == 1.0
    assert result.artifacts["status"] == "valid"
    assert "C5 bound" in result.artifacts["message"]


def test_evaluator_returns_zero_score_and_artifacts_for_bad_program(tmp_path):
    bad_program = tmp_path / "bad_program.py"
    bad_program.write_text(
        """
def run(seed=42, budget_s=1, **kwargs):
    return ([float("nan")], 0.5, 1)
"""
    )

    result = evaluator.evaluate(bad_program)

    assert result.metrics["combined_score"] == 0.0
    assert result.metrics["valid"] == 0.0
    assert result.artifacts["status"] == "invalid"
    assert "NaN or inf" in result.artifacts["message"]


def test_cascade_stage1_uses_same_validity_contract():
    result = evaluator.evaluate_stage1(Path("initial_program.py"))

    assert result.metrics["combined_score"] > 0
    assert result.metrics["valid"] == 1.0
