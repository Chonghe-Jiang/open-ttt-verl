from guidance_ttt.verifier.erdos import verify_erdos_solution_text


VALID_CODE = """
```python
def run(seed=42, budget_s=1, **kwargs):
    return ([0.5, 0.5], 0.5, 2)
```
"""


INVALID_CODE = """
```python
def run(seed=42, budget_s=1, **kwargs):
    return ([float("nan")], 0.5, 1)
```
"""


def test_valid_erdos_code_receives_positive_reward():
    result = verify_erdos_solution_text(VALID_CODE, timeout_s=5)

    assert result.valid is True
    assert result.status == "valid"
    assert result.reward > 0
    assert result.raw_score == 0.5


def test_invalid_erdos_code_receives_zero_reward():
    result = verify_erdos_solution_text(INVALID_CODE, timeout_s=5)

    assert result.valid is False
    assert result.reward == 0.0
    assert result.status == "invalid"


def test_missing_python_code_receives_parse_error():
    result = verify_erdos_solution_text("no code here", timeout_s=5)

    assert result.valid is False
    assert result.reward == 0.0
    assert result.status == "parse_error"

