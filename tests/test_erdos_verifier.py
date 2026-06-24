import math

import numpy as np

from guidance_ttt.tasks.erdos import create_root_node
from guidance_ttt.verifier.erdos import verify_c5_solution, verify_erdos_solution_text


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
    assert result.artifacts["h_values"] == [0.5, 0.5]
    assert result.artifacts["c5_bound"] == 0.5
    assert result.artifacts["n_points"] == 2


def test_valid_nonconstant_solution_can_beat_constant_baseline():
    result = verify_erdos_solution_text(
        """
```python
import numpy as np

def run(seed=42, budget_s=1, **kwargs):
    h = np.array([0.38196601125, 0.61803398875])
    c5 = float(np.max(np.correlate(h, 1 - h, mode="full") * (2.0 / len(h))))
    return (h, c5, len(h))
```
""",
        timeout_s=5,
    )

    assert result.valid is True
    assert result.status == "valid"
    assert result.raw_score < 0.5


def test_invalid_erdos_code_receives_zero_reward():
    result = verify_erdos_solution_text(INVALID_CODE, timeout_s=5)

    assert result.valid is False
    assert result.reward == 0.0
    assert result.status == "invalid"


def test_mass_constraint_violation_is_invalid_instead_of_normalized():
    result = verify_erdos_solution_text(
        """
```python
def run(seed=42, budget_s=1, **kwargs):
    return ([0.25, 0.25], 0.375, 2)
```
""",
        timeout_s=5,
    )

    assert result.valid is False
    assert result.reward == 0.0
    assert result.status == "invalid"
    assert "sum(h)" in result.message


def test_missing_python_code_receives_parse_error():
    result = verify_erdos_solution_text("no code here", timeout_s=5)

    assert result.valid is False
    assert result.reward == 0.0
    assert result.status == "parse_error"


def test_harmony_python_channel_is_extracted_and_verified():
    result = verify_erdos_solution_text(
        """
<execution_thinking>
Use the constant baseline.
assistantcommentary to=python codeimport numpy as np

def run(seed=42, budget_s=1, **kwargs):
    h = np.full(200, 0.5)
    c5 = float(np.max(np.correlate(h, 1 - h, mode="full") * (2.0 / len(h))))
    return (h, c5, len(h))
""",
        timeout_s=5,
    )

    assert result.valid is True
    assert result.status == "valid"
    assert result.raw_score == 0.5


def test_erdos_root_node_stores_valid_random_initial_construction():
    root = create_root_node(seed=123)
    h_values = np.asarray(root.metadata["h_values"], dtype=np.float64)
    n_points = int(root.metadata["n_points"])
    c5_bound = float(root.metadata["c5_bound"])

    verified_raw_score = verify_c5_solution(h_values, c5_bound, n_points)

    assert root.metadata["initialization"] == "random_perturbed_constant"
    assert 40 <= n_points <= 99
    assert len(h_values) == n_points
    assert np.all(np.isfinite(h_values))
    assert np.all((0.0 <= h_values) & (h_values <= 1.0))
    assert math.isclose(float(np.sum(h_values)), n_points / 2.0, abs_tol=1e-8)
    assert root.raw_score == verified_raw_score
    assert root.value == 1.0 / (1e-8 + verified_raw_score)
