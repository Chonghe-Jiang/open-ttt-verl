from __future__ import annotations

from dataclasses import dataclass

import pytest

from guidance_ttt.state import VerificationResult
from guidance_ttt.verifier.frontiercs_adapter import FrontierCSEnvironmentError, FrontierCSResult
from guidance_ttt.verifier.polyomino import extract_cpp_solution_code, verify_polyomino_solution_text


CPP_RESPONSE = """<execution_thinking>
Use shelf packing.
</execution_thinking>

<solution>
```cpp
#include <bits/stdc++.h>
using namespace std;
int main() { return 0; }
```
</solution>

<summary>
Use a shelf heuristic.
</summary>"""


def test_extract_cpp_solution_prefers_solution_tag():
    text = CPP_RESPONSE + "\n```cpp\nint helper_only() { return 1; }\n```"

    code = extract_cpp_solution_code(text)

    assert code is not None
    assert "int main()" in code
    assert "helper_only" not in code


@pytest.mark.parametrize("lang", ["cpp", "c++", "C++"])
def test_extract_cpp_solution_accepts_cpp_fence_languages(lang):
    text = f"<solution>\n```{lang}\nint main() {{ return 0; }}\n```\n</solution>"

    assert extract_cpp_solution_code(text) == "int main() { return 0; }"


def test_extract_cpp_solution_rejects_python_solution():
    text = "<solution>\n```python\ndef run(): return None\n```\n</solution>"

    assert extract_cpp_solution_code(text) is None


def test_polyomino_verifier_maps_frontiercs_score(monkeypatch):
    calls: list[dict[str, object]] = []

    def fake_evaluate_cpp_solution(code: str, *, problem_id: str, config: dict[str, object] | None = None):
        calls.append({"code": code, "problem_id": problem_id, "config": config})
        return FrontierCSResult(valid=True, score=87.25, message="accepted", artifacts={"cases": 70})

    monkeypatch.setattr("guidance_ttt.verifier.polyomino.evaluate_cpp_solution", fake_evaluate_cpp_solution)

    result = verify_polyomino_solution_text(
        CPP_RESPONSE,
        problem_id="0",
        config={"n_cases": 70},
    )

    assert result == VerificationResult(
        reward=87.25,
        raw_score=87.25,
        valid=True,
        status="valid",
        message="accepted",
        artifacts={"code": "#include <bits/stdc++.h>\nusing namespace std;\nint main() { return 0; }", "cases": 70},
    )
    assert calls == [
        {
            "code": "#include <bits/stdc++.h>\nusing namespace std;\nint main() { return 0; }",
            "problem_id": "0",
            "config": {"n_cases": 70},
        }
    ]


def test_polyomino_verifier_maps_invalid_frontiercs_result(monkeypatch):
    def fake_evaluate_cpp_solution(code: str, *, problem_id: str, config: dict[str, object] | None = None):
        return FrontierCSResult(valid=False, score=None, message="wrong answer", artifacts={"stderr": "bad"})

    monkeypatch.setattr("guidance_ttt.verifier.polyomino.evaluate_cpp_solution", fake_evaluate_cpp_solution)

    result = verify_polyomino_solution_text(CPP_RESPONSE)

    assert result.valid is False
    assert result.status == "invalid"
    assert result.reward == 0.0
    assert result.raw_score is None
    assert result.message == "wrong answer"
    assert result.artifacts["stderr"] == "bad"
    assert "int main()" in result.artifacts["code"]


def test_polyomino_verifier_reports_frontiercs_environment_errors(monkeypatch):
    def fake_evaluate_cpp_solution(code: str, *, problem_id: str, config: dict[str, object] | None = None):
        raise FrontierCSEnvironmentError("frontier_cs is not installed")

    monkeypatch.setattr("guidance_ttt.verifier.polyomino.evaluate_cpp_solution", fake_evaluate_cpp_solution)

    result = verify_polyomino_solution_text(CPP_RESPONSE)

    assert result.valid is False
    assert result.status == "environment_error"
    assert result.reward == 0.0
    assert result.raw_score is None
    assert result.message == "frontier_cs is not installed"
    assert "int main()" in result.artifacts["code"]


def test_polyomino_verifier_requires_cpp_solution_block():
    result = verify_polyomino_solution_text("<solution>\n```python\ndef run(): return None\n```\n</solution>")

    assert result.valid is False
    assert result.status == "parse_error"
    assert result.reward == 0.0
    assert result.raw_score is None
    assert "No C++17 code block found" in result.message
