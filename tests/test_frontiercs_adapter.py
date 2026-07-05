from __future__ import annotations

import sys
import types

import pytest

from guidance_ttt.verifier.frontiercs_adapter import (
    FrontierCSEnvironmentError,
    evaluate_cpp_solution,
)


def test_frontiercs_adapter_reports_missing_package(monkeypatch):
    original_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "frontier_cs":
            raise ModuleNotFoundError("No module named 'frontier_cs'")
        return original_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "frontier_cs", raising=False)
    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(FrontierCSEnvironmentError, match="frontier_cs is not installed"):
        evaluate_cpp_solution("int main() { return 0; }", problem_id="0")


def test_frontiercs_adapter_maps_successful_result(monkeypatch):
    seen = {}

    class FakeSingleEvaluator:
        def __init__(self, register_cleanup, base_dir=None, judge_url="http://localhost:8081"):
            seen["register_cleanup"] = register_cleanup
            seen["base_dir"] = base_dir
            seen["judge_url"] = judge_url

        def evaluate(self, problem_type, *, problem_id, code):
            assert problem_type == "algorithmic"
            assert problem_id == "0"
            assert "int main()" in code
            return types.SimpleNamespace(success=True, score=91.5, message="ok")

    monkeypatch.setitem(sys.modules, "frontier_cs", types.SimpleNamespace(SingleEvaluator=FakeSingleEvaluator))

    result = evaluate_cpp_solution("int main() { return 0; }", problem_id="0", config={"n_cases": 70})

    assert seen == {"register_cleanup": False, "base_dir": None, "judge_url": "http://localhost:8081"}
    assert result.valid is True
    assert result.score == 91.5
    assert result.message == "ok"
    assert result.artifacts["problem_id"] == "0"
    assert result.artifacts["frontiercs_config"] == {"n_cases": 70}


def test_frontiercs_adapter_passes_base_dir_to_evaluator(monkeypatch, tmp_path):
    seen = {}

    class FakeSingleEvaluator:
        def __init__(self, register_cleanup, base_dir=None, judge_url="http://localhost:8081"):
            seen["register_cleanup"] = register_cleanup
            seen["base_dir"] = base_dir
            seen["judge_url"] = judge_url

        def evaluate(self, problem_type, *, problem_id, code):
            return types.SimpleNamespace(success=True, score=1.0, message="ok")

    monkeypatch.setitem(sys.modules, "frontier_cs", types.SimpleNamespace(SingleEvaluator=FakeSingleEvaluator))

    evaluate_cpp_solution("int main() { return 0; }", problem_id="0", config={"base_dir": str(tmp_path)})

    assert seen["register_cleanup"] is False
    assert seen["base_dir"] == tmp_path
    assert seen["judge_url"] == "http://localhost:8081"


def test_frontiercs_adapter_passes_judge_url_to_evaluator(monkeypatch):
    seen = {}

    class FakeSingleEvaluator:
        def __init__(self, register_cleanup, base_dir=None, judge_url="http://localhost:8081"):
            seen["register_cleanup"] = register_cleanup
            seen["base_dir"] = base_dir
            seen["judge_url"] = judge_url

        def evaluate(self, problem_type, *, problem_id, code):
            return types.SimpleNamespace(success=True, score=1.0, message="ok")

    monkeypatch.setitem(sys.modules, "frontier_cs", types.SimpleNamespace(SingleEvaluator=FakeSingleEvaluator))

    result = evaluate_cpp_solution(
        "int main() { return 0; }",
        problem_id="0",
        config={"judge_url": "http://127.0.0.1:8081", "n_cases": 70},
    )

    assert seen["register_cleanup"] is False
    assert seen["judge_url"] == "http://127.0.0.1:8081"
    assert result.artifacts["frontiercs_config"] == {"n_cases": 70}


def test_frontiercs_adapter_reports_evaluator_runtime_error(monkeypatch):
    class FakeSingleEvaluator:
        def __init__(self, register_cleanup, base_dir=None, judge_url="http://localhost:8081"):
            pass

        def evaluate(self, problem_type, *, problem_id, code):
            raise RuntimeError("go-judge unavailable")

    monkeypatch.setitem(sys.modules, "frontier_cs", types.SimpleNamespace(SingleEvaluator=FakeSingleEvaluator))

    with pytest.raises(FrontierCSEnvironmentError, match="Docker/go-judge"):
        evaluate_cpp_solution("int main() { return 0; }", problem_id="0")
