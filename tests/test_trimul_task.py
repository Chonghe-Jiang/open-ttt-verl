from __future__ import annotations

import hashlib
import io
import json
import urllib.error
from pathlib import Path

import pytest

from guidance_ttt.bootstrap import build_bootstrap_execution_prompt
from guidance_ttt.prompts import build_execution_prompt, build_guidance_prompt
from guidance_ttt.state import LibraryEntry
from guidance_ttt.tasks import get_task_spec
from guidance_ttt.tasks.trimul import TRIMUL_BASELINE_SOLUTION, TRIMUL_BASELINE_SUMMARY
from guidance_ttt.tasks.trimul_prompt import TRIMUL_PROMPT
from guidance_ttt.verifier.trimul import extract_trimul_solution_code, verify_trimul_solution_text
from guidance_ttt.verifier.trimul_adapter import (
    TriMulEnvironmentError,
    TriMulResult,
    evaluate_trimul_solution,
)
from guidance_ttt.verifier.trimul_official_runner import (
    build_case_file,
    geometric_mean_runtime_us,
)

EVALUATOR_HASHES = {
    "task.yml": "77a36e2e74141dccb35341f54cc18731504687dcaf80e15f5344c0ba14b73be9",
    "task.py": "856c8f04c06fc4b0768fca5e267c34f1884355054f89c629a6aac07b3c099174",
    "utils.py": "a8a6a725c4fcc3d5e9cd50b9299a77d9f8a39e41830a994481171033f95b2681",
    "reference.py": "7fd9dfc1de86cd41063b50584ad0e214edd43285fc080e767d921ba86e1024ae",
    "eval.py": "94f5b0a0ff72e797c0c83c30478697a897dc2ac369f8cc8e07dbcd0272803f51",
}


def _response(code: str) -> str:
    return f"""<solution>
```python
{code}
```
</solution>

<summary>
Fuse one stage while retaining the official custom_kernel interface.
</summary>"""


def test_trimul_task_uses_original_prompt_without_implicit_discover_seed():
    spec = get_task_spec("trimul")

    assert hashlib.sha256(TRIMUL_PROMPT.encode()).hexdigest() == (
        "b9add300e4bb518525701c9d2972b555675425bfe0d9da61a90e9d5a8b64c6f6"
    )
    assert spec.problem_prompt.startswith(TRIMUL_PROMPT)
    assert "You must use trition 3.3.1" in spec.problem_prompt
    assert spec.solution_language == "python"
    assert spec.score_direction == "min"
    assert spec.raw_score_label == "H100 geometric-mean runtime (us)"
    assert spec.bootstrap_solution is None
    assert spec.bootstrap_summary is None
    assert spec.scratch_bootstrap_constraint is not None
    assert "Prioritize correctness and Triton compilability" in spec.scratch_bootstrap_constraint
    assert "torch.einsum" in spec.scratch_bootstrap_constraint
    assert 'torch.einsum("bikd,bjkd->bijd", left, right)' in spec.scratch_bootstrap_constraint
    assert "must not be followed by a compensating reshape" in spec.scratch_bootstrap_constraint
    assert "exactly one simple, functionally required Triton kernel" in spec.scratch_bootstrap_constraint
    assert "flat unit-stride offsets" in spec.scratch_bootstrap_constraint
    assert "@triton.jit" in TRIMUL_BASELINE_SOLUTION
    assert "def custom_kernel" in TRIMUL_BASELINE_SOLUTION
    assert TRIMUL_BASELINE_SUMMARY
    assert spec.create_root_node().raw_score is None


def test_trimul_scratch_bootstrap_prompt_contains_no_discover_candidate():
    spec = get_task_spec("trimul")

    prompt = build_bootstrap_execution_prompt(
        task_spec=spec,
        execution_prompt_style="gpt_api_brief_thinking",
        prompt_mode="summary_only",
        bootstrap_source="scratch",
    )

    assert spec.problem_prompt in prompt.user
    assert "There is no parent candidate, prior solution" in prompt.user
    assert "solely from the public problem statement" in prompt.user
    assert "<parent_code>" not in prompt.user
    assert "<selected_summary>" not in prompt.user
    assert "<guidance>" not in prompt.user
    assert "<baseline_candidate>" not in prompt.user
    assert "<baseline_code>" not in prompt.user
    assert TRIMUL_BASELINE_SOLUTION not in prompt.user
    assert TRIMUL_BASELINE_SUMMARY not in prompt.user
    assert "Task-specific scratch-seed constraints:" in prompt.user
    assert "Do not write a custom contraction kernel" in prompt.user
    assert "flatten them" in prompt.user
    assert "verified baseline" not in prompt.system
    assert "scratch seed" in prompt.system


def test_trimul_summary_only_hides_parent_code_from_guidance_but_keeps_it_for_execution():
    spec = get_task_spec("trimul")
    node = spec.create_root_node(raw_score=1250.0, reward=1.2)
    entry = LibraryEntry(
        id="trimul-seed",
        parent_id=node.id,
        problem_id="trimul",
        timestep=0,
        guidance="bootstrap",
        execution_thinking="",
        solution=TRIMUL_BASELINE_SOLUTION,
        verifier_reward=1.2,
        verifier_raw_score=1250.0,
        verifier_status="valid",
        verifier_message="accepted",
        summary="canonical wrapper",
        reusable_idea="preserve fused projection and contraction",
        failure_mode=None,
        metadata={
            "bootstrap": True,
            "prompt_mode": "summary_only",
            "summary_semantics": "canonical_full_candidate",
            "raw_model_summary": "Fuse projection and gating, then use a Triton contraction kernel.",
        },
    )

    guidance_prompt = build_guidance_prompt(
        problem_prompt=spec.problem_prompt,
        selected_node=node,
        selected_entry=entry,
        global_best_entries=[],
        local_failure_entries=[],
        objective_text=spec.guidance_objective(None),
        mechanism_constraint=spec.guidance_mechanism_constraint,
        raw_score_label=spec.raw_score_label,
        solution_language=spec.solution_language,
        prompt_mode="summary_only",
    )
    execution_prompt = build_execution_prompt(
        problem_prompt=spec.problem_prompt,
        selected_node=node,
        selected_entry=entry,
        guidance="Reduce launch overhead while preserving numerical behavior.",
        solution_language=spec.solution_language,
        solution_contract=spec.execution_solution_contract,
        score_direction=spec.score_direction,
        raw_score_label=spec.raw_score_label,
        prompt_mode="summary_only",
        execution_prompt_style="gpt_api_brief_thinking",
    )

    assert "<selected_summary>" in guidance_prompt.user
    assert "Fuse projection and gating" in guidance_prompt.user
    assert "H100 geometric-mean runtime (us): 1250.0" in guidance_prompt.user
    assert "<parent_code>" not in guidance_prompt.user
    assert TRIMUL_BASELINE_SOLUTION not in guidance_prompt.user
    assert "Scope each attempt to one primary optimization mechanism" in guidance_prompt.user
    assert "preserve every unrelated stage of the verified parent" in guidance_prompt.user
    assert "<parent_code>" in execution_prompt.user
    assert TRIMUL_BASELINE_SOLUTION in execution_prompt.user
    assert "Write a concise natural-language summary of the candidate" in execution_prompt.user
    assert "Do not re-summarize the complete algorithm" not in execution_prompt.user


def test_trimul_official_evaluator_assets_are_pinned_verbatim():
    asset_dir = Path("guidance_ttt/tasks/assets/trimul_evaluator")
    for name, expected_hash in EVALUATOR_HASHES.items():
        assert hashlib.sha256((asset_dir / name).read_bytes()).hexdigest() == expected_hash


def test_trimul_case_serialization_matches_libkernelbot():
    assert build_case_file(
        [{"seqlen": 32, "bs": 1, "nomask": True, "distribution": "normal"}]
    ) == "seqlen: 32; bs: 1; nomask: True; distribution: normal\n"


def test_trimul_geometric_mean_is_computed_in_microseconds():
    score = geometric_mean_runtime_us([{"mean_us": 100.0}, {"mean_us": 400.0}])
    assert score == pytest.approx(200.0)


def test_trimul_solution_parser_and_prechecks(monkeypatch):
    valid_code = """import triton
import triton.language as tl
@triton.jit
def kernel(x):
    pass
def custom_kernel(data):
    return data[0]
"""
    assert extract_trimul_solution_code(_response(valid_code)) == valid_code.strip()

    called = False

    def fake_evaluate(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr("guidance_ttt.verifier.trimul.evaluate_trimul_solution", fake_evaluate)
    missing_triton = verify_trimul_solution_text(
        _response("def custom_kernel(data):\n    return data[0]"), timeout_s=10
    )
    assert missing_triton.status == "parse_error"
    assert called is False

    identity = verify_trimul_solution_text(
        _response(valid_code + "\nidentity = True"), timeout_s=10
    )
    assert identity.status == "parse_error"
    assert called is False


def test_trimul_verifier_uses_discover_reward_direction(monkeypatch):
    code = """import triton
@triton.jit
def kernel(x):
    pass
def custom_kernel(data):
    return data[0]
"""

    def fake_evaluate(solution: str, *, timeout_s: int, config: dict):
        assert solution == code.strip()
        assert timeout_s == 1200
        assert config == {"provider": "modal_http", "reward_scale": 1500.0}
        return TriMulResult(
            valid=True,
            score_us=750.0,
            message="accepted",
            artifacts={"benchmarks": [{"mean_us": 750.0}]},
        )

    monkeypatch.setattr("guidance_ttt.verifier.trimul.evaluate_trimul_solution", fake_evaluate)
    result = verify_trimul_solution_text(
        _response(code),
        timeout_s=1200,
        config={"provider": "modal_http", "reward_scale": 1500.0},
    )

    assert result.valid is True
    assert result.raw_score == 750.0
    assert result.reward == 2.0
    assert result.artifacts["reward_formula"] == "reward_scale / geometric_mean_runtime_us"


def test_trimul_adapter_is_remote_only():
    with pytest.raises(TriMulEnvironmentError, match="remote-only"):
        evaluate_trimul_solution("pass", timeout_s=10, config={"provider": "local"})


def test_trimul_http_adapter_forwards_payload_and_auth(monkeypatch):
    captured: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {
                    "report": {
                        "all_correct": True,
                        "score_us": 625.0,
                        "ranking_by": "geom",
                        "benchmarks": [],
                    },
                    "provider": "modal_http",
                    "evaluation_gpu": "NVIDIA H100 80GB HBM3",
                }
            ).encode()

    def fake_urlopen(request, *, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setenv("TEST_TRIMUL_URL", "https://judge.example.test/evaluate")
    monkeypatch.setenv("TEST_TRIMUL_TOKEN", "test-token")
    monkeypatch.setattr("guidance_ttt.verifier.trimul_adapter.urllib.request.urlopen", fake_urlopen)

    result = evaluate_trimul_solution(
        "def custom_kernel(data): pass",
        timeout_s=1200,
        config={
            "provider": "modal_http",
            "endpoint_env": "TEST_TRIMUL_URL",
            "api_key_env": "TEST_TRIMUL_TOKEN",
            "runner_timeout_s": 520,
            "request_timeout_s": 1150,
            "max_retries": 0,
        },
    )

    assert captured["url"] == "https://judge.example.test/evaluate"
    assert captured["payload"] == {
        "solution": "def custom_kernel(data): pass",
        "runner_timeout_s": 520,
    }
    assert captured["headers"]["Authorization"] == "Bearer test-token"
    assert captured["timeout"] == 1150
    assert result.valid is True
    assert result.score_us == 625.0


def test_trimul_http_adapter_retries_transient_408(monkeypatch):
    attempts = 0

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {
                    "report": {
                        "all_correct": True,
                        "score_us": 600.0,
                        "ranking_by": "geom",
                    }
                }
            ).encode()

    def fake_urlopen(request, *, timeout):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise urllib.error.HTTPError(
                request.full_url,
                408,
                "cold-start request timeout",
                {},
                io.BytesIO(b""),
            )
        return FakeResponse()

    monkeypatch.setenv("TEST_TRIMUL_URL", "https://judge.example.test/evaluate")
    monkeypatch.setattr("guidance_ttt.verifier.trimul_adapter.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("guidance_ttt.verifier.trimul_adapter.time.sleep", lambda _: None)

    result = evaluate_trimul_solution(
        "def custom_kernel(data): pass",
        timeout_s=1200,
        config={
            "provider": "modal_http",
            "endpoint_env": "TEST_TRIMUL_URL",
            "request_timeout_s": 1150,
            "max_retries": 2,
        },
    )

    assert attempts == 2
    assert result.valid is True
    assert result.score_us == 600.0
    assert result.artifacts["http_attempts"] == 2
