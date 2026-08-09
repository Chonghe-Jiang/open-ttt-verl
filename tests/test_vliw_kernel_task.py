from __future__ import annotations

import hashlib
import json

from guidance_ttt.tasks import get_task_spec
from guidance_ttt.verifier.edgebench_adapter import (
    EdgeBenchResult,
    evaluate_vliw_solution,
    rescale_vliw_cycles,
    vliw_training_reward,
)
from guidance_ttt.verifier.vliw_kernel import extract_vliw_solution_code, verify_vliw_solution_text


def _response(code: str) -> str:
    return f"""<solution>
```python
{code}
```
</solution>

<summary>
Pack independent scalar operations into VLIW bundles.
</summary>"""


def test_vliw_task_spec_exposes_official_baseline_and_minimize_direction():
    spec = get_task_spec("vliw_kernel_optimization")

    assert spec.solution_language == "python"
    assert spec.score_direction == "min"
    assert spec.raw_score_label == "Simulator cycles"
    assert spec.bootstrap_solution is not None
    assert "class KernelBuilder" in spec.bootstrap_solution
    assert "def build_kernel" in spec.bootstrap_solution
    assert len(spec.bootstrap_solution.splitlines()) == 272
    assert "dependency-aware bundle packing" in spec.guidance_mechanism_constraint
    assert "Lower raw cycles" in spec.guidance_objective(None)
    assert spec.create_root_node().raw_score is None


def test_vliw_official_starter_is_pinned_byte_exactly():
    spec = get_task_spec("vliw_kernel_optimization")

    assert hashlib.sha256(spec.bootstrap_solution.encode()).hexdigest() == (
        "fb165d18bf4230b4b91ac889b3eff09f9931f559b151cf11c7890fb53e22c01e"
    )


def test_vliw_rescale_matches_official_log_min_anchors():
    assert rescale_vliw_cycles(None) == 0.0
    assert rescale_vliw_cycles(147734) == 0.0
    assert rescale_vliw_cycles(4475.526541978607) == 0.0
    assert abs(rescale_vliw_cycles(1000.0) - 100.0) < 1e-12
    assert rescale_vliw_cycles(500.0) == 100.0


def test_vliw_dense_reward_is_positive_and_improves_as_cycles_fall():
    assert vliw_training_reward(None) == 0.0
    assert vliw_training_reward(100_000) == 10.0
    assert vliw_training_reward(50_000) == 20.0
    assert vliw_training_reward(1000, mode="official_log_min") == 100.0


def test_vliw_solution_extractor_requires_python_fence():
    code = "class KernelBuilder:\n    def build_kernel(self, *args):\n        self.instrs = []"

    assert extract_vliw_solution_code(_response(code)) == code
    assert extract_vliw_solution_code("<solution>```cpp\nint main(){}\n```</solution>") is None


def test_vliw_verifier_maps_cycles_and_normalized_reward(monkeypatch):
    code = "class KernelBuilder:\n    def build_kernel(self, *args):\n        self.instrs = []"

    def fake_evaluate(solution: str, *, timeout_s: int, config: dict):
        assert solution == code
        assert timeout_s == 700
        assert config == {"provider": "modal_http"}
        return EdgeBenchResult(
            valid=True,
            cycles=2000.0,
            normalized_score=rescale_vliw_cycles(2000.0),
            message="accepted",
            artifacts={"results": [{"correct": True}]},
        )

    monkeypatch.setattr("guidance_ttt.verifier.vliw_kernel.evaluate_vliw_solution", fake_evaluate)
    result = verify_vliw_solution_text(
        _response(code),
        timeout_s=700,
        config={"provider": "modal_http"},
    )

    assert result.valid is True
    assert result.status == "valid"
    assert result.raw_score == 2000.0
    assert result.reward == 500.0
    assert "training reward=500.000000" in result.message
    assert result.artifacts["code"] == code
    assert result.artifacts["results"] == [{"correct": True}]
    assert result.artifacts["official_normalized_score"] == rescale_vliw_cycles(2000.0)
    assert result.artifacts["training_reward_mode"] == "inverse_cycles"


def test_vliw_verifier_rejects_non_kernel_python_without_calling_judge(monkeypatch):
    called = False

    def fake_evaluate(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr("guidance_ttt.verifier.vliw_kernel.evaluate_vliw_solution", fake_evaluate)
    result = verify_vliw_solution_text(_response("print('not a kernel')"), timeout_s=10)

    assert result.status == "parse_error"
    assert result.reward == 0.0
    assert result.raw_score is None
    assert called is False


def test_vliw_http_adapter_forwards_code_timeout_and_auth(monkeypatch):
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
                        "score_cycles": 4321,
                        "results": [{"correct": True}],
                    },
                    "provider": "modal_http",
                    "judge_image": "official-test-image",
                }
            ).encode()

    def fake_urlopen(request, *, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setenv("TEST_VLIW_JUDGE_URL", "https://judge.example.test/evaluate")
    monkeypatch.setenv("TEST_VLIW_JUDGE_TOKEN", "test-bearer-token")
    monkeypatch.setattr(
        "guidance_ttt.verifier.edgebench_adapter.urllib.request.urlopen",
        fake_urlopen,
    )

    result = evaluate_vliw_solution(
        "class KernelBuilder:\n    def build_kernel(self, *args):\n        self.instrs = []",
        timeout_s=120,
        config={
            "provider": "modal_http",
            "endpoint_env": "TEST_VLIW_JUDGE_URL",
            "api_key_env": "TEST_VLIW_JUDGE_TOKEN",
            "runner_timeout_s": 37,
            "request_timeout_s": 45,
            "max_retries": 0,
        },
    )

    assert captured["url"] == "https://judge.example.test/evaluate"
    assert captured["payload"] == {
        "solution": "class KernelBuilder:\n    def build_kernel(self, *args):\n        self.instrs = []",
        "runner_timeout_s": 37,
    }
    assert captured["headers"]["Authorization"] == "Bearer test-bearer-token"
    assert captured["timeout"] == 45
    assert result.valid is True
    assert result.cycles == 4321
    assert result.artifacts["judge_image"] == "official-test-image"
