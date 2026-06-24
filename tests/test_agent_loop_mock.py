import pytest
from omegaconf import OmegaConf

from guidance_ttt.agent_loop import (
    EXECUTION_SUMMARY_SECTIONS,
    GuidanceExecutionAgentLoop,
    _verify_execution_without_fallback,
    build_agent_loop_output,
    build_execution_summary,
)
from guidance_ttt.state import VerificationResult


def test_agent_loop_output_trains_only_guidance_tokens():
    output = build_agent_loop_output(
        prompt_ids=[1, 2, 3],
        response_ids=[4, 5],
        response_logprobs=[-0.1, -0.2],
        reward=3.0,
        extra_fields={"solution": "def run(): pass"},
    )

    assert output.prompt_ids == [1, 2, 3]
    assert output.response_ids == [4, 5]
    assert output.response_mask == [1, 1]
    assert output.reward_score == 3.0
    assert output.extra_fields["solution"] == "def run(): pass"


def test_verification_result_for_execution_error_has_zero_reward():
    result = VerificationResult.execution_error("api failed")

    assert result.valid is False
    assert result.reward == 0.0
    assert result.status == "execution_error"
    assert result.message == "api failed"


def test_invalid_execution_is_not_replaced_by_minimax_fallback():
    result = _verify_execution_without_fallback(
        execution_text="",
        guidance="Use deterministic minimax search.",
        timeout_s=20,
    )

    assert result.fallback_used is False
    assert result.original_execution_text == ""
    assert result.verification.valid is False
    assert result.verification.status in {"parse_error", "execution_error"}
    assert result.verification.raw_score is None
    assert result.solution == ""
    assert "project_to_box_sum" not in result.solution


def test_valid_execution_is_kept_without_fallback():
    valid_text = """<execution_thinking>
Use the baseline to confirm verifier plumbing.
</execution_thinking>

<solution>
```python
def run(seed=42, budget_s=1, **kwargs):
    return ([0.5, 0.5], 0.5, 2)
```
</solution>
<summary>
Execution Interpretation
Baseline execution.

Implemented Algorithm
Return the constant profile.

New Ideas Introduced
valid baseline

Empirical Outcome
pending

Failure / Bottleneck Analysis
weak score

Next Guidance Delta
improve score
</summary>"""

    result = _verify_execution_without_fallback(
        execution_text=valid_text,
        guidance="Keep valid code.",
        timeout_s=20,
    )

    assert result.fallback_used is False
    assert result.execution_text == valid_text
    assert result.execution_thinking == "Use the baseline to confirm verifier plumbing."
    assert result.verification.valid is True
    assert result.verification.raw_score == 0.5
    assert "Execution Interpretation" in result.summary
    assert "Use the baseline to confirm verifier plumbing." in result.summary
    assert "Implemented Algorithm" in result.summary
    assert "```python\ndef run(seed=42, budget_s=1, **kwargs):" in result.summary
    assert "Empirical Outcome\nVerifier status: valid\nRaw C5: 0.5" in result.summary
    assert "Verified returned profile: n_points=2, c5_bound=0.5" in result.summary
    assert "head=[0.5, 0.5]" in result.summary
    assert "Next Guidance Delta\nimprove score" in result.summary


def test_solution_tag_is_preferred_when_summary_contains_python_block():
    execution_text = """<execution_thinking>
Use tagged solution.
</execution_thinking>

<solution>
```python
def run(seed=42, budget_s=1, **kwargs):
    return ([0.5, 0.5], 0.5, 2)
```
</solution>

<summary>
Execution Interpretation
The summary includes a non-solution code example.

Implemented Algorithm
```python
def helper_only():
    return None
```

New Ideas Introduced
tagged extraction

Empirical Outcome
pending

Failure / Bottleneck Analysis
none

Next Guidance Delta
continue
</summary>"""

    result = _verify_execution_without_fallback(
        execution_text=execution_text,
        guidance="Use tagged solution.",
        timeout_s=20,
    )

    assert result.verification.valid is True
    assert "def run(seed=42" in result.solution
    assert "helper_only" not in result.solution


def test_build_execution_summary_normalizes_all_sections_and_verifier_outcome():
    verification = VerificationResult(
        reward=2.5,
        raw_score=0.4,
        valid=True,
        status="valid",
        message="C5 bound: 0.400000",
        artifacts={"h_values": [0.25, 0.75], "c5_bound": 0.4, "n_points": 2},
    )

    summary = build_execution_summary(
        model_summary="""Execution Interpretation
model interpretation

Implemented Algorithm
model algorithm

New Ideas Introduced
coordinate repair

Empirical Outcome
model guessed success

Failure / Bottleneck Analysis
weak local basin

Next Guidance Delta
preserve repair and shrink steps""",
        execution_thinking="execution thought",
        solution="def run(seed=42, budget_s=1, **kwargs):\n    return ([0.5, 0.5], 0.5, 2)",
        guidance="try repair",
        verification=verification,
    )

    for section in EXECUTION_SUMMARY_SECTIONS:
        assert section in summary
    assert "Execution Interpretation\nexecution thought" in summary
    assert "model interpretation" in summary
    assert "Implemented Algorithm\nmodel algorithm" in summary
    assert "```python\ndef run(seed=42, budget_s=1, **kwargs):" in summary
    assert "Empirical Outcome\nVerifier status: valid\nRaw C5: 0.4\nReward: 2.5" in summary
    assert "Verified returned profile: n_points=2, c5_bound=0.4" in summary
    assert "model guessed success" not in summary


def test_build_execution_summary_synthesizes_missing_summary_with_code():
    verification = VerificationResult.execution_error("missing run")

    summary = build_execution_summary(
        model_summary=None,
        execution_thinking="I attempted a search.",
        solution="def helper():\n    pass",
        guidance="Use deterministic search.",
        verification=verification,
    )

    for section in EXECUTION_SUMMARY_SECTIONS:
        assert section in summary
    assert "Execution Interpretation\nI attempted a search." in summary
    assert "```python\ndef helper():" in summary
    assert "Empirical Outcome\nVerifier status: execution_error\nRaw C5: None\nReward: 0.0" in summary
    assert "Verifier reported execution_error: missing run" in summary
    assert "Next Guidance Delta\nContinue from the submitted guidance: Use deterministic search." in summary


def test_agent_loop_loads_execution_llm_from_rollout_config_path(tmp_path):
    config_path = tmp_path / "agent_loop.yaml"
    config_path.write_text(
        """
- name: guidance_execution_erdos
  _target_: guidance_ttt.agent_loop.GuidanceExecutionAgentLoop
  execution_llm:
    provider: local
    model: models/gpt-oss-20b
    device_map: auto
"""
    )
    loop = GuidanceExecutionAgentLoop.__new__(GuidanceExecutionAgentLoop)
    loop.rollout_config = OmegaConf.create(
        {
            "agent": {
                "agent_loop_config_path": str(config_path),
                "default_agent_loop": "guidance_execution_erdos",
            }
        }
    )

    execution_llm = loop._execution_llm_from_rollout_config()

    assert execution_llm == {
        "provider": "local",
        "model": "models/gpt-oss-20b",
        "device_map": "auto",
    }


@pytest.mark.anyio
async def test_empty_guidance_generation_retries_with_min_tokens():
    class FakeTokenizer:
        def decode(self, token_ids, skip_special_tokens=True):
            if token_ids == [0]:
                return "" if skip_special_tokens else "<|im_end|>"
            return "<guidance>try coordinate descent</guidance>"

    class FakeServerManager:
        def __init__(self):
            self.calls = []

        async def generate(self, *, request_id, prompt_ids, sampling_params):
            self.calls.append(dict(sampling_params))
            if len(self.calls) == 1:
                return type("Output", (), {"token_ids": [0], "log_probs": [-0.1], "stop_reason": "stop"})()
            return type("Output", (), {"token_ids": [10, 11, 12], "log_probs": [-0.2, -0.3, -0.4], "stop_reason": "length"})()

    loop = GuidanceExecutionAgentLoop.__new__(GuidanceExecutionAgentLoop)
    loop.server_manager = FakeServerManager()
    loop.tokenizer = FakeTokenizer()
    loop.response_length = 128

    generation = await loop._generate_guidance_response([1, 2, 3], {"temperature": 1.0})

    assert generation.text == "<guidance>try coordinate descent</guidance>"
    assert generation.response_ids == [10, 11, 12]
    assert generation.attempts == 2
    assert loop.server_manager.calls[0] == {"temperature": 1.0}
    assert loop.server_manager.calls[1]["min_tokens"] >= 16
    assert loop.server_manager.calls[1]["max_tokens"] <= 128
