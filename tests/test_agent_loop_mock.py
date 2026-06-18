import pytest
from omegaconf import OmegaConf

from guidance_ttt.agent_loop import (
    GuidanceExecutionAgentLoop,
    _verify_execution_without_fallback,
    build_agent_loop_output,
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
    valid_text = """```python
def run(seed=42, budget_s=1, **kwargs):
    return ([0.5, 0.5], 0.5, 2)
```
<summary>
Outcome hypothesis: baseline
Reusable idea: valid baseline
Risk / possible failure mode: weak score
What future guidance should preserve: validity
What future guidance should change: improve score
</summary>"""

    result = _verify_execution_without_fallback(
        execution_text=valid_text,
        guidance="Keep valid code.",
        timeout_s=20,
    )

    assert result.fallback_used is False
    assert result.execution_text == valid_text
    assert result.verification.valid is True
    assert result.verification.raw_score == 0.5


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
