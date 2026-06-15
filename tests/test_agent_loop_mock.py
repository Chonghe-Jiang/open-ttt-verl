from guidance_ttt.agent_loop import build_agent_loop_output
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

