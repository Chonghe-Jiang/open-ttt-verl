import pytest

from guidance_ttt.llm_client import LLMRequest, MockLLMClient


@pytest.mark.anyio
async def test_mock_llm_returns_execution_code_for_execution_prompt():
    client = MockLLMClient()
    response = await client.complete(
        LLMRequest(
            system="You are the execution model.",
            user="<guidance>try projected search</guidance>",
            model="mock-exec",
            temperature=0.0,
            max_tokens=512,
            metadata={"purpose": "execution"},
        )
    )

    assert "<execution_thinking>" in response.text
    assert "```python" in response.text
    assert response.model == "mock-exec"


@pytest.mark.anyio
async def test_mock_llm_returns_summary_for_summary_prompt():
    client = MockLLMClient()
    response = await client.complete(
        LLMRequest(
            system="You summarize attempts for a TTT library.",
            user="Verifier reward: 1.0",
            model="mock-summary",
            temperature=0.0,
            max_tokens=512,
            metadata={"purpose": "summary"},
        )
    )

    assert "<summary>" in response.text
    assert "Reusable idea" in response.text
