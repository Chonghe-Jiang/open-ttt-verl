import pytest

from guidance_ttt.llm_client import LLMRequest, MockLLMClient, OpenAICompatibleLLMClient, make_llm_client


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
    assert "<summary>" in response.text
    assert response.model == "mock-exec"


@pytest.mark.anyio
async def test_openai_compatible_client_maps_chat_completion(monkeypatch):
    monkeypatch.setenv("ENDPOINT", "https://llm.example/v1")
    monkeypatch.setenv("API_KEY", "test-key")
    client = make_llm_client({"provider": "openai_compatible"})
    assert isinstance(client, OpenAICompatibleLLMClient)

    captured = {}

    def fake_post_json(path, payload):
        captured["path"] = path
        captured["payload"] = payload
        return {
            "model": "api-model",
            "choices": [
                {
                    "message": {"content": "<execution_thinking>x</execution_thinking>\n```python\npass\n```"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        }

    monkeypatch.setattr(client, "_post_json", fake_post_json)
    response = await client.complete(
        LLMRequest(
            system="system prompt",
            user="user prompt",
            model="executor-model",
            temperature=0.2,
            max_tokens=128,
            metadata={"purpose": "execution"},
        )
    )

    assert captured["path"] == "/chat/completions"
    assert captured["payload"]["model"] == "executor-model"
    assert captured["payload"]["messages"] == [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "user prompt"},
    ]
    assert response.text.startswith("<execution_thinking>")
    assert response.model == "api-model"
    assert response.finish_reason == "stop"
