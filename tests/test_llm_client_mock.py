import asyncio
import http.client
import json
import os
import time

import pytest

import guidance_ttt.llm_client as llm_client_module
from guidance_ttt.llm_client import (
    GPTOSSTwoPhaseOpenAICompatibleLLMClient,
    LLMRequest,
    LocalTransformersLLMClient,
    LocalVLLMLLMClient,
    MockLLMClient,
    OpenAICompatibleLLMClient,
    _extract_local_generated_text,
    _load_local_text_generation_pipeline,
    make_llm_client,
)


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


@pytest.mark.anyio
async def test_openai_compatible_client_forwards_qwen_options_and_reasoning(monkeypatch):
    client = OpenAICompatibleLLMClient(
        {
            "base_url": "http://127.0.0.1:8000/v1",
            "api_key": "local-vllm",
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "chat_template_kwargs": {"enable_thinking": True},
            "reasoning": {"effort": "high"},
            "reasoning_effort": "low",
            "verbosity": "low",
        }
    )
    captured = {}

    def fake_post_json(path, payload):
        captured["path"] = path
        captured["payload"] = payload
        return {
            "model": "Qwen/Qwen3-8B",
            "choices": [
                {
                    "message": {
                        "reasoning": "Consider several skyline mutations.",
                        "content": "<solution>```cpp\nint main(){return 0;}\n```</solution><summary>delta</summary>",
                    },
                    "finish_reason": "stop",
                }
            ],
        }

    monkeypatch.setattr(client, "_post_json", fake_post_json)
    response = await client.complete(
        LLMRequest(
            system="system",
            user="user",
            model="Qwen/Qwen3-8B",
            temperature=0.6,
            max_tokens=16384,
        )
    )

    assert captured["path"] == "/chat/completions"
    assert captured["payload"]["top_p"] == 0.95
    assert captured["payload"]["top_k"] == 20
    assert captured["payload"]["min_p"] == 0.0
    assert captured["payload"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert captured["payload"]["reasoning"] == {"effort": "high"}
    assert captured["payload"]["reasoning_effort"] == "low"
    assert captured["payload"]["verbosity"] == "low"
    assert response.reasoning == "Consider several skyline mutations."
    assert response.text.startswith("<solution>")


@pytest.mark.anyio
async def test_openai_compatible_client_records_request_metrics_and_upstream_provider(monkeypatch):
    client = OpenAICompatibleLLMClient(
        {
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "test-key",
            "capture_request_metrics": True,
        }
    )

    def fake_post_json(path, payload):
        return {
            "model": "z-ai/glm-5.2",
            "provider": "Z.AI",
            "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
            "_client_request_metadata": {
                "api_attempts": 2,
                "api_retry_count": 1,
                "api_elapsed_s": 3.5,
                "api_http_status": 200,
            },
        }

    monkeypatch.setattr(client, "_post_json", fake_post_json)
    response = await client.complete(
        LLMRequest(
            system="system",
            user="user",
            model="z-ai/glm-5.2",
            temperature=1.0,
            max_tokens=None,
        )
    )

    assert response.metadata["api_response_provider"] == "Z.AI"
    assert response.metadata["api_response_model"] == "z-ai/glm-5.2"
    assert response.metadata["api_attempts"] == 2
    assert response.metadata["api_retry_count"] == 1
    assert response.metadata["api_elapsed_s"] == 3.5


class _FakeHTTPResponse:
    status = 200

    def __init__(self, body: bytes = b"", read_error: Exception | None = None):
        self.body = body
        self.read_error = read_error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        if self.read_error is not None:
            raise self.read_error
        return self.body


def _successful_chat_body() -> bytes:
    return json.dumps(
        {
            "model": "z-ai/glm-5.2",
            "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
        }
    ).encode()


def test_openai_compatible_client_retries_incomplete_read(monkeypatch):
    client = OpenAICompatibleLLMClient(
        {
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "test-key",
            "max_retries": 2,
            "retry_backoff_s": 0,
            "capture_request_metrics": True,
        }
    )
    responses = iter(
        [
            _FakeHTTPResponse(read_error=http.client.IncompleteRead(b'{"partial":', 100)),
            _FakeHTTPResponse(_successful_chat_body()),
        ]
    )
    monkeypatch.setattr(llm_client_module.urllib.request, "urlopen", lambda *args, **kwargs: next(responses))

    data = client._post_json("/chat/completions", {"model": "z-ai/glm-5.2"})

    assert data["choices"][0]["message"]["content"] == "OK"
    assert data["_client_request_metadata"]["api_attempts"] == 2
    assert data["_client_request_metadata"]["api_retry_errors"] == ["IncompleteRead"]


def test_openai_compatible_client_retries_truncated_json_and_error_finish(monkeypatch):
    client = OpenAICompatibleLLMClient(
        {
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "test-key",
            "max_retries": 3,
            "retry_backoff_s": 0,
            "capture_request_metrics": True,
        }
    )
    error_finish = json.dumps(
        {
            "model": "z-ai/glm-5.2",
            "choices": [{"message": {"reasoning": "unfinished"}, "finish_reason": "error"}],
        }
    ).encode()
    responses = iter(
        [
            _FakeHTTPResponse(b'{"choices": ['),
            _FakeHTTPResponse(error_finish),
            _FakeHTTPResponse(_successful_chat_body()),
        ]
    )
    monkeypatch.setattr(llm_client_module.urllib.request, "urlopen", lambda *args, **kwargs: next(responses))

    data = client._post_json("/chat/completions", {"model": "z-ai/glm-5.2"})

    assert data["choices"][0]["message"]["content"] == "OK"
    assert data["_client_request_metadata"]["api_attempts"] == 3
    assert data["_client_request_metadata"]["api_retry_errors"] == [
        "JSONDecodeError",
        "finish_reason_error",
    ]


@pytest.mark.parametrize(
    ("bad_response", "expected_reason"),
    [
        ({"model": "z-ai/glm-5.2"}, "missing_choices"),
        (
            {
                "model": "z-ai/glm-5.2",
                "choices": [{"message": {"reasoning": "unfinished"}, "finish_reason": None}],
            },
            "missing_finish_reason",
        ),
        (
            {
                "model": "z-ai/glm-5.2",
                "choices": [{"message": {"reasoning": "unfinished"}, "finish_reason": "stop"}],
            },
            "missing_final_content",
        ),
    ],
)
def test_openai_compatible_client_retries_incomplete_chat_response(
    monkeypatch, bad_response, expected_reason
):
    client = OpenAICompatibleLLMClient(
        {
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "test-key",
            "max_retries": 1,
            "retry_backoff_s": 0,
            "capture_request_metrics": True,
        }
    )
    responses = iter(
        [
            _FakeHTTPResponse(json.dumps(bad_response).encode()),
            _FakeHTTPResponse(_successful_chat_body()),
        ]
    )
    monkeypatch.setattr(llm_client_module.urllib.request, "urlopen", lambda *args, **kwargs: next(responses))

    data = client._post_json("/chat/completions", {"model": "z-ai/glm-5.2"})

    assert data["choices"][0]["message"]["content"] == "OK"
    assert data["_client_request_metadata"]["api_retry_errors"] == [expected_reason]


@pytest.mark.anyio
async def test_gptoss_two_phase_client_forces_final_with_remaining_context(monkeypatch):
    client = make_llm_client(
        {
            "provider": "openai_compatible",
            "base_url": "http://127.0.0.1:8000/v1",
            "api_key": "local-vllm",
            "reasoning_effort": "high",
            "phase1_max_tokens": 10,
            "context_window": 20,
            "context_buffer": 1,
        }
    )
    assert isinstance(client, GPTOSSTwoPhaseOpenAICompatibleLLMClient)

    monkeypatch.setattr(client, "_render_harmony_prompt", lambda request: [1, 2])
    monkeypatch.setattr(client, "_harmony_stop_token_ids", lambda: [99])

    def fake_encode(text):
        if text == client.GPTOSS_FINAL_CHANNEL_INDICATOR:
            return [70]
        if text == "<|end|>":
            return [71]
        return [80, 81]

    monkeypatch.setattr(client, "_encode_harmony", fake_encode)
    monkeypatch.setattr(
        client,
        "_parse_harmony_output",
        lambda token_ids: ("long reasoning", "<solution>code</solution>"),
    )
    calls = []

    def fake_post_json(path, payload):
        calls.append((path, payload))
        if len(calls) == 1:
            return {
                "model": "openai/gpt-oss-120b",
                "choices": [{"token_ids": list(range(10, 18)), "finish_reason": "length"}],
            }
        return {
            "model": "openai/gpt-oss-120b",
            "choices": [{"token_ids": [90, 91, 99], "finish_reason": "stop"}],
        }

    monkeypatch.setattr(client, "_post_json", fake_post_json)
    response = await client.complete(
        LLMRequest(
            system="system",
            user="user",
            model="openai/gpt-oss-120b",
            temperature=0.0,
            max_tokens=None,
            metadata={"purpose": "execution"},
        )
    )

    assert [path for path, _payload in calls] == ["/completions", "/completions"]
    assert calls[0][1]["prompt"] == [1, 2]
    assert calls[0][1]["max_tokens"] == 8
    assert calls[0][1]["return_token_ids"] is True
    assert calls[1][1]["prompt"] == [1, 2] + list(range(10, 18)) + [80, 81]
    assert calls[1][1]["max_tokens"] == 7
    assert response.text == "<solution>code</solution>"
    assert response.reasoning == "long reasoning"
    assert response.finish_reason == "stop"
    assert response.usage == {
        "prompt_tokens": 2,
        "completion_tokens": 13,
        "total_tokens": 15,
        "phase1_completion_tokens": 8,
        "phase1_discarded_header_tokens": 0,
        "phase2_prefill_tokens": 2,
        "phase2_completion_tokens": 3,
    }
    assert response.metadata["forced_final"] is True
    assert response.metadata["phase1_finish_reason"] == "length"
    assert response.metadata["phase2_finish_reason"] == "stop"
    assert response.metadata["harmony_parse_fallback"] is False


def test_gptoss_two_phase_trims_partial_final_channel_header(monkeypatch):
    client = GPTOSSTwoPhaseOpenAICompatibleLLMClient(
        {
            "base_url": "http://127.0.0.1:8000/v1",
            "api_key": "local-vllm",
            "phase1_max_tokens": 10,
            "context_window": 20,
            "context_buffer": 1,
        }
    )
    encodings = {
        client.GPTOSS_FINAL_MARKER: [7, 8, 9, 10, 11, 12],
        "<|start|>assistant<|channel|>final<|message|>": [8, 9, 10, 11, 12],
        client.GPTOSS_FINAL_CHANNEL_INDICATOR: [10, 11, 12],
    }
    monkeypatch.setattr(client, "_encode_harmony", lambda text: encodings[text])

    kept, discarded = client._trim_incomplete_final_header([1, 2, 7, 8, 9, 10])

    assert kept == [1, 2]
    assert discarded == [7, 8, 9, 10]


def test_gptoss_two_phase_harmony_error_falls_back_to_phase2_text(monkeypatch):
    client = GPTOSSTwoPhaseOpenAICompatibleLLMClient(
        {
            "base_url": "http://127.0.0.1:8000/v1",
            "api_key": "local-vllm",
            "phase1_max_tokens": 10,
            "context_window": 20,
            "context_buffer": 1,
        }
    )

    class HarmonyError(RuntimeError):
        pass

    monkeypatch.setattr(
        client,
        "_parse_harmony_output",
        lambda _tokens: (_ for _ in ()).throw(HarmonyError("bad header")),
    )
    monkeypatch.setattr(
        client,
        "_best_effort_reasoning",
        lambda _tokens: "truncated reasoning",
    )
    monkeypatch.setattr(client, "_decode_harmony", lambda tokens: "final answer")

    reasoning, text, fallback = client._parse_harmony_output_resilient(
        combined_ids=[1, 2, 3],
        phase1_ids=[1],
        phase2_ids=[3],
        forced_final=True,
        injected_prefill=True,
    )

    assert reasoning == "truncated reasoning"
    assert text == "final answer"
    assert fallback is True


@pytest.mark.anyio
async def test_gptoss_two_phase_client_keeps_natural_phase1_completion(monkeypatch):
    client = GPTOSSTwoPhaseOpenAICompatibleLLMClient(
        {
            "base_url": "http://127.0.0.1:8000/v1",
            "api_key": "local-vllm",
            "reasoning_effort": "high",
            "phase1_max_tokens": 10,
            "context_window": 20,
            "context_buffer": 1,
        }
    )
    monkeypatch.setattr(client, "_render_harmony_prompt", lambda request: [1, 2])
    monkeypatch.setattr(client, "_harmony_stop_token_ids", lambda: [99])
    monkeypatch.setattr(
        client,
        "_parse_harmony_output",
        lambda token_ids: ("brief reasoning", "final answer"),
    )
    calls = []

    def fake_post_json(path, payload):
        calls.append((path, payload))
        return {
            "model": "openai/gpt-oss-120b",
            "choices": [{"token_ids": [10, 11, 99], "finish_reason": "stop"}],
        }

    monkeypatch.setattr(client, "_post_json", fake_post_json)
    response = await client.complete(
        LLMRequest(
            system="system",
            user="user",
            model="openai/gpt-oss-120b",
            temperature=0.0,
            max_tokens=None,
        )
    )

    assert len(calls) == 1
    assert response.text == "final answer"
    assert response.metadata["forced_final"] is False
    assert response.usage["phase2_completion_tokens"] == 0


@pytest.mark.anyio
async def test_openai_compatible_client_can_read_api_key_from_named_env(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-test-key")
    client = make_llm_client(
        {
            "provider": "openai_compatible",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_API_KEY",
        }
    )

    assert isinstance(client, OpenAICompatibleLLMClient)
    assert client.endpoint == "https://openrouter.ai/api/v1"
    assert client.api_key == "openrouter-test-key"


def test_openai_compatible_client_can_read_api_key_file(tmp_path, monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    key_path = tmp_path / "api_key"
    key_path.write_text("file-test-key\n")

    client = OpenAICompatibleLLMClient(
        {
            "base_url": "https://llm.example/v1",
            "api_key_file": str(key_path),
        }
    )

    assert client.api_key == "file-test-key"


def test_openai_compatible_client_accepts_request_timeout(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-key")
    client = make_llm_client(
        {
            "provider": "openai_compatible",
            "base_url": "http://127.0.0.1:8000/v1",
            "timeout_s": 600,
        }
    )

    assert isinstance(client, OpenAICompatibleLLMClient)
    assert client.timeout_s == 600


@pytest.mark.anyio
async def test_openai_compatible_client_forwards_enable_thinking(monkeypatch):
    client = OpenAICompatibleLLMClient(
        {
            "base_url": "https://llm.example/v1",
            "api_key": "test-key",
            "enable_thinking": False,
        }
    )
    captured = {}

    def fake_post_json(path, payload):
        captured["path"] = path
        captured["payload"] = payload
        return {
            "model": "glm-5.2",
            "choices": [{"message": {"content": "done"}, "finish_reason": "stop"}],
            "usage": {},
        }

    monkeypatch.setattr(client, "_post_json", fake_post_json)
    await client.complete(
        LLMRequest(
            system="system",
            user="user",
            model="glm-5.2",
            temperature=0.0,
            max_tokens=8192,
        )
    )

    assert captured["path"] == "/chat/completions"
    assert captured["payload"]["enable_thinking"] is False
    assert captured["payload"]["max_tokens"] == 8192


@pytest.mark.anyio
async def test_local_transformers_client_loads_model_lazily_and_uses_chat_messages(monkeypatch):
    captured = {}

    def fake_load_pipeline(config):
        captured["load_config"] = config

        def fake_pipeline(messages, **generation_kwargs):
            captured["messages"] = messages
            captured["generation_kwargs"] = generation_kwargs
            return [
                {
                    "generated_text": [
                        {"role": "system", "content": "system prompt"},
                        {"role": "user", "content": "user prompt"},
                        {
                            "role": "assistant",
                            "content": "<execution_thinking>local</execution_thinking>\n```python\npass\n```",
                        },
                    ]
                }
            ]

        return fake_pipeline

    monkeypatch.setattr("guidance_ttt.llm_client._load_local_text_generation_pipeline", fake_load_pipeline)
    client = make_llm_client(
        {
            "provider": "local",
            "model": "models/gpt-oss-20b",
            "device_map": "auto",
            "torch_dtype": "auto",
            "trust_remote_code": True,
            "reasoning_effort": "low",
        }
    )

    assert isinstance(client, LocalTransformersLLMClient)
    assert captured == {}

    response = await client.complete(
        LLMRequest(
            system="system prompt",
            user="user prompt",
            model="models/gpt-oss-20b",
            temperature=0.4,
            max_tokens=256,
            metadata={"purpose": "execution"},
        )
    )

    assert captured["load_config"]["model"] == "models/gpt-oss-20b"
    assert captured["load_config"]["device_map"] == "auto"
    assert captured["messages"] == [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "user prompt"},
    ]
    assert captured["generation_kwargs"]["max_new_tokens"] == 256
    assert captured["generation_kwargs"]["temperature"] == 0.4
    assert captured["generation_kwargs"]["do_sample"] is True
    assert captured["generation_kwargs"]["tokenizer_encode_kwargs"]["reasoning_effort"] == "low"
    assert response.text.startswith("<execution_thinking>local</execution_thinking>")
    assert response.model == "models/gpt-oss-20b"
    assert response.metadata["provider"] == "local_transformers"


@pytest.mark.anyio
async def test_local_vllm_client_passes_execution_engine_kwargs(monkeypatch):
    llm_client_module._LOCAL_VLLM_CACHE.clear()
    llm_client_module._LOCAL_VLLM_GENERATE_LOCKS.clear()
    llm_client_module._LOCAL_VLLM_BATCHERS.clear()
    captured = {}

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            captured["messages"] = messages
            captured["chat_template_kwargs"] = kwargs
            return "formatted prompt"

    class FakeGeneration:
        text = "assistantfinal<execution_thinking>vllm</execution_thinking>\n```python\npass\n```"

    class FakeRequestOutput:
        outputs = [FakeGeneration()]

    class FakeLLM:
        def get_tokenizer(self):
            return FakeTokenizer()

        def generate(self, prompts, sampling_params):
            captured["prompts"] = prompts
            captured["sampling_params"] = sampling_params
            return [FakeRequestOutput()]

    def fake_load_vllm(config):
        captured["load_config"] = config
        return FakeLLM()

    def fake_sampling_params(**kwargs):
        captured["sampling_kwargs"] = kwargs
        return kwargs

    monkeypatch.setattr("guidance_ttt.llm_client._load_local_vllm", fake_load_vllm)
    monkeypatch.setattr("guidance_ttt.llm_client._call_vllm_sampling_params", fake_sampling_params)

    client = make_llm_client(
        {
            "provider": "local_vllm",
            "model": "models/gpt-oss-20b",
            "tensor_model_parallel_size": 8,
            "gpu_memory_utilization": 0.35,
            "max_num_seqs": 32,
            "max_model_len": 32768,
            "reasoning_effort": "low",
            "top_p": 0.95,
        }
    )

    assert isinstance(client, LocalVLLMLLMClient)
    response = await client.complete(
        LLMRequest(
            system="system prompt",
            user="user prompt",
            model="models/gpt-oss-20b",
            temperature=0.35,
            max_tokens=26000,
            metadata={"purpose": "execution"},
        )
    )

    assert captured["load_config"]["tensor_model_parallel_size"] == 8
    assert captured["load_config"]["gpu_memory_utilization"] == 0.35
    assert captured["load_config"]["max_num_seqs"] == 32
    assert captured["load_config"]["max_model_len"] == 32768
    assert captured["chat_template_kwargs"]["reasoning_effort"] == "low"
    assert captured["sampling_kwargs"]["max_tokens"] == 26000
    assert captured["sampling_kwargs"]["temperature"] == 0.35
    assert captured["sampling_kwargs"]["top_p"] == 0.95
    assert response.text.startswith("<execution_thinking>vllm")
    assert response.metadata["provider"] == "local_vllm"


def test_local_vllm_loader_temporarily_scopes_cuda_visible_devices(monkeypatch):
    captured = {}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4")

    def fake_call_vllm_llm(**kwargs):
        captured["cuda_visible_devices_during_load"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr("guidance_ttt.llm_client._call_vllm_llm", fake_call_vllm_llm)

    llm_client_module._load_local_vllm(
        {
            "model": "models/gpt-oss-120b",
            "tensor_model_parallel_size": 1,
            "cuda_visible_devices": "4",
        }
    )

    assert captured["cuda_visible_devices_during_load"] == "4"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0,1,2,3,4"
    assert captured["kwargs"]["model"] == "models/gpt-oss-120b"


@pytest.mark.anyio
async def test_local_vllm_client_uses_remaining_context_when_max_tokens_is_none(monkeypatch):
    llm_client_module._LOCAL_VLLM_CACHE.clear()
    llm_client_module._LOCAL_VLLM_GENERATE_LOCKS.clear()
    llm_client_module._LOCAL_VLLM_BATCHERS.clear()
    captured = {}

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return "formatted prompt with five tokens"

        def encode(self, text, add_special_tokens=False):
            return text.split()

    class FakeGeneration:
        text = "<execution_thinking>auto</execution_thinking>\n```python\npass\n```"

    class FakeRequestOutput:
        outputs = [FakeGeneration()]

    class FakeLLM:
        def get_tokenizer(self):
            return FakeTokenizer()

        def generate(self, prompts, sampling_params):
            captured["prompts"] = prompts
            captured["sampling_params"] = sampling_params
            return [FakeRequestOutput()]

    monkeypatch.setattr("guidance_ttt.llm_client._load_local_vllm", lambda config: FakeLLM())
    monkeypatch.setattr(
        "guidance_ttt.llm_client._call_vllm_sampling_params",
        lambda **kwargs: captured.setdefault("sampling_kwargs", kwargs),
    )

    client = make_llm_client(
        {
            "provider": "local_vllm",
            "model": "models/gpt-oss-20b",
            "max_model_len": 32768,
        }
    )
    response = await client.complete(
        LLMRequest(
            system="system prompt",
            user="user prompt",
            model="models/gpt-oss-20b",
            temperature=0.35,
            max_tokens=None,
            metadata={"purpose": "execution"},
        )
    )

    assert captured["sampling_kwargs"]["max_tokens"] == 32763
    assert response.text.startswith("<execution_thinking>auto")


@pytest.mark.anyio
async def test_local_vllm_client_passes_qwen_thinking_chat_template_kwargs(monkeypatch):
    llm_client_module._LOCAL_VLLM_CACHE.clear()
    llm_client_module._LOCAL_VLLM_GENERATE_LOCKS.clear()
    llm_client_module._LOCAL_VLLM_BATCHERS.clear()
    captured = {}

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            captured["messages"] = messages
            captured["chat_template_kwargs"] = kwargs
            return "formatted qwen prompt"

    class FakeGeneration:
        text = "<execution_thinking>qwen</execution_thinking>\n```cpp\nint main(){return 0;}\n```"

    class FakeRequestOutput:
        outputs = [FakeGeneration()]

    class FakeLLM:
        def get_tokenizer(self):
            return FakeTokenizer()

        def generate(self, prompts, sampling_params):
            captured["prompts"] = prompts
            captured["sampling_params"] = sampling_params
            return [FakeRequestOutput()]

    monkeypatch.setattr("guidance_ttt.llm_client._load_local_vllm", lambda config: FakeLLM())
    monkeypatch.setattr(
        "guidance_ttt.llm_client._call_vllm_sampling_params",
        lambda **kwargs: captured.setdefault("sampling_kwargs", kwargs),
    )

    client = make_llm_client(
        {
            "provider": "local_vllm",
            "model": "Qwen/Qwen3-8B",
            "max_model_len": 32768,
            "chat_template_kwargs": {"enable_thinking": True},
        }
    )
    response = await client.complete(
        LLMRequest(
            system="system prompt",
            user="user prompt",
            model="Qwen/Qwen3-8B",
            temperature=0.35,
            max_tokens=128,
            metadata={"purpose": "execution"},
        )
    )

    assert captured["chat_template_kwargs"]["enable_thinking"] is True
    assert captured["chat_template_kwargs"]["tokenize"] is False
    assert captured["chat_template_kwargs"]["add_generation_prompt"] is True
    assert "reasoning_effort" not in captured["chat_template_kwargs"]
    assert response.text.startswith("<execution_thinking>qwen")


@pytest.mark.anyio
async def test_local_vllm_client_batches_concurrent_generate_calls(monkeypatch):
    llm_client_module._LOCAL_VLLM_CACHE.clear()
    llm_client_module._LOCAL_VLLM_GENERATE_LOCKS.clear()
    llm_client_module._LOCAL_VLLM_BATCHERS.clear()
    generate_batches = []

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return f"formatted {messages[-1]['content']}"

    class FakeGeneration:
        def __init__(self, text):
            self.text = text

    class FakeRequestOutput:
        def __init__(self, text):
            self.outputs = [FakeGeneration(text)]

    class FakeLLM:
        def get_tokenizer(self):
            return FakeTokenizer()

        def generate(self, prompts, sampling_params):
            generate_batches.append(list(prompts))
            time.sleep(0.03)
            return [
                FakeRequestOutput(f"<execution_thinking>{prompt}</execution_thinking>\n```python\npass\n```")
                for prompt in prompts
            ]

    monkeypatch.setattr("guidance_ttt.llm_client._load_local_vllm", lambda config: FakeLLM())
    monkeypatch.setattr("guidance_ttt.llm_client._call_vllm_sampling_params", lambda **kwargs: kwargs)
    client = LocalVLLMLLMClient(
        {
            "provider": "local_vllm",
            "model": "models/gpt-oss-20b",
            "batch_wait_ms": 50,
            "max_batch_size": 8,
        }
    )
    request_one = LLMRequest(
        system="system prompt",
        user="user one",
        model="models/gpt-oss-20b",
        temperature=0.0,
        max_tokens=16,
        metadata={"purpose": "execution"},
    )
    request_two = LLMRequest(
        system="system prompt",
        user="user two",
        model="models/gpt-oss-20b",
        temperature=0.0,
        max_tokens=16,
        metadata={"purpose": "execution"},
    )

    response_one, response_two = await asyncio.gather(client.complete(request_one), client.complete(request_two))

    assert generate_batches == [["formatted user one", "formatted user two"]]
    assert response_one.text.startswith("<execution_thinking>formatted user one")
    assert response_two.text.startswith("<execution_thinking>formatted user two")


@pytest.mark.anyio
async def test_local_transformers_clients_share_pipeline_cache(monkeypatch):
    llm_client_module._LOCAL_PIPELINE_CACHE.clear()
    load_count = 0

    def fake_load_pipeline(config):
        nonlocal load_count
        load_count += 1

        def fake_pipeline(messages, **generation_kwargs):
            return [{"generated_text": "<execution_thinking>cached</execution_thinking>\n```python\npass\n```"}]

        return fake_pipeline

    monkeypatch.setattr("guidance_ttt.llm_client._load_local_text_generation_pipeline", fake_load_pipeline)
    config = {"provider": "local", "model": "models/cache-test", "device_map": "auto"}
    request = LLMRequest(
        system="system",
        user="user",
        model="models/cache-test",
        temperature=0.0,
        max_tokens=16,
        metadata={"purpose": "execution"},
    )

    response_one = await LocalTransformersLLMClient(config).complete(request)
    response_two = await LocalTransformersLLMClient(config).complete(request)

    assert response_one.text.startswith("<execution_thinking>cached")
    assert response_two.text.startswith("<execution_thinking>cached")
    assert load_count == 1


def test_local_pipeline_loader_passes_trust_remote_code_once(monkeypatch):
    captured = {}

    def fake_pipeline(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("guidance_ttt.llm_client._call_transformers_pipeline", fake_pipeline)

    _load_local_text_generation_pipeline(
        {
            "model": "models/gpt-oss-20b",
            "device_map": "auto",
            "torch_dtype": "auto",
            "trust_remote_code": True,
        }
    )

    assert captured["trust_remote_code"] is True
    assert "trust_remote_code" not in captured.get("model_kwargs", {})


def test_local_pipeline_loader_passes_model_kwargs_for_memory_limits(monkeypatch):
    captured = {}

    def fake_pipeline(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("guidance_ttt.llm_client._call_transformers_pipeline", fake_pipeline)

    _load_local_text_generation_pipeline(
        {
            "model": "models/gpt-oss-20b",
            "device_map": "auto",
            "model_kwargs": {"max_memory": {0: "8GiB", "cpu": "256GiB"}},
        }
    )

    assert captured["model_kwargs"]["max_memory"] == {0: "8GiB", "cpu": "256GiB"}


def test_local_pipeline_loader_strips_gpu_memory_limits_for_cpu_device_map(monkeypatch):
    captured = {}

    def fake_pipeline(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("guidance_ttt.llm_client._call_transformers_pipeline", fake_pipeline)

    _load_local_text_generation_pipeline(
        {
            "model": "models/gpt-oss-20b",
            "device_map": "cpu",
            "model_kwargs": {"max_memory": {0: "8GiB", 1: "8GiB", "cpu": "256GiB"}},
        }
    )

    assert captured["model_kwargs"]["max_memory"] == {"cpu": "256GiB"}


def test_local_client_cache_key_handles_mixed_memory_keys_before_loading(monkeypatch):
    llm_client_module._LOCAL_PIPELINE_CACHE.clear()
    captured = {}

    def fake_pipeline(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("guidance_ttt.llm_client._call_transformers_pipeline", fake_pipeline)
    client = LocalTransformersLLMClient(
        {
            "provider": "local",
            "model": "models/gpt-oss-20b",
            "device_map": "cpu",
            "model_kwargs": {"max_memory": {0: "8GiB", 1: "8GiB", "cpu": "256GiB"}},
        }
    )

    client._get_pipeline()

    assert captured["model_kwargs"]["max_memory"] == {"cpu": "256GiB"}


def test_extract_local_generated_text_prefers_harmony_final_channel():
    text = (
        "analysisWe need to produce code."
        "assistantfinal<execution_thinking>ok</execution_thinking>\n"
        "```python\npass\n```\n<summary>done</summary>"
    )

    assert _extract_local_generated_text([{"generated_text": text}]).startswith("<execution_thinking>ok")
