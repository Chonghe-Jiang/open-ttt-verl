import asyncio
import io
import json
import os
import threading
import urllib.error

import pytest
import time

import guidance_ttt.llm_client as llm_client_module
from guidance_ttt.llm_client import (
    LLMRequest,
    LocalTransformersLLMClient,
    LocalVLLMLLMClient,
    MockLLMClient,
    OpenAICompatibleLLMClient,
    _load_local_text_generation_pipeline,
    _extract_local_generated_text,
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


@pytest.mark.anyio
async def test_openai_compatible_client_forwards_reasoning_options(monkeypatch):
    monkeypatch.setenv("OR_KEY", "openrouter-test-key")
    client = make_llm_client(
        {
            "provider": "openai_compatible",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "OR_KEY",
            "reasoning": {"effort": "high", "exclude": False},
            "top_p": 1.0,
        }
    )
    captured = {}

    def fake_post_json(path, payload):
        captured["path"] = path
        captured["payload"] = payload
        return {
            "model": "z-ai/glm-5.2",
            "choices": [{"message": {"content": "<solution>ok</solution>"}, "finish_reason": "stop"}],
        }

    monkeypatch.setattr(client, "_post_json", fake_post_json)
    await client.complete(
        LLMRequest(
            system="system",
            user="user",
            model="z-ai/glm-5.2",
            temperature=1.0,
            max_tokens=32768,
        )
    )

    assert captured["path"] == "/chat/completions"
    assert captured["payload"]["reasoning"] == {"effort": "high", "exclude": False}
    assert captured["payload"]["top_p"] == 1.0


@pytest.mark.anyio
async def test_openai_compatible_client_uses_configured_http_concurrency(monkeypatch):
    monkeypatch.setenv("OR_KEY", "openrouter-test-key")
    client = OpenAICompatibleLLMClient(
        {
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "OR_KEY",
            "concurrency": 128,
        }
    )
    lock = threading.Lock()
    active = 0
    max_active = 0

    def fake_post_json(_path, payload):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(0.05)
            return {
                "model": payload["model"],
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            }
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(client, "_post_json", fake_post_json)
    try:
        await asyncio.gather(
            *(
                client.complete(
                    LLMRequest(
                        system="system",
                        user=f"user-{index}",
                        model="z-ai/glm-5.2",
                        temperature=1.0,
                        max_tokens=128,
                    )
                )
                for index in range(128)
            )
        )
    finally:
        client.close()

    assert client.concurrency == 128
    assert max_active == 128


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


class _HTTPResponse:
    status = 200

    def __init__(self, payload):
        self.payload = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.payload


def test_openai_compatible_client_retries_malformed_json_and_http_503(monkeypatch):
    client = OpenAICompatibleLLMClient(
        {
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "not-a-real-key",
            "max_retries": 2,
            "retry_backoff_s": 0,
        }
    )
    malformed = _HTTPResponse({})
    malformed.payload = b'{"choices": ['
    unavailable = urllib.error.HTTPError(
        "https://openrouter.ai/api/v1/chat/completions",
        503,
        "unavailable",
        {},
        io.BytesIO(b"temporary"),
    )
    successful = _HTTPResponse(
        {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        }
    )
    responses = iter([malformed, unavailable, successful])

    def urlopen(*_args, **_kwargs):
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(llm_client_module.urllib.request, "urlopen", urlopen)

    data = client._post_json("/chat/completions", {"model": "z-ai/glm-5.2"})

    assert data["choices"][0]["message"]["content"] == "ok"


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
    monkeypatch.setattr("guidance_ttt.llm_client._call_vllm_sampling_params", lambda **kwargs: captured.setdefault("sampling_kwargs", kwargs))

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
    monkeypatch.setattr("guidance_ttt.llm_client._call_vllm_sampling_params", lambda **kwargs: captured.setdefault("sampling_kwargs", kwargs))

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
