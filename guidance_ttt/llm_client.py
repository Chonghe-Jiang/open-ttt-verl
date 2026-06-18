from __future__ import annotations

import asyncio
import json
import os
import threading
from collections.abc import Callable
from typing import Protocol
import urllib.error
import urllib.request

from guidance_ttt.state import LLMRequest, LLMResponse


class BaseLLMClient(Protocol):
    async def complete(self, request: LLMRequest) -> LLMResponse: ...


class MockLLMClient:
    async def complete(self, request: LLMRequest) -> LLMResponse:
        purpose = request.metadata.get("purpose")
        if purpose == "summary":
            text = """<summary>
Outcome: Mock verifier evidence was recorded.
Reusable idea: Preserve the selected branch idea and make a small concrete change.
Failure mode: None observed in mock mode.
What future guidance should preserve: The parent construction context.
What future guidance should change: Try a more targeted local optimization.
</summary>"""
        elif purpose == "execution":
            text = """<execution_thinking>
Use the guidance to return a simple valid Erdos baseline construction.
</execution_thinking>

```python
def run(seed=42, budget_s=1, **kwargs):
    return ([0.5, 0.5], 0.5, 2)
```

<summary>
Outcome hypothesis: The baseline construction should pass verifier checks but is not a discovery improvement.
Reusable idea: Preserve a simple verifier-valid construction while testing the guidance/execution pipeline.
Risk / possible failure mode: The construction is too trivial to improve reward.
What future guidance should preserve: The exact run(seed=42, budget_s=..., **kwargs) interface.
What future guidance should change: Propose a nontrivial search or optimization over h_values.
</summary>"""
        else:
            text = request.user
        return LLMResponse(
            text=text,
            model=request.model,
            finish_reason="stop",
            usage={"prompt_chars": len(request.user), "completion_chars": len(text)},
            metadata={"mock": True, **request.metadata},
        )


class UnconfiguredLLMClient:
    async def complete(self, request: LLMRequest) -> LLMResponse:
        raise RuntimeError(
            f"No real LLM client configured for model {request.model!r}. "
            "Use provider=mock for tests or wire an API client before real runs."
        )


class OpenAICompatibleLLMClient:
    def __init__(self, config: dict):
        self.endpoint = str(config.get("endpoint") or config.get("base_url") or os.environ.get("ENDPOINT", "")).rstrip("/")
        self.api_key = str(config.get("api_key") or os.environ.get("API_KEY", ""))
        if not self.endpoint or not self.api_key:
            raise ValueError("OpenAI-compatible executor requires endpoint/base_url and api_key, or ENDPOINT/API_KEY env vars")

    async def complete(self, request: LLMRequest) -> LLMResponse:
        payload = {
            "model": request.model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        data = await asyncio.to_thread(self._post_json, "/chat/completions", payload)
        choice = data.get("choices", [{}])[0]
        message = choice.get("message") or {}
        text = message.get("content") or choice.get("text") or ""
        return LLMResponse(
            text=text,
            model=data.get("model") or request.model,
            finish_reason=choice.get("finish_reason", ""),
            usage=data.get("usage", {}),
            metadata={"provider": "openai_compatible", **request.metadata},
        )

    def _post_json(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            self.endpoint + path,
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")[:1000]
            raise RuntimeError(f"LLM API request failed with HTTP {exc.code}: {body}") from exc


_LOCAL_PIPELINE_CACHE: dict[str, Callable] = {}
_LOCAL_PIPELINE_CACHE_LOCK = threading.Lock()


def _local_pipeline_cache_key(config: dict) -> str:
    return json.dumps(config, sort_keys=True, default=str)


def _call_transformers_pipeline(**kwargs):
    from transformers import pipeline

    return pipeline(**kwargs)


def _load_local_text_generation_pipeline(config: dict) -> Callable:
    model = str(config.get("model") or config.get("model_path"))
    pipeline_kwargs: dict = {
        "task": "text-generation",
        "model": model,
        "device_map": config.get("device_map", "auto"),
    }
    if "torch_dtype" in config:
        pipeline_kwargs["dtype"] = config["torch_dtype"]
    model_kwargs = dict(config.get("model_kwargs") or {})
    for key in ("trust_remote_code", "revision"):
        if key in config:
            pipeline_kwargs[key] = config[key]
    for key in ("attn_implementation",):
        if key in config:
            model_kwargs[key] = config[key]
    if str(config.get("device_map", "")).lower() == "cpu" and isinstance(model_kwargs.get("max_memory"), dict):
        cpu_memory = model_kwargs["max_memory"].get("cpu")
        if cpu_memory is not None:
            model_kwargs["max_memory"] = {"cpu": cpu_memory}
        else:
            model_kwargs.pop("max_memory")
    if model_kwargs:
        pipeline_kwargs["model_kwargs"] = model_kwargs
    pipeline_kwargs.update(config.get("pipeline_kwargs") or {})
    return _call_transformers_pipeline(**pipeline_kwargs)


class LocalTransformersLLMClient:
    """Lazy local Transformers text-generation client for execution LLM calls."""

    def __init__(self, config: dict):
        self.config = dict(config or {})
        self.model = str(self.config.get("model") or self.config.get("model_path") or "")
        if not self.model:
            raise ValueError("Local Transformers executor requires llm.execution.model or model_path")
        self._pipeline = None

    async def complete(self, request: LLMRequest) -> LLMResponse:
        messages = [
            {"role": "system", "content": request.system},
            {"role": "user", "content": request.user},
        ]
        generation_kwargs = self._generation_kwargs(request)
        text = await asyncio.to_thread(self._complete_sync, messages, generation_kwargs)
        return LLMResponse(
            text=text,
            model=self.model,
            finish_reason="stop",
            usage={"prompt_chars": len(request.system) + len(request.user), "completion_chars": len(text)},
            metadata={"provider": "local_transformers", **request.metadata},
        )

    def _generation_kwargs(self, request: LLMRequest) -> dict:
        kwargs = {
            "max_new_tokens": int(request.max_tokens),
            "return_full_text": False,
        }
        if request.temperature > 0:
            kwargs["do_sample"] = True
            kwargs["temperature"] = float(request.temperature)
        else:
            kwargs["do_sample"] = False
        if "top_p" in self.config:
            kwargs["top_p"] = float(self.config["top_p"])
        tokenizer_encode_kwargs = dict(self.config.get("tokenizer_encode_kwargs") or {})
        if "reasoning_effort" in self.config:
            tokenizer_encode_kwargs["reasoning_effort"] = str(self.config["reasoning_effort"])
        if tokenizer_encode_kwargs:
            kwargs["tokenizer_encode_kwargs"] = tokenizer_encode_kwargs
        kwargs.update(self.config.get("generation_kwargs") or {})
        return kwargs

    def _complete_sync(self, messages: list[dict[str, str]], generation_kwargs: dict) -> str:
        pipe = self._get_pipeline()
        result = pipe(messages, **generation_kwargs)
        return _extract_local_generated_text(result)

    def _get_pipeline(self):
        if self._pipeline is None:
            cache_key = _local_pipeline_cache_key(self.config)
            with _LOCAL_PIPELINE_CACHE_LOCK:
                self._pipeline = _LOCAL_PIPELINE_CACHE.get(cache_key)
                if self._pipeline is None:
                    self._pipeline = _load_local_text_generation_pipeline(self.config)
                    _LOCAL_PIPELINE_CACHE[cache_key] = self._pipeline
        return self._pipeline


def _extract_local_generated_text(result) -> str:
    if isinstance(result, list) and result:
        result = result[0]
    if isinstance(result, dict):
        generated = result.get("generated_text", result.get("text", ""))
    else:
        generated = result
    if isinstance(generated, list):
        for message in reversed(generated):
            if isinstance(message, dict) and message.get("role") == "assistant":
                return _strip_harmony_to_final(str(message.get("content", "")))
        text = str(generated[-1].get("content", "")) if generated and isinstance(generated[-1], dict) else str(generated)
        return _strip_harmony_to_final(text)
    return _strip_harmony_to_final(str(generated or ""))


def _strip_harmony_to_final(text: str) -> str:
    markers = ("<|channel|>final<|message|>", "assistantfinal")
    for marker in markers:
        if marker in text:
            text = text.rsplit(marker, 1)[1]
            break
    for end_marker in ("<|return|>", "<|end|>"):
        if end_marker in text:
            text = text.split(end_marker, 1)[0]
    return text.strip()


def make_llm_client(config: dict) -> BaseLLMClient:
    provider = (config or {}).get("provider", "mock")
    if provider == "mock":
        return MockLLMClient()
    if provider in {"openai", "openai_compatible"}:
        return OpenAICompatibleLLMClient(config)
    if provider in {"local", "transformers", "local_transformers"}:
        return LocalTransformersLLMClient(config)
    return UnconfiguredLLMClient()
