from __future__ import annotations

import asyncio
import http.client
import json
import os
import threading
import time
from collections.abc import Callable
from typing import Any, Protocol
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
Execution Interpretation
Mock verifier evidence was recorded.

Implemented Algorithm
No execution code is generated for summary-only mock calls.

New Ideas Introduced
Preserve the selected branch idea and make a small concrete change.

Empirical Outcome
Pending verifier execution.

Failure / Bottleneck Analysis
None observed in mock mode.

Next Guidance Delta
Try a more targeted local optimization while preserving the parent construction context.
</summary>"""
        elif purpose == "execution":
            text = """<execution_thinking>
Use the guidance to return a simple valid Erdos baseline construction.
</execution_thinking>

<solution>
```python
def run(seed=42, budget_s=1, **kwargs):
    return ([0.5, 0.5], 0.5, 2)
```
</solution>

<summary>
Execution Interpretation
The baseline construction should pass verifier checks but is not a discovery improvement.

Implemented Algorithm
Return a constant verifier-valid h profile through the required run interface.

New Ideas Introduced
Preserve a simple verifier-valid construction while testing the guidance/execution pipeline.

Empirical Outcome
Pending verifier execution.

Failure / Bottleneck Analysis
The construction is too trivial to improve reward.

Next Guidance Delta
Preserve the exact run(seed=42, budget_s=..., **kwargs) interface and propose a nontrivial search over h_values.
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
        api_key_env = str(config.get("api_key_env") or "").strip()
        env_api_key = os.environ.get(api_key_env, "") if api_key_env else ""
        self.api_key = str(config.get("api_key") or env_api_key or os.environ.get("API_KEY", ""))
        if not self.endpoint or not self.api_key:
            raise ValueError(
                "OpenAI-compatible executor requires endpoint/base_url and api_key, api_key_env, or ENDPOINT/API_KEY env vars"
            )
        self.timeout_s = float(config.get("timeout_s", 120))
        self.max_retries = max(0, int(config.get("max_retries", 0)))
        self.retry_backoff_s = max(0.0, float(config.get("retry_backoff_s", 1.0)))
        self.request_options = {
            key: dict(config[key]) if key in {"chat_template_kwargs", "reasoning"} else config[key]
            for key in (
                "top_p",
                "top_k",
                "min_p",
                "chat_template_kwargs",
                "reasoning",
                "reasoning_effort",
                "verbosity",
            )
            if key in config
        }

    async def complete(self, request: LLMRequest) -> LLMResponse:
        payload = {
            "model": request.model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
            "temperature": request.temperature,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        payload.update(self.request_options)
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
        retry_errors: list[str] = []
        req = urllib.request.Request(
            self.endpoint + path,
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    data = json.loads(resp.read().decode())
                    if not isinstance(data, dict):
                        raise ValueError(f"expected a JSON object, got {type(data).__name__}")
                    response_error = self._retryable_response_error(data)
                    if response_error:
                        raise _RetryableLLMResponseError(response_error)
                    return data
            except urllib.error.HTTPError as exc:
                try:
                    body = exc.read().decode(errors="replace")[:1000]
                except (http.client.HTTPException, OSError):
                    body = "<failed to read HTTP error body>"
                retryable = exc.code in {408, 429, 500, 502, 503, 504}
                if not retryable or attempt >= self.max_retries:
                    raise RuntimeError(f"LLM API request failed with HTTP {exc.code}: {body}") from exc
                retry_errors.append(f"HTTP {exc.code}")
                retry_after = exc.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after is not None else self.retry_backoff_s * (2**attempt)
                except ValueError:
                    delay = self.retry_backoff_s * (2**attempt)
                print(
                    f"LLM API HTTP {exc.code}; retrying attempt {attempt + 2}/"
                    f"{self.max_retries + 1} after {max(0.0, delay):.1f}s",
                    flush=True,
                )
                time.sleep(max(0.0, delay))
            except (
                http.client.HTTPException,
                urllib.error.URLError,
                OSError,
                UnicodeError,
                ValueError,
                _RetryableLLMResponseError,
            ) as exc:
                error_name = (
                    exc.reason
                    if isinstance(exc, _RetryableLLMResponseError)
                    else type(exc).__name__
                )
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        f"LLM API request failed after {attempt + 1} attempts with "
                        f"{error_name}: {exc}; prior retries={retry_errors}"
                    ) from exc
                retry_errors.append(error_name)
                delay = self.retry_backoff_s * (2**attempt)
                print(
                    f"LLM API {error_name}; retrying attempt {attempt + 2}/"
                    f"{self.max_retries + 1} after {delay:.1f}s",
                    flush=True,
                )
                time.sleep(delay)
        raise AssertionError("unreachable")

    @staticmethod
    def _retryable_response_error(data: dict) -> str | None:
        if data.get("error"):
            return "api_error_response"
        choices = data.get("choices") or []
        if not choices:
            return "missing_choices"
        choice = choices[0] or {}
        finish_reason = str(choice.get("finish_reason") or "").lower()
        if not finish_reason:
            return "missing_finish_reason"
        if finish_reason == "error":
            return "finish_reason_error"
        message = choice.get("message") or {}
        if not (message.get("content") or choice.get("text")):
            return "missing_final_content"
        return None


class _RetryableLLMResponseError(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


_LOCAL_PIPELINE_CACHE: dict[str, Callable] = {}
_LOCAL_PIPELINE_CACHE_LOCK = threading.Lock()
_LOCAL_VLLM_CACHE: dict[str, Any] = {}
_LOCAL_VLLM_GENERATE_LOCKS: dict[str, threading.Lock] = {}
_LOCAL_VLLM_BATCHERS: dict[str, "_LocalVLLMBatcher"] = {}
_LOCAL_VLLM_CACHE_LOCK = threading.Lock()


def _local_pipeline_cache_key(config: dict) -> str:
    return json.dumps(_stringify_mapping_keys(config), sort_keys=True, default=str)


def _stringify_mapping_keys(value):
    if isinstance(value, dict):
        return {str(key): _stringify_mapping_keys(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_stringify_mapping_keys(item) for item in value]
    return value


def _count_prompt_tokens(tokenizer, prompt: str) -> int | None:
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        try:
            return len(encode(prompt, add_special_tokens=False))
        except TypeError:
            return len(encode(prompt))
    if callable(tokenizer):
        try:
            encoded = tokenizer(prompt, add_special_tokens=False)
        except TypeError:
            encoded = tokenizer(prompt)
        if isinstance(encoded, dict) and "input_ids" in encoded:
            return len(encoded["input_ids"])
        if isinstance(encoded, list):
            return len(encoded)
    return None


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
            "return_full_text": False,
        }
        if request.max_tokens is not None:
            kwargs["max_new_tokens"] = int(request.max_tokens)
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


def _call_vllm_llm(**kwargs):
    from vllm import LLM

    return LLM(**kwargs)


def _call_vllm_sampling_params(**kwargs):
    from vllm import SamplingParams

    return SamplingParams(**kwargs)


def _load_local_vllm(config: dict):
    model = str(config.get("model") or config.get("model_path"))
    llm_kwargs: dict[str, Any] = {
        "model": model,
        "tensor_parallel_size": int(config.get("tensor_model_parallel_size", config.get("tensor_parallel_size", 1))),
        "gpu_memory_utilization": float(config.get("gpu_memory_utilization", 0.5)),
    }
    optional_keys = {
        "max_num_seqs": "max_num_seqs",
        "max_model_len": "max_model_len",
        "trust_remote_code": "trust_remote_code",
        "dtype": "dtype",
        "enforce_eager": "enforce_eager",
        "download_dir": "download_dir",
        "tokenizer": "tokenizer",
        "tokenizer_mode": "tokenizer_mode",
        "load_format": "load_format",
        "disable_custom_all_reduce": "disable_custom_all_reduce",
    }
    if "torch_dtype" in config and "dtype" not in config:
        llm_kwargs["dtype"] = config["torch_dtype"]
    for source_key, target_key in optional_keys.items():
        if source_key in config:
            llm_kwargs[target_key] = config[source_key]
    llm_kwargs.update(config.get("engine_kwargs") or {})
    cuda_visible_devices = config.get("cuda_visible_devices")
    if cuda_visible_devices is None:
        return _call_vllm_llm(**llm_kwargs)
    previous_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    try:
        return _call_vllm_llm(**llm_kwargs)
    finally:
        if previous_cuda_visible_devices is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous_cuda_visible_devices


class LocalVLLMLLMClient:
    """Lazy local vLLM execution client for long-context execution calls."""

    def __init__(self, config: dict):
        self.config = dict(config or {})
        self.model = str(self.config.get("model") or self.config.get("model_path") or "")
        if not self.model:
            raise ValueError("Local vLLM executor requires llm.execution.model or model_path")
        self._cache_key = _local_pipeline_cache_key(self.config)
        self._llm = None

    async def complete(self, request: LLMRequest) -> LLMResponse:
        text = await asyncio.to_thread(self._get_batcher().complete, request)
        return LLMResponse(
            text=text,
            model=self.model,
            finish_reason="stop",
            usage={"prompt_chars": len(request.system) + len(request.user), "completion_chars": len(text)},
            metadata={"provider": "local_vllm", **request.metadata},
        )

    def _complete_sync(self, request: LLMRequest) -> str:
        return self._get_batcher().complete(request)

    def _extract_vllm_text(self, first) -> str:
        generations = getattr(first, "outputs", None) or []
        if not generations:
            return ""
        return _strip_harmony_to_final(str(getattr(generations[0], "text", "") or ""))

    def _format_prompt(self, llm, request: LLMRequest) -> str:
        messages = [
            {"role": "system", "content": request.system},
            {"role": "user", "content": request.user},
        ]
        tokenizer = llm.get_tokenizer()
        chat_template_kwargs = dict(self.config.get("chat_template_kwargs") or {})
        if "reasoning_effort" in self.config:
            chat_template_kwargs["reasoning_effort"] = str(self.config["reasoning_effort"])
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )

    def _sampling_params(self, request: LLMRequest, *, prompt: str, llm):
        kwargs = {
            "temperature": float(request.temperature),
        }
        max_tokens = self._resolve_max_tokens(request, prompt=prompt, llm=llm)
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if "top_p" in self.config:
            kwargs["top_p"] = float(self.config["top_p"])
        kwargs.update(self.config.get("sampling_kwargs") or {})
        return _call_vllm_sampling_params(**kwargs)

    def _resolve_max_tokens(self, request: LLMRequest, *, prompt: str, llm) -> int | None:
        if request.max_tokens is not None:
            return int(request.max_tokens)
        max_model_len = self.config.get("max_model_len")
        if max_model_len is None:
            return None
        prompt_tokens = _count_prompt_tokens(llm.get_tokenizer(), prompt)
        if prompt_tokens is None:
            return int(max_model_len)
        return max(1, int(max_model_len) - int(prompt_tokens))

    def _get_llm(self):
        if self._llm is None:
            with _LOCAL_VLLM_CACHE_LOCK:
                self._llm = _LOCAL_VLLM_CACHE.get(self._cache_key)
                if self._llm is None:
                    self._llm = _load_local_vllm(self.config)
                    _LOCAL_VLLM_CACHE[self._cache_key] = self._llm
        return self._llm

    def _get_generate_lock(self) -> threading.Lock:
        with _LOCAL_VLLM_CACHE_LOCK:
            lock = _LOCAL_VLLM_GENERATE_LOCKS.get(self._cache_key)
            if lock is None:
                lock = threading.Lock()
                _LOCAL_VLLM_GENERATE_LOCKS[self._cache_key] = lock
            return lock

    def _get_batcher(self) -> "_LocalVLLMBatcher":
        with _LOCAL_VLLM_CACHE_LOCK:
            batcher = _LOCAL_VLLM_BATCHERS.get(self._cache_key)
            if batcher is None:
                batcher = _LocalVLLMBatcher(self)
                _LOCAL_VLLM_BATCHERS[self._cache_key] = batcher
            return batcher


class _LocalVLLMBatchItem:
    def __init__(self, request: LLMRequest):
        self.request = request
        self.text = ""
        self.error: BaseException | None = None
        self.done = False


class _LocalVLLMBatcher:
    def __init__(self, client: LocalVLLMLLMClient):
        self.client = client
        self.max_batch_size = max(
            1,
            int(client.config.get("max_batch_size", client.config.get("max_num_seqs", 8))),
        )
        self.batch_wait_s = max(0.0, float(client.config.get("batch_wait_ms", 20)) / 1000.0)
        self._condition = threading.Condition()
        self._queue: list[_LocalVLLMBatchItem] = []
        self._leader_active = False

    def complete(self, request: LLMRequest) -> str:
        item = _LocalVLLMBatchItem(request)
        with self._condition:
            self._queue.append(item)
            self._condition.notify_all()

        while True:
            batch = self._take_batch_or_wait(item)
            if batch is not None:
                self._run_batch(batch)
                with self._condition:
                    self._leader_active = False
                    self._condition.notify_all()

            with self._condition:
                if item.done:
                    if item.error is not None:
                        raise item.error
                    return item.text

    def _take_batch_or_wait(self, item: _LocalVLLMBatchItem) -> list[_LocalVLLMBatchItem] | None:
        with self._condition:
            if item.done:
                return None
            if self._leader_active:
                self._condition.wait()
                return None
            self._leader_active = True
            deadline = time.monotonic() + self.batch_wait_s
            while len(self._queue) < self.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            batch = self._queue[: self.max_batch_size]
            del self._queue[: len(batch)]
            return batch

    def _run_batch(self, batch: list[_LocalVLLMBatchItem]) -> None:
        try:
            llm = self.client._get_llm()
            prompts = []
            sampling_params = []
            for item in batch:
                prompt = self.client._format_prompt(llm, item.request)
                prompts.append(prompt)
                sampling_params.append(self.client._sampling_params(item.request, prompt=prompt, llm=llm))
            generate_sampling_params = sampling_params[0] if len(sampling_params) == 1 else sampling_params
            outputs = llm.generate(prompts, sampling_params=generate_sampling_params)
            texts = [self.client._extract_vllm_text(output) for output in (outputs or [])]
            if len(texts) < len(batch):
                texts.extend([""] * (len(batch) - len(texts)))
            with self._condition:
                for item, text in zip(batch, texts, strict=False):
                    item.text = text
                    item.done = True
                self._condition.notify_all()
        except BaseException as exc:
            with self._condition:
                for item in batch:
                    item.error = exc
                    item.done = True
                self._condition.notify_all()


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
    if provider in {"local_vllm", "vllm"}:
        return LocalVLLMLLMClient(config)
    return UnconfiguredLLMClient()
