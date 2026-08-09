from __future__ import annotations

import asyncio
import http.client
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

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
        self.endpoint = str(
            config.get("endpoint")
            or config.get("base_url")
            or os.environ.get("ENDPOINT", "")
        ).rstrip("/")
        api_key_env = str(config.get("api_key_env") or "").strip()
        env_api_key = os.environ.get(api_key_env, "") if api_key_env else ""
        api_key_file = str(config.get("api_key_file") or "").strip()
        file_api_key = Path(api_key_file).expanduser().read_text().strip() if api_key_file else ""
        self.api_key = str(config.get("api_key") or env_api_key or file_api_key or os.environ.get("API_KEY", ""))
        if not self.endpoint or not self.api_key:
            raise ValueError(
                "OpenAI-compatible executor requires endpoint/base_url and api_key, "
                "api_key_env, or ENDPOINT/API_KEY env vars"
            )
        self.timeout_s = float(config.get("timeout_s", 120))
        self.max_retries = max(0, int(config.get("max_retries", 0)))
        self.retry_backoff_s = max(0.0, float(config.get("retry_backoff_s", 1.0)))
        self.capture_request_metrics = bool(config.get("capture_request_metrics", False))
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
                "enable_thinking",
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
        reasoning = (
            message.get("reasoning_content")
            or message.get("reasoning")
            or choice.get("reasoning_content")
            or choice.get("reasoning")
            or ""
        )
        request_metrics = data.pop("_client_request_metadata", {})
        response_provider = data.get("provider")
        response_metadata = {"provider": "openai_compatible", **request_metrics}
        if data.get("model"):
            response_metadata["api_response_model"] = data["model"]
        if response_provider:
            response_metadata["api_response_provider"] = response_provider
        response_metadata.update(request.metadata)
        return LLMResponse(
            text=text,
            model=data.get("model") or request.model,
            finish_reason=choice.get("finish_reason", ""),
            reasoning=str(reasoning),
            usage=data.get("usage", {}),
            metadata=response_metadata,
        )

    def _post_json(self, path: str, payload: dict) -> dict:
        started_at = time.monotonic()
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
                    retryable_response_error = self._retryable_response_error(data)
                    if retryable_response_error:
                        raise _RetryableLLMResponseError(retryable_response_error)
                    if self.capture_request_metrics:
                        data["_client_request_metadata"] = {
                            "api_attempts": attempt + 1,
                            "api_retry_count": attempt,
                            "api_retry_errors": list(retry_errors),
                            "api_elapsed_s": time.monotonic() - started_at,
                            "api_http_status": int(getattr(resp, "status", 200)),
                        }
                    return data
            except urllib.error.HTTPError as exc:
                try:
                    body = exc.read().decode(errors="replace")[:1000]
                except (http.client.HTTPException, OSError):
                    body = "<failed to read HTTP error body>"
                retryable = exc.code == 429 or exc.code in {408, 500, 502, 503, 504}
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
                        f"{error_name}: {exc}"
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


class GPTOSSTwoPhaseOpenAICompatibleLLMClient(OpenAICompatibleLLMClient):
    """OpenAI-compatible GPT-OSS client with Discover's forced-final sampling.

    The regular chat-completions endpoint parses Harmony output into separate
    reasoning and content fields, which makes an exact continuation impossible
    after a length stop.  vLLM's completions endpoint accepts and returns token
    IDs, so both phases operate on the original Harmony sequence just like the
    official Discover ``TwoPhaseTokenCompleter``.
    """

    PHASE2_PREFILL = "\n\n... okay, I am out of thinking tokens. I need to send my final message now."
    GPTOSS_FINAL_MARKER = "<|end|><|start|>assistant<|channel|>final<|message|>"
    GPTOSS_FINAL_CHANNEL_INDICATOR = "<|channel|>final<|message|>"

    def __init__(self, config: dict):
        super().__init__(config)
        self.phase1_max_tokens = int(config["phase1_max_tokens"])
        self.context_window = int(config.get("context_window", 32768))
        self.context_buffer = int(config.get("context_buffer", 50))
        if self.phase1_max_tokens <= 0:
            raise ValueError("phase1_max_tokens must be positive")
        if self.context_window <= self.phase1_max_tokens + self.context_buffer:
            raise ValueError(
                "context_window must exceed phase1_max_tokens + context_buffer "
                f"({self.context_window} <= {self.phase1_max_tokens} + {self.context_buffer})"
            )

    async def complete(self, request: LLMRequest) -> LLMResponse:
        return await asyncio.to_thread(self._complete_two_phase, request)

    def _complete_two_phase(self, request: LLMRequest) -> LLMResponse:
        prompt_ids = self._render_harmony_prompt(request)
        phase1_max = self.phase1_max_tokens - len(prompt_ids)
        if phase1_max <= 0:
            raise ValueError(
                f"Prompt length {len(prompt_ids)} exceeds phase1_max_tokens "
                f"{self.phase1_max_tokens}"
            )

        phase1_data = self._post_json(
            "/completions",
            self._completion_payload(
                request=request,
                prompt_ids=prompt_ids,
                max_tokens=phase1_max,
            ),
        )
        phase1_choice = (phase1_data.get("choices") or [{}])[0]
        phase1_ids = self._choice_token_ids(phase1_choice, phase="phase1")
        phase1_finish = str(phase1_choice.get("finish_reason") or "")
        stop_ids = set(self._harmony_stop_token_ids())
        hit_stop = (
            bool(phase1_ids and phase1_ids[-1] in stop_ids)
            or phase1_finish == "stop"
        )
        exhausted = not hit_stop and (
            phase1_finish == "length" or len(phase1_ids) >= phase1_max
        )

        forced_final = False
        prefill_ids: list[int] = []
        phase2_ids: list[int] = []
        phase2_finish = ""
        discarded_phase1_ids: list[int] = []
        combined_ids = list(phase1_ids)

        if exhausted:
            forced_final = True
            if not self._contains_subsequence(
                phase1_ids, self._encode_harmony(self.GPTOSS_FINAL_CHANNEL_INDICATOR)
            ):
                # A length stop can land in the middle of Harmony's multi-token
                # final-channel header.  Appending the forced header after that
                # fragment creates an invalid stream (for example,
                # ``<|channel|><normal text>``).  Remove only a suffix that is a
                # strict prefix of one of the valid final headers, then inject
                # the complete Discover prefill below.
                phase1_ids, discarded_phase1_ids = self._trim_incomplete_final_header(
                    phase1_ids
                )
                end_ids = self._encode_harmony("<|end|>")
                if self._ends_with(phase1_ids, end_ids):
                    prefill = (
                        self.PHASE2_PREFILL
                        + "<|start|>assistant<|channel|>final<|message|>"
                    )
                else:
                    prefill = self.PHASE2_PREFILL + self.GPTOSS_FINAL_MARKER
                prefill_ids = self._encode_harmony(prefill)

            phase2_prompt_ids = prompt_ids + phase1_ids + prefill_ids
            phase2_max = self.context_window - len(phase2_prompt_ids) - self.context_buffer
            if phase2_max <= 0:
                raise RuntimeError(
                    "No GPT-OSS phase-2 budget remains after forced-final prefill: "
                    f"context_window={self.context_window}, prompt={len(prompt_ids)}, "
                    f"phase1={len(phase1_ids)}, prefill={len(prefill_ids)}, "
                    f"buffer={self.context_buffer}"
                )
            phase2_data = self._post_json(
                "/completions",
                self._completion_payload(
                    request=request,
                    prompt_ids=phase2_prompt_ids,
                    max_tokens=phase2_max,
                ),
            )
            phase2_choice = (phase2_data.get("choices") or [{}])[0]
            phase2_ids = self._choice_token_ids(phase2_choice, phase="phase2")
            phase2_finish = str(phase2_choice.get("finish_reason") or "")
            combined_ids = phase1_ids + prefill_ids + phase2_ids

        reasoning, text, harmony_parse_fallback = self._parse_harmony_output_resilient(
            combined_ids=combined_ids,
            phase1_ids=phase1_ids,
            phase2_ids=phase2_ids,
            forced_final=forced_final,
            injected_prefill=bool(prefill_ids),
        )
        completion_tokens = (
            len(phase1_ids)
            + len(discarded_phase1_ids)
            + len(prefill_ids)
            + len(phase2_ids)
        )
        usage = {
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": completion_tokens,
            "total_tokens": len(prompt_ids) + completion_tokens,
            "phase1_completion_tokens": len(phase1_ids),
            "phase1_discarded_header_tokens": len(discarded_phase1_ids),
            "phase2_prefill_tokens": len(prefill_ids),
            "phase2_completion_tokens": len(phase2_ids),
        }
        final_finish = phase2_finish if forced_final else phase1_finish
        metadata = {
            "provider": "openai_compatible",
            **request.metadata,
            "two_phase": True,
            "forced_final": forced_final,
            "phase1_finish_reason": phase1_finish,
            "phase2_finish_reason": phase2_finish or None,
            "phase1_max_context_tokens": self.phase1_max_tokens,
            "context_window": self.context_window,
            "context_buffer": self.context_buffer,
            "harmony_parse_fallback": harmony_parse_fallback,
        }
        return LLMResponse(
            text=text,
            model=phase1_data.get("model") or request.model,
            finish_reason=final_finish,
            reasoning=reasoning,
            usage=usage,
            metadata=metadata,
        )

    def _completion_payload(
        self,
        *,
        request: LLMRequest,
        prompt_ids: list[int],
        max_tokens: int,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model,
            "prompt": prompt_ids,
            "temperature": request.temperature,
            "max_tokens": int(max_tokens),
            "stop_token_ids": self._harmony_stop_token_ids(),
            "return_token_ids": True,
            "skip_special_tokens": False,
        }
        for key in ("top_p", "top_k", "min_p"):
            if key in self.request_options:
                payload[key] = self.request_options[key]
        return payload

    @staticmethod
    def _choice_token_ids(choice: dict[str, Any], *, phase: str) -> list[int]:
        token_ids = choice.get("token_ids")
        if not isinstance(token_ids, list) or not all(
            isinstance(token_id, int) for token_id in token_ids
        ):
            raise RuntimeError(
                f"GPT-OSS two-phase {phase} response did not include integer token_ids; "
                "vLLM >= 0.17 with return_token_ids support is required"
            )
        return list(token_ids)

    def _render_harmony_prompt(self, request: LLMRequest) -> list[int]:
        try:
            from vllm.entrypoints.openai.parser.harmony_utils import (
                get_system_message,
                parse_chat_inputs_to_harmony_messages,
                render_for_completion,
            )
        except ImportError as exc:
            raise RuntimeError(
                "GPT-OSS two-phase execution requires vLLM's Harmony utilities"
            ) from exc
        reasoning_effort = self.request_options.get("reasoning_effort")
        messages = [get_system_message(reasoning_effort=reasoning_effort)]
        messages.extend(
            parse_chat_inputs_to_harmony_messages(
                [
                    {"role": "system", "content": request.system},
                    {"role": "user", "content": request.user},
                ]
            )
        )
        return list(render_for_completion(messages))

    @staticmethod
    def _harmony_stop_token_ids() -> list[int]:
        try:
            from vllm.entrypoints.openai.parser.harmony_utils import (
                get_stop_tokens_for_assistant_actions,
            )
        except ImportError as exc:
            raise RuntimeError(
                "GPT-OSS two-phase execution requires vLLM's Harmony utilities"
            ) from exc
        return list(get_stop_tokens_for_assistant_actions())

    @staticmethod
    def _encode_harmony(text: str) -> list[int]:
        try:
            from vllm.entrypoints.openai.parser.harmony_utils import get_encoding
        except ImportError as exc:
            raise RuntimeError(
                "GPT-OSS two-phase execution requires vLLM's Harmony utilities"
            ) from exc
        return list(get_encoding().encode(text, allowed_special="all"))

    @staticmethod
    def _parse_harmony_output(token_ids: list[int]) -> tuple[str, str]:
        try:
            from vllm.entrypoints.openai.parser.harmony_utils import parse_chat_output
        except ImportError as exc:
            raise RuntimeError(
                "GPT-OSS two-phase execution requires vLLM's Harmony utilities"
            ) from exc
        reasoning, final_content, _is_tool_call = parse_chat_output(token_ids)
        return str(reasoning or ""), str(final_content or "")

    def _parse_harmony_output_resilient(
        self,
        *,
        combined_ids: list[int],
        phase1_ids: list[int],
        phase2_ids: list[int],
        forced_final: bool,
        injected_prefill: bool,
    ) -> tuple[str, str, bool]:
        try:
            reasoning, text = self._parse_harmony_output(combined_ids)
            return reasoning, text, False
        except RuntimeError as exc:
            # openai_harmony.HarmonyError derives from RuntimeError.  Do not
            # hide configuration/import failures raised by our wrapper.
            if exc.__class__.__name__ != "HarmonyError":
                raise
            if not forced_final:
                raise

        # The final phase starts immediately after a final-channel message
        # marker, so its decoded generated tokens are exactly the final answer.
        # This fallback handles arbitrary length cuts inside a Harmony header
        # without losing an otherwise valid executor answer.
        phase1_reasoning = self._best_effort_reasoning(phase1_ids)
        if injected_prefill:
            return phase1_reasoning, self._decode_harmony(phase2_ids), True

        # If phase 1 had already entered the final channel, preserve any final
        # content it emitted before the cutoff and append the continuation.
        try:
            reasoning, partial_final = self._parse_harmony_output(phase1_ids)
        except RuntimeError:
            reasoning, partial_final = phase1_reasoning, ""
        return reasoning, partial_final + self._decode_harmony(phase2_ids), True

    def _best_effort_reasoning(self, token_ids: list[int]) -> str:
        try:
            reasoning, _final = self._parse_harmony_output(token_ids)
            if reasoning:
                return reasoning
        except RuntimeError:
            pass
        decoded = self._decode_harmony(token_ids)
        return re.sub(r"<\|[^|]+\|>", "", decoded).strip()

    @staticmethod
    def _decode_harmony(token_ids: list[int]) -> str:
        try:
            from vllm.entrypoints.openai.parser.harmony_utils import get_encoding
        except ImportError as exc:
            raise RuntimeError(
                "GPT-OSS two-phase execution requires vLLM's Harmony utilities"
            ) from exc
        return str(get_encoding().decode(token_ids))

    def _trim_incomplete_final_header(
        self, token_ids: list[int]
    ) -> tuple[list[int], list[int]]:
        header_variants = (
            self.GPTOSS_FINAL_MARKER,
            "<|start|>assistant<|channel|>final<|message|>",
            self.GPTOSS_FINAL_CHANNEL_INDICATOR,
        )
        longest = 0
        for header in header_variants:
            header_ids = self._encode_harmony(header)
            for prefix_len in range(1, len(header_ids)):
                if self._ends_with(token_ids, header_ids[:prefix_len]):
                    longest = max(longest, prefix_len)
        if not longest:
            return list(token_ids), []
        return list(token_ids[:-longest]), list(token_ids[-longest:])

    @staticmethod
    def _contains_subsequence(tokens: list[int], pattern: list[int]) -> bool:
        if not pattern or len(pattern) > len(tokens):
            return False
        return any(
            tokens[index : index + len(pattern)] == pattern
            for index in range(len(tokens) - len(pattern) + 1)
        )

    @staticmethod
    def _ends_with(tokens: list[int], suffix: list[int]) -> bool:
        return bool(suffix) and len(suffix) <= len(tokens) and tokens[-len(suffix) :] == suffix


_LOCAL_PIPELINE_CACHE: dict[str, Callable] = {}
_LOCAL_PIPELINE_CACHE_LOCK = threading.Lock()
_LOCAL_VLLM_CACHE: dict[str, Any] = {}
_LOCAL_VLLM_GENERATE_LOCKS: dict[str, threading.Lock] = {}
_LOCAL_VLLM_BATCHERS: dict[str, _LocalVLLMBatcher] = {}
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

    def _get_batcher(self) -> _LocalVLLMBatcher:
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
        text = (
            str(generated[-1].get("content", ""))
            if generated and isinstance(generated[-1], dict)
            else str(generated)
        )
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
        if config.get("phase1_max_tokens") is not None:
            return GPTOSSTwoPhaseOpenAICompatibleLLMClient(config)
        return OpenAICompatibleLLMClient(config)
    if provider in {"local", "transformers", "local_transformers"}:
        return LocalTransformersLLMClient(config)
    if provider in {"local_vllm", "vllm"}:
        return LocalVLLMLLMClient(config)
    return UnconfiguredLLMClient()
