from __future__ import annotations

import asyncio
import json
import os
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


def make_llm_client(config: dict) -> BaseLLMClient:
    provider = (config or {}).get("provider", "mock")
    if provider == "mock":
        return MockLLMClient()
    if provider in {"openai", "openai_compatible"}:
        return OpenAICompatibleLLMClient(config)
    return UnconfiguredLLMClient()
