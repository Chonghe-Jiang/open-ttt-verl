from __future__ import annotations

from typing import Protocol

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


def make_llm_client(config: dict) -> BaseLLMClient:
    provider = (config or {}).get("provider", "mock")
    if provider == "mock":
        return MockLLMClient()
    return UnconfiguredLLMClient()
