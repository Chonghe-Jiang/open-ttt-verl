#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
import urllib.error
import urllib.request


@dataclass
class VLLMServerCheck:
    ok: bool
    message: str
    models: list[str]


def check_vllm_server(base_url: str, model: str, timeout_s: int = 5) -> VLLMServerCheck:
    url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode())
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        return VLLMServerCheck(
            ok=False,
            message=(
                f"Could not reach vLLM server at {url}: {exc}. "
                "Start it with scripts/start_vllm_qwen3_8b.sh."
            ),
            models=[],
        )

    models = [str(item.get("id", "")) for item in payload.get("data", []) if item.get("id")]
    if model not in models:
        return VLLMServerCheck(
            ok=False,
            message=f"vLLM server is reachable, but model {model!r} is not listed. Available models: {models}",
            models=models,
        )
    return VLLMServerCheck(ok=True, message=f"vLLM server is ready with model {model}", models=models)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check a local vLLM OpenAI-compatible server.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--timeout-s", type=int, default=5)
    args = parser.parse_args()

    result = check_vllm_server(args.base_url, args.model, timeout_s=args.timeout_s)
    print(result.message)
    if result.models:
        print("models:", ", ".join(result.models))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

