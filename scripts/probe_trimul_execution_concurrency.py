#!/usr/bin/env python3
"""Probe high-thinking Evolvent capacity without generating full candidates."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from omegaconf import OmegaConf

from guidance_ttt.llm_client import make_llm_client
from guidance_ttt.state import LLMRequest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "guidance_ttt/config/"
            "trimul_b200_1gpu_qwen3_8b_evolvent_glm52_thinking_cache_"
            "batch8_group16_1step.yaml"
        ),
    )
    parser.add_argument(
        "--seed-library",
        type=Path,
        default=Path("guidance_ttt/seeds/trimul/glm52_scratch_bootstrap_library.json"),
    )
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--propagation-wait-s", type=float, default=4.0)
    return parser.parse_args()


def load_execution_prompt(path: Path) -> tuple[str, str]:
    payload = json.loads(path.read_text())
    roots = [
        entry
        for entry in (payload.get("entries") or {}).values()
        if isinstance(entry, dict) and int(entry.get("timestep") or 0) == 0
    ]
    if len(roots) != 1:
        raise RuntimeError(f"Expected one root in {path}, found {len(roots)}")
    prompt = (roots[0].get("metadata") or {}).get("execution_prompt") or {}
    system = str(prompt.get("system") or "").strip()
    user = str(prompt.get("user") or "").strip()
    if not system or not user:
        raise RuntimeError(f"Root in {path} has no complete execution prompt")
    return system, user


def cached_tokens(usage: dict) -> int:
    details = usage.get("prompt_tokens_details") or {}
    return int(details.get("cached_tokens") or usage.get("cache_read_input_tokens") or 0)


async def main_async(args: argparse.Namespace) -> None:
    if args.concurrency < 1 or args.requests < 1:
        raise SystemExit("--concurrency and --requests must be positive")
    recipe = OmegaConf.load(args.config)
    execution_config = dict(recipe["llm"]["execution"])
    execution_config["max_retries"] = 0
    client = make_llm_client(execution_config)
    system, user_prefix = load_execution_prompt(args.seed_library)

    def request(user: str, purpose: str) -> LLMRequest:
        return LLMRequest(
            system=system,
            user=user,
            model=str(execution_config["model"]),
            temperature=float(execution_config.get("temperature", 1.0)),
            max_tokens=1,
            metadata={
                "purpose": purpose,
                "accept_partial_stream": True,
            },
        )

    primer = await client.complete(request(user_prefix, "execution_concurrency_primer"))
    await asyncio.sleep(max(0.0, args.propagation_wait_s))
    semaphore = asyncio.Semaphore(args.concurrency)

    async def one(index: int) -> dict[str, object]:
        started = time.monotonic()
        try:
            async with semaphore:
                response = await client.complete(
                    request(
                        f"{user_prefix}\n\n<!-- concurrency-probe-{index} -->",
                        "execution_concurrency_probe",
                    )
                )
            usage = dict(response.usage)
            return {
                "ok": bool(response.reasoning or response.text),
                "elapsed_s": round(time.monotonic() - started, 3),
                "finish_reason": response.finish_reason,
                "has_reasoning": bool(response.reasoning),
                "cached_tokens": cached_tokens(usage),
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "error": "",
            }
        except Exception as exc:
            return {
                "ok": False,
                "elapsed_s": round(time.monotonic() - started, 3),
                "finish_reason": "",
                "has_reasoning": False,
                "cached_tokens": 0,
                "prompt_tokens": 0,
                "error": f"{type(exc).__name__}: {exc}",
            }

    started = time.monotonic()
    rows = await asyncio.gather(*(one(index) for index in range(args.requests)))
    summary = {
        "requested_concurrency": args.concurrency,
        "requests": args.requests,
        "successes": sum(bool(row["ok"]) for row in rows),
        "reasoning_responses": sum(bool(row["has_reasoning"]) for row in rows),
        "cache_hit_responses": sum(int(row["cached_tokens"]) > 0 for row in rows),
        "wall_time_s": round(time.monotonic() - started, 3),
        "primer": {
            "finish_reason": primer.finish_reason,
            "has_reasoning": bool(primer.reasoning),
            "cached_tokens": cached_tokens(dict(primer.usage)),
            "prompt_tokens": int(primer.usage.get("prompt_tokens") or 0),
        },
        "errors": [str(row["error"]) for row in rows if row["error"]],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["successes"] != args.requests or summary["reasoning_responses"] != args.requests:
        raise SystemExit(1)


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
