#!/usr/bin/env python3
"""Find the highest GPT-5.4 API concurrency with no capacity-limit errors."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import yaml

from guidance_ttt.llm_client import make_llm_client
from guidance_ttt.state import LLMRequest


CAPACITY_MARKERS = (
    "http 429",
    "http 503",
    "rate limit",
    "rate_limit",
    "too many requests",
    "concurren",
    "capacity",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--levels", default="16,8,4,2,1")
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


async def probe_level(config: dict, concurrency: int, request_count: int) -> dict:
    probe_config = dict(config)
    probe_config["max_retries"] = 0
    client = make_llm_client(probe_config)
    semaphore = asyncio.Semaphore(concurrency)

    async def one(index: int) -> dict:
        async with semaphore:
            started = time.monotonic()
            try:
                response = await client.complete(
                    LLMRequest(
                        system="You are an API health-check assistant. Return only the requested short token.",
                        user=f"Return exactly OK-{index} and nothing else.",
                        model=str(probe_config["model"]),
                        temperature=float(probe_config.get("temperature", 0.2)),
                        max_tokens=32,
                        metadata={"purpose": "concurrency_probe"},
                    )
                )
                return {
                    "ok": bool(response.text.strip()),
                    "elapsed_s": time.monotonic() - started,
                    "usage": response.usage,
                    "error": "" if response.text.strip() else "empty response",
                }
            except Exception as exc:
                return {
                    "ok": False,
                    "elapsed_s": time.monotonic() - started,
                    "usage": {},
                    "error": str(exc),
                }

    started = time.monotonic()
    results = await asyncio.gather(*(one(index) for index in range(request_count)))
    errors = [str(result["error"]) for result in results if not result["ok"]]
    capacity_errors = [
        error for error in errors if any(marker in error.lower() for marker in CAPACITY_MARKERS)
    ]
    non_capacity_errors = [error for error in errors if error not in capacity_errors]
    return {
        "concurrency": concurrency,
        "requests": request_count,
        "successes": request_count - len(errors),
        "errors": errors,
        "capacity_error_count": len(capacity_errors),
        "non_capacity_error_count": len(non_capacity_errors),
        "elapsed_s": time.monotonic() - started,
        "max_request_elapsed_s": max(float(result["elapsed_s"]) for result in results),
    }


async def main_async(args: argparse.Namespace) -> None:
    recipe = yaml.safe_load(args.config.read_text())
    config = dict(recipe["llm"]["execution"])
    levels = [int(value) for value in args.levels.split(",") if value.strip()]
    reports = []
    selected = None
    for concurrency in levels:
        report = await probe_level(config, concurrency, max(args.requests, concurrency))
        reports.append(report)
        print(json.dumps(report, sort_keys=True))
        if report["non_capacity_error_count"]:
            raise SystemExit("GPT-5.4 probe hit a non-capacity API error; lowering concurrency cannot fix it")
        if report["capacity_error_count"] == 0:
            selected = concurrency
            break
    if selected is None:
        raise SystemExit("GPT-5.4 API still returned capacity-limit errors at concurrency=1")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    temp.write_text(f"{selected}\n")
    temp.replace(args.output)
    print(json.dumps({"selected_concurrency": selected, "reports": reports}, indent=2))


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
