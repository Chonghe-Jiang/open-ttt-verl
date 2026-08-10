#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe concurrent official TriMul evaluations.")
    parser.add_argument("--url-file", default=".runtime/trimul_judge_url")
    parser.add_argument("--token-file", default=".secrets/trimul_judge_token")
    parser.add_argument(
        "--seed-library",
        default="guidance_ttt/seeds/trimul/glm52_scratch_bootstrap_library.json",
    )
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--timeout-s", type=float, default=1160)
    return parser.parse_args()


def load_seed_solution(path: Path) -> str:
    payload = json.loads(path.read_text())
    roots = [
        entry
        for entry in (payload.get("entries") or {}).values()
        if isinstance(entry, dict) and int(entry.get("timestep") or 0) == 0
    ]
    if len(roots) != 1 or not str(roots[0].get("solution") or "").strip():
        raise RuntimeError(f"Expected one non-empty root solution in {path}")
    return str(roots[0]["solution"])


async def main() -> None:
    args = parse_args()
    if args.concurrency < 1 or args.requests < 1:
        raise SystemExit("--concurrency and --requests must be positive")
    url = Path(args.url_file).read_text().strip()
    token = Path(args.token_file).read_text().strip()
    solution = load_seed_solution(Path(args.seed_library))
    semaphore = asyncio.Semaphore(args.concurrency)

    async def evaluate(index: int) -> dict[str, object]:
        request = urllib.request.Request(
            url,
            data=json.dumps({"solution": solution, "runner_timeout_s": 520}).encode(),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        started = time.monotonic()

        def send() -> tuple[int, dict]:
            try:
                with urllib.request.urlopen(request, timeout=args.timeout_s) as response:
                    return int(response.status), json.loads(response.read().decode())
            except urllib.error.HTTPError as exc:
                body = exc.read().decode(errors="replace")[:500]
                return int(exc.code), {"error": body}

        async with semaphore:
            status, payload = await asyncio.to_thread(send)
        report = payload.get("report") if isinstance(payload, dict) else None
        return {
            "index": index,
            "http_status": status,
            "elapsed_s": round(time.monotonic() - started, 3),
            "all_correct": bool(isinstance(report, dict) and report.get("all_correct")),
            "score_us": report.get("score_us") if isinstance(report, dict) else None,
            "evaluation_gpu": payload.get("evaluation_gpu") if isinstance(payload, dict) else None,
            "error": payload.get("error") if isinstance(payload, dict) else None,
        }

    started = time.monotonic()
    rows = await asyncio.gather(*(evaluate(index) for index in range(args.requests)))
    result = {
        "requested_concurrency": args.concurrency,
        "requests": args.requests,
        "wall_time_s": round(time.monotonic() - started, 3),
        "successful": sum(row["http_status"] == 200 and row["all_correct"] for row in rows),
        "rows": rows,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["successful"] != args.requests:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
