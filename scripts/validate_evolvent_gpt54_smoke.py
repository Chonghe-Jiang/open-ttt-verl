#!/usr/bin/env python3
"""Validate GPT-5.4 provenance, output format, and API health in a full smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CAPACITY_MARKERS = (
    "http 429",
    "http 503",
    "rate limit",
    "rate_limit",
    "too many requests",
    "concurren",
    "capacity",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--model", default="sub2api-gpt-5.4")
    parser.add_argument(
        "--bootstrap-model",
        default=None,
        help="Expected model for the root seed; defaults to --model.",
    )
    args = parser.parse_args()

    payload = json.loads((args.output_dir / "library.json").read_text())
    entries = [entry for entry in (payload.get("entries") or {}).values() if isinstance(entry, dict)]
    bootstrap = [entry for entry in entries if int(entry.get("timestep") or 0) == 0]
    children = [entry for entry in entries if int(entry.get("timestep") or 0) > 0]
    errors = []

    if len(bootstrap) != 1:
        errors.append(f"bootstrap entries={len(bootstrap)}, expected 1")
    expected_models = {
        "bootstrap": args.bootstrap_model or args.model,
        "child": args.model,
    }
    for label, candidates in (("bootstrap", bootstrap), ("child", children)):
        for entry in candidates:
            metadata = entry.get("metadata") or {}
            if metadata.get("execution_model") != expected_models[label]:
                errors.append(f"{label} used execution model {metadata.get('execution_model')!r}")
            if metadata.get("execution_provider") != "openai_compatible":
                errors.append(f"{label} used provider {metadata.get('execution_provider')!r}")
            execution_text = str(metadata.get("execution_text") or "")
            if "<execution_thinking>" in execution_text or "<think>" in execution_text:
                errors.append(f"{label} leaked a thinking XML block into the final answer")

    execution_errors = [
        entry
        for entry in children
        if str(entry.get("verifier_status") or "") == "execution_error"
    ]
    capacity_errors = [
        entry
        for entry in execution_errors
        if any(marker in str(entry.get("verifier_message") or "").lower() for marker in CAPACITY_MARKERS)
    ]
    if capacity_errors:
        errors.append(f"capacity-limited executions={len(capacity_errors)}")
    if execution_errors:
        errors.append(f"total execution API errors={len(execution_errors)}")

    usage_records = [
        (entry.get("metadata") or {}).get("execution_response_usage") or {}
        for entry in children
    ]
    cached_tokens = 0
    for usage in usage_records:
        cached_tokens += int(usage.get("cache_read_input_tokens") or 0)
        details = usage.get("prompt_tokens_details") or {}
        cached_tokens += int(details.get("cached_tokens") or 0)

    report = {
        "bootstrap_entries": len(bootstrap),
        "child_entries": len(children),
        "execution_errors": len(execution_errors),
        "capacity_errors": len(capacity_errors),
        "usage_records": sum(bool(usage) for usage in usage_records),
        "reported_cached_tokens": cached_tokens,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        raise SystemExit("GPT-5.4 smoke validation failed: " + "; ".join(errors))


if __name__ == "__main__":
    main()
