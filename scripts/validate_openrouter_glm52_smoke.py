#!/usr/bin/env python3
"""Validate GLM-5.2 provenance, API health, parsing, and request concurrency telemetry."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import median


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--expected-children", type=int, default=128)
    parser.add_argument("--min-parse-fraction", type=float, default=0.95)
    args = parser.parse_args()

    payload = json.loads((args.output_dir / "library.json").read_text())
    entries = [entry for entry in (payload.get("entries") or {}).values() if isinstance(entry, dict)]
    roots = [entry for entry in entries if int(entry.get("timestep") or 0) == 0]
    children = [entry for entry in entries if int(entry.get("timestep") or 0) > 0]
    errors: list[str] = []

    if len(roots) != 1:
        errors.append(f"bootstrap entries={len(roots)}, expected 1")
    if len(children) != args.expected_children:
        errors.append(f"child entries={len(children)}, expected {args.expected_children}")

    model_count = 0
    response_model_count = 0
    metrics_count = 0
    retry_counts: list[int] = []
    elapsed: list[float] = []
    providers: Counter[str] = Counter()
    finish_reasons: Counter[str] = Counter()
    execution_errors = 0
    parsed = 0
    reasoning = 0
    valid = 0
    prompt_tokens = 0
    completion_tokens = 0
    reasoning_tokens = 0
    reported_cost = 0.0
    costs_reported = 0

    for entry in children:
        metadata = entry.get("metadata") or {}
        response_metadata = metadata.get("execution_response_metadata") or {}
        usage = metadata.get("execution_response_usage") or {}
        model_count += metadata.get("execution_model") == "z-ai/glm-5.2"
        response_model_count += response_metadata.get("api_response_model") == "z-ai/glm-5.2"
        metrics_count += "api_elapsed_s" in response_metadata
        if "api_retry_count" in response_metadata:
            retry_counts.append(int(response_metadata["api_retry_count"]))
        if "api_elapsed_s" in response_metadata:
            elapsed.append(float(response_metadata["api_elapsed_s"]))
        if response_metadata.get("api_response_provider"):
            providers[str(response_metadata["api_response_provider"])] += 1
        finish_reasons[str(response_metadata.get("finish_reason") or "missing")] += 1
        execution_errors += entry.get("verifier_status") == "execution_error"
        parsed += bool(str(entry.get("solution") or "").strip()) and bool(
            str(entry.get("summary") or "").strip()
        )
        reasoning += bool(str(entry.get("execution_thinking") or "").strip())
        valid += entry.get("verifier_status") == "valid"
        prompt_tokens += int(usage.get("prompt_tokens") or 0)
        completion_tokens += int(usage.get("completion_tokens") or 0)
        details = usage.get("completion_tokens_details") or {}
        reasoning_tokens += int(details.get("reasoning_tokens") or usage.get("reasoning_tokens") or 0)
        if usage.get("cost") is not None:
            reported_cost += float(usage["cost"])
            costs_reported += 1

    if model_count != len(children):
        errors.append(f"GLM-5.2 model entries={model_count}, expected {len(children)}")
    if response_model_count != len(children):
        errors.append(
            f"OpenRouter responses from GLM-5.2={response_model_count}, expected {len(children)}"
        )
    if metrics_count != len(children):
        errors.append(f"request telemetry entries={metrics_count}, expected {len(children)}")
    if execution_errors:
        errors.append(f"execution API/parse errors={execution_errors}, expected 0")
    parse_fraction = parsed / len(children) if children else 0.0
    if parse_fraction < args.min_parse_fraction:
        errors.append(
            f"parsed solution+summary fraction={parse_fraction:.3f}, "
            f"expected >= {args.min_parse_fraction:.3f}"
        )
    if not valid:
        errors.append("no rollout passed FrontierCS verification")

    scores = [float(entry["verifier_raw_score"]) for entry in children if entry.get("verifier_raw_score") is not None]
    report = {
        "children": len(children),
        "valid_children": valid,
        "parsed_children": parsed,
        "parse_fraction": round(parse_fraction, 6),
        "reasoning_children": reasoning,
        "execution_errors": execution_errors,
        "finish_reasons": dict(finish_reasons),
        "upstream_providers": dict(providers),
        "requests_with_retries": sum(count > 0 for count in retry_counts),
        "total_retries": sum(retry_counts),
        "latency_s_p50": round(median(elapsed), 3) if elapsed else None,
        "latency_s_p95": round(_percentile(elapsed, 0.95), 3) if elapsed else None,
        "latency_s_max": round(max(elapsed), 3) if elapsed else None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "reported_cost_usd": round(reported_cost, 6) if costs_reported else None,
        "best_raw_score": max(scores) if scores else None,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        raise SystemExit("OpenRouter GLM-5.2 smoke validation failed: " + "; ".join(errors))


if __name__ == "__main__":
    main()
