#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the cached GLM-5.2 TriMul smoke.")
    parser.add_argument("output_dir")
    parser.add_argument("--expected-children", type=int, default=4)
    parser.add_argument("--expected-groups", type=int, default=1)
    parser.add_argument("--expected-group-size", type=int, default=4)
    parser.add_argument("--minimum-cache-hits", type=int, default=1)
    return parser.parse_args()


def cached_tokens(entry: dict) -> int:
    usage = (entry.get("metadata") or {}).get("execution_response_usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    return int(details.get("cached_tokens") or usage.get("cache_read_input_tokens") or 0)


def main() -> None:
    args = parse_args()
    library_path = Path(args.output_dir) / "library.json"
    payload = json.loads(library_path.read_text())
    discover_compat = bool((payload.get("config") or {}).get("discover_compat", False))
    entries = [entry for entry in (payload.get("entries") or {}).values() if isinstance(entry, dict)]
    roots = [entry for entry in entries if int(entry.get("timestep") or 0) == 0]
    children = [entry for entry in entries if int(entry.get("timestep") or 0) == 1]
    groups = [group for group in (payload.get("groups") or {}).values() if isinstance(group, dict)]
    roles = Counter(
        str(
            (((entry.get("metadata") or {}).get("execution_response_metadata") or {}).get("cache_warm_role"))
            or "missing"
        )
        for entry in children
    )
    finish_reasons = Counter(
        str(
            (((entry.get("metadata") or {}).get("execution_response_metadata") or {}).get("finish_reason"))
            or "missing"
        )
        for entry in children
    )
    cache_hits = sum(cached_tokens(entry) > 0 for entry in children)
    follower_cache_hits = sum(
        cached_tokens(entry) > 0
        and str(
            (((entry.get("metadata") or {}).get("execution_response_metadata") or {}).get(
                "cache_warm_role"
            ))
            or "missing"
        )
        == "follower"
        for entry in children
    )
    prompt_tokens = sum(
        int(
            (((entry.get("metadata") or {}).get("execution_response_usage") or {}).get(
                "prompt_tokens"
            ))
            or 0
        )
        for entry in children
    )
    total_cached_tokens = sum(cached_tokens(entry) for entry in children)
    valid_children = sum(entry.get("verifier_status") == "valid" for entry in children)
    reasoning_children = sum(bool(str(entry.get("execution_thinking") or "").strip()) for entry in children)
    parsed_children = sum(
        bool(str(entry.get("solution") or "").strip())
        and bool(str((entry.get("metadata") or {}).get("raw_model_summary") or "").strip())
        for entry in children
    )
    streaming_children = sum(
        bool(
            ((entry.get("metadata") or {}).get("execution_response_metadata") or {}).get(
                "api_streaming"
            )
        )
        for entry in children
    )
    recovered_children = sum(
        bool(
            ((entry.get("metadata") or {}).get("execution_response_metadata") or {}).get(
                "api_stream_recovered"
            )
        )
        for entry in children
    )
    follower_failures = sum(
        role == "follower"
        and not bool(
            ((entry.get("metadata") or {}).get("execution_response_metadata") or {}).get(
                "cache_prime_succeeded"
            )
        )
        for entry, role in zip(
            children,
            [
                str(
                    (((entry.get("metadata") or {}).get("execution_response_metadata") or {}).get("cache_warm_role"))
                    or "missing"
                )
                for entry in children
            ],
            strict=True,
        )
    )
    errors: list[str] = []
    expected_roots = args.expected_groups if discover_compat else 1
    valid_roots = sum(entry.get("verifier_status") == "valid" for entry in roots)
    if len(roots) != expected_roots or valid_roots != expected_roots:
        errors.append(
            "valid root accounting mismatch: "
            f"roots={len(roots)}, valid_roots={valid_roots}, expected={expected_roots}"
        )
    if len(children) != args.expected_children:
        errors.append(f"children={len(children)}, expected={args.expected_children}")
    if args.expected_groups * args.expected_group_size != args.expected_children:
        errors.append(
            "invalid expected shape: "
            f"{args.expected_groups}*{args.expected_group_size}!={args.expected_children}"
        )
    if len(groups) != args.expected_groups:
        errors.append(f"groups={len(groups)}, expected={args.expected_groups}")
    malformed_groups = 0
    for group in groups:
        archived_children = len(group.get("children") or [])
        malformed = (
            not bool(group.get("finalized"))
            or int(group.get("submitted") or 0) != args.expected_group_size
            or len(group.get("entry_ids") or []) != args.expected_group_size
            or archived_children > args.expected_group_size
        )
        if not discover_compat:
            malformed = malformed or archived_children != args.expected_group_size
        malformed_groups += int(malformed)
    if malformed_groups:
        errors.append(f"malformed or incomplete groups={malformed_groups}")
    expected_puct_updates = args.expected_children if discover_compat else args.expected_groups
    if int(payload.get("puct_T") or 0) != expected_puct_updates:
        errors.append(
            f"puct_T={payload.get('puct_T')}, expected={expected_puct_updates}"
        )
    expected_legacy_roles = Counter({"follower": args.expected_children - 1, "leader": 1})
    primer_owners = int(roles.get("primer_owner") or 0)
    expected_primer_roles = Counter(
        {
            "follower": args.expected_children - primer_owners,
            "primer_owner": primer_owners,
        }
    )
    primer_roles_ok = primer_owners >= 1 and roles == expected_primer_roles
    if roles != expected_legacy_roles and not primer_roles_ok:
        errors.append(f"cache warm roles={dict(roles)}")
    if primer_roles_ok and follower_failures:
        errors.append(f"followers released after failed primer={follower_failures}")
    if follower_cache_hits < args.minimum_cache_hits:
        errors.append(
            f"follower cache hits={follower_cache_hits}, minimum={args.minimum_cache_hits}"
        )
    if reasoning_children != args.expected_children:
        errors.append(f"reasoning children={reasoning_children}, expected={args.expected_children}")
    if parsed_children != args.expected_children:
        errors.append(f"parsed children={parsed_children}, expected={args.expected_children}")
    if streaming_children != args.expected_children:
        errors.append(f"streaming children={streaming_children}, expected={args.expected_children}")
    if finish_reasons != Counter({"stop": args.expected_children}):
        errors.append(f"finish reasons={dict(finish_reasons)}")
    if valid_children < 1:
        errors.append("no child passed the official H100 evaluator")
    summary = {
        "library": str(library_path),
        "discover_compat": discover_compat,
        "roots": len(roots),
        "groups": len(groups),
        "puct_T": int(payload.get("puct_T") or 0),
        "children": len(children),
        "valid_children": valid_children,
        "parsed_children": parsed_children,
        "reasoning_children": reasoning_children,
        "streaming_children": streaming_children,
        "recovered_children": recovered_children,
        "cache_warm_roles": dict(roles),
        "cache_prime_usage": next(
            (
                (((entry.get("metadata") or {}).get("execution_response_metadata") or {}).get(
                    "cache_prime_usage"
                ))
                for entry in children
                if (((entry.get("metadata") or {}).get("execution_response_metadata") or {}).get(
                    "cache_warm_role"
                ))
                == "primer_owner"
            ),
            {},
        ),
        "cache_hit_children": cache_hits,
        "follower_cache_hit_children": follower_cache_hits,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": total_cached_tokens,
        "cached_prompt_fraction": (
            total_cached_tokens / prompt_tokens if prompt_tokens else 0.0
        ),
        "finish_reasons": dict(finish_reasons),
        "best_runtime_us": min(
            (
                float(entry["verifier_raw_score"])
                for entry in entries
                if entry.get("verifier_status") == "valid" and entry.get("verifier_raw_score") is not None
            ),
            default=None,
        ),
        "root_runtime_us": min(
            (
                float(entry["verifier_raw_score"])
                for entry in roots
                if entry.get("verifier_status") == "valid" and entry.get("verifier_raw_score") is not None
            ),
            default=None,
        ),
        "best_child_runtime_us": min(
            (
                float(entry["verifier_raw_score"])
                for entry in children
                if entry.get("verifier_status") == "valid" and entry.get("verifier_raw_score") is not None
            ),
            default=None,
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if errors:
        raise SystemExit("TriMul Evolvent cache smoke failed: " + "; ".join(errors))


if __name__ == "__main__":
    main()
