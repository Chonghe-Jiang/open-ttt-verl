#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
METRIC = re.compile(r"(?:^| - )(?P<name>[A-Za-z0-9_./-]+):(?P<value>[^ ]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate one completed TriMul actor update.")
    parser.add_argument("trainer_log")
    parser.add_argument("--expected-step", type=int, default=1)
    parser.add_argument("--expected-learning-rate", type=float, default=4.0e-5)
    return parser.parse_args()


def parse_float(metrics: dict[str, str], name: str, errors: list[str]) -> float:
    raw = metrics.get(name)
    if raw is None:
        errors.append(f"missing metric {name}")
        return math.nan
    try:
        value = float(raw)
    except ValueError:
        errors.append(f"non-numeric metric {name}={raw!r}")
        return math.nan
    if not math.isfinite(value):
        errors.append(f"non-finite metric {name}={value}")
    return value


def main() -> None:
    args = parse_args()
    text = ANSI_ESCAPE.sub("", Path(args.trainer_log).read_text(errors="replace"))
    lines = [
        line
        for line in text.splitlines()
        if f"step:{args.expected_step} -" in line
        and f"training/global_step:{args.expected_step}" in line
    ]
    if not lines:
        raise SystemExit(f"No completed step {args.expected_step} metrics in {args.trainer_log}")
    metrics = {match.group("name"): match.group("value") for match in METRIC.finditer(lines[-1])}
    errors: list[str] = []
    global_step = parse_float(metrics, "training/global_step", errors)
    learning_rate = parse_float(metrics, "actor/lr", errors)
    grad_norm = parse_float(metrics, "actor/grad_norm", errors)
    pg_loss = parse_float(metrics, "actor/pg_loss", errors)
    update_actor_s = parse_float(metrics, "timing_s/update_actor", errors)
    total_tokens = parse_float(metrics, "perf/total_num_tokens", errors)
    probs_valid = parse_float(metrics, "training/rollout_probs_diff_valid", errors)
    empty_mask = parse_float(metrics, "policy/empty_optimization_mask", errors)
    aborted_ratio = parse_float(metrics, "response/aborted_ratio", errors)
    kl_loss = parse_float(metrics, "actor/kl_loss", errors)

    if global_step != args.expected_step:
        errors.append(f"training/global_step={global_step}, expected={args.expected_step}")
    if not math.isclose(learning_rate, args.expected_learning_rate, rel_tol=0.0, abs_tol=1e-12):
        errors.append(
            f"actor/lr={learning_rate}, expected={args.expected_learning_rate}"
        )
    if not grad_norm > 0:
        errors.append(f"actor/grad_norm={grad_norm}, expected positive")
    if not update_actor_s > 0:
        errors.append(f"timing_s/update_actor={update_actor_s}, expected positive")
    if not total_tokens > 0:
        errors.append(f"perf/total_num_tokens={total_tokens}, expected positive")
    if probs_valid != 1:
        errors.append(f"training/rollout_probs_diff_valid={probs_valid}, expected=1")
    if empty_mask != 0:
        errors.append(f"policy/empty_optimization_mask={empty_mask}, expected=0")
    if aborted_ratio != 0:
        errors.append(f"response/aborted_ratio={aborted_ratio}, expected=0")
    if kl_loss != 0:
        errors.append(f"actor/kl_loss={kl_loss}, expected=0")

    summary = {
        "trainer_log": str(args.trainer_log),
        "training_global_step": global_step,
        "learning_rate": learning_rate,
        "actor_grad_norm": grad_norm,
        "actor_pg_loss": pg_loss,
        "actor_update_s": update_actor_s,
        "total_tokens": total_tokens,
        "rollout_probs_diff_valid": probs_valid,
        "empty_optimization_mask": empty_mask,
        "aborted_ratio": aborted_ratio,
        "actor_kl_loss": kl_loss,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if errors:
        raise SystemExit("TriMul training update validation failed: " + "; ".join(errors))


if __name__ == "__main__":
    main()
