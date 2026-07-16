#!/usr/bin/env python3
"""Fail a smoke job when rollout and actor probabilities are not aligned."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


METRICS = {
    "diff_mean": "training/rollout_probs_diff_mean",
    "pearson_corr": "training/rollout_actor_probs_pearson_corr",
    "kl": "rollout_corr/kl",
    "ess": "rollout_corr/rollout_is_eff_sample_size",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--max-diff-mean", type=float, default=0.03)
    parser.add_argument("--min-pearson-corr", type=float, default=0.98)
    parser.add_argument("--max-kl", type=float, default=0.05)
    parser.add_argument("--min-ess", type=float, default=0.98)
    return parser.parse_args()


def extract_last_step_metrics(text: str) -> dict[str, float]:
    lines = [line for line in text.splitlines() if "training/rollout_probs_diff_mean:" in line]
    if not lines:
        raise ValueError("no completed-step rollout/actor metrics found")

    line = lines[-1]
    values: dict[str, float] = {}
    for short_name, metric_name in METRICS.items():
        match = re.search(rf"(?:^|\s)-?\s*{re.escape(metric_name)}:([^\s]+)", line)
        if match is None:
            raise ValueError(f"missing {metric_name} in completed-step metrics")
        values[short_name] = float(match.group(1))
    return values


def main() -> None:
    args = parse_args()
    values = extract_last_step_metrics(args.log.read_text(errors="replace"))
    print(
        "rollout/actor parity: "
        f"diff_mean={values['diff_mean']:.6f}, "
        f"pearson_corr={values['pearson_corr']:.6f}, "
        f"kl={values['kl']:.6f}, ess={values['ess']:.6f}"
    )

    failures = []
    if values["diff_mean"] > args.max_diff_mean:
        failures.append(f"diff_mean>{args.max_diff_mean}")
    if values["pearson_corr"] < args.min_pearson_corr:
        failures.append(f"pearson_corr<{args.min_pearson_corr}")
    if values["kl"] > args.max_kl:
        failures.append(f"kl>{args.max_kl}")
    if values["ess"] < args.min_ess:
        failures.append(f"ess<{args.min_ess}")
    if failures:
        raise SystemExit("rollout/actor parity gate failed: " + ", ".join(failures))


if __name__ == "__main__":
    main()
