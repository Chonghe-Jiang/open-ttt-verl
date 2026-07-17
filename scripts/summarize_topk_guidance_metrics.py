#!/usr/bin/env python3
"""Write comparable search/guidance metrics for top-k guidance experiments."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from guidance_ttt.strategy import strategy_similarity, strategy_tags


TARGET_SCORE = 81.3115


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def main(output_dir: str) -> None:
    root = Path(output_dir)
    data = json.loads((root / "library.json").read_text())
    entries = list((data.get("entries") or {}).values())
    groups = list((data.get("groups") or {}).values())
    by_step: dict[int, list[dict]] = defaultdict(list)
    for entry in entries:
        by_step[int(entry.get("timestep", 0))].append(entry)

    step_metrics = {}
    first_exceeding_step = None
    all_guidance = []
    all_valid = []
    for step, step_entries in sorted(by_step.items()):
        scored = [entry for entry in step_entries if entry.get("verifier_raw_score") is not None]
        valid = [entry for entry in scored if entry.get("verifier_status") == "valid"]
        best = max((float(entry["verifier_raw_score"]) for entry in scored), default=None)
        if best is not None and best > TARGET_SCORE and first_exceeding_step is None:
            first_exceeding_step = step
        tags = [tag for entry in step_entries for tag in strategy_tags(entry.get("guidance"))]
        step_metrics[str(step)] = {
            "best_verifier_score": best,
            "valid_execution_rate": len(valid) / len(step_entries) if step_entries else None,
            "heuristic_coverage": dict(sorted(Counter(tags).items())),
        }
        all_guidance.extend(str(entry.get("guidance") or "") for entry in step_entries)
        all_valid.extend(valid)

    # Bound the quadratic comparison cost for 50-step runs while retaining a
    # deterministic, recent cross-section of the active search distribution.
    diversity_sample = all_guidance[-512:]
    similarities = [strategy_similarity(left, right) for left, right in combinations(diversity_sample, 2)]
    ref_metadata = [group.get("reference_selection") or {} for group in groups]
    ref_similarities = [
        float(item["reference_strategy_similarity"])
        for item in ref_metadata
        if item.get("reference_strategy_similarity") is not None
    ]
    branch_diverse = [
        item
        for item in ref_metadata
        if len(set(item.get("reference_branch_ids") or [])) == 2
    ]
    metrics = {
        "target_score_to_beat": TARGET_SCORE,
        "best_verifier_score": max(
            (float(entry["verifier_raw_score"]) for entry in entries if entry.get("verifier_raw_score") is not None),
            default=None,
        ),
        "first_step_exceeding_target": first_exceeding_step,
        "valid_execution_rate": len(all_valid) / len(entries) if entries else None,
        "guidance_strategy_diversity": {
            "metric": "1 - mean lexical Jaccard similarity of guidance text; raw guidance/summary are retained for embedding analysis",
            "score": None if _mean(similarities) is None else 1.0 - _mean(similarities),
            "sample_size": len(diversity_sample),
        },
        "heuristic_coverage": dict(sorted(Counter(tag for text in all_guidance for tag in strategy_tags(text)).items())),
        "reference_diversity": {
            "groups_with_two_distinct_branches": len(branch_diverse),
            "groups_with_reference_metadata": len(ref_metadata),
            "mean_strategy_similarity": _mean(ref_similarities),
            "lineage_related_groups": sum(bool(item.get("reference_lineage_related")) for item in ref_metadata),
        },
        "per_step": step_metrics,
    }
    (root / "topk_guidance_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: summarize_topk_guidance_metrics.py OUTPUT_DIR")
    main(sys.argv[1])
