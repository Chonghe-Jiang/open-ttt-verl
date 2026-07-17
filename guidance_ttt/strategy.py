"""Lightweight, reproducible strategy labels and similarity for search telemetry."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable


_TOKEN_RE = re.compile(r"[a-z][a-z0-9_-]{2,}", re.IGNORECASE)
_STOPWORDS = frozenset(
    {
        "the", "and", "for", "with", "from", "that", "this", "into", "using", "use", "used", "are",
        "was", "will", "would", "should", "can", "could", "each", "current", "next", "new", "also",
        "piece", "pieces", "placement", "placements", "board", "score", "guidance", "algorithm",
    }
)
_STRATEGY_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("aspect_ratio", ("aspect ratio", "aspect-ratio", "target ratio", "ratio control")),
    ("ler", ("largest empty rectangle", " ler", "ler-")),
    ("lookahead", ("look-ahead", "lookahead", "predictive", "future projection", "future piece")),
    ("orientation", ("orientation", "rotation", "prun")),
    ("gap_scoring", ("gap", "empty space", "space fit")),
    ("restart_backtracking", ("restart", "backtrack", "undo placement", "rollback")),
    ("piece_ordering", ("piece ordering", "ordering", "order pieces", "order of pieces")),
    ("local_search", ("local search", "local rearrangement", "local reopt", "swap")),
    ("exact_subproblem", ("exact search", "dynamic programming", "branch and bound", "integer program")),
    ("shape_features", ("irregularity", "boundary density", "compactness", "shape compatibility")),
    ("density", ("density", "packing density", "fill ratio")),
)


def strategy_tags(text: str | None) -> list[str]:
    normalized = str(text or "").lower()
    return [tag for tag, patterns in _STRATEGY_PATTERNS if any(pattern in normalized for pattern in patterns)]


def strategy_tokens(text: str | None) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(str(text or "")) if token.lower() not in _STOPWORDS}


def strategy_similarity(left: str | None, right: str | None) -> float:
    """Jaccard similarity of guidance/summary tokens; 0 means no lexical overlap."""
    left_tokens, right_tokens = strategy_tokens(left), strategy_tokens(right)
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def strategy_frequency(texts: Iterable[str | None]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(strategy_tags(text))
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
