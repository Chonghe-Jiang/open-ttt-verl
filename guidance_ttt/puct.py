from __future__ import annotations

import math

from guidance_ttt.state import LibraryNode


def compute_scale(nodes: list[LibraryNode], *, initial_ids: set[str] | None = None) -> float:
    """Reward scale used by TTT-Discover's PUCT sampler."""
    if not nodes:
        return 1.0
    candidates = [node for node in nodes if initial_ids is None or node.id not in initial_ids]
    if not candidates:
        candidates = nodes
    values = [float(node.value) for node in candidates]
    return max(max(values) - min(values), 1e-6)


def rank_priors(nodes: list[LibraryNode]) -> dict[str, float]:
    """Rank-based prior P(i), matching discover's descending-value rank weights."""
    if not nodes:
        return {}
    ranked = sorted(enumerate(nodes), key=lambda item: float(item[1].value), reverse=True)
    weights_by_index: dict[int, float] = {}
    total = 0.0
    n_nodes = len(nodes)
    for rank, (index, _node) in enumerate(ranked):
        weight = float(n_nodes - rank)
        weights_by_index[index] = weight
        total += weight
    return {node.id: weights_by_index[index] / total for index, node in enumerate(nodes)}


def archive_puct_score(
    *,
    node: LibraryNode,
    visit_count: int,
    best_reachable_value: float | None,
    prior: float,
    scale: float,
    total_visits: int,
    puct_c: float,
) -> float:
    """TTT-Discover archive PUCT score.

    score(i) = Q(i) + c * scale * P(i) * sqrt(1 + T) / (1 + n[i])
    Q(i) = m[i] if n[i] > 0 else R(i)
    """
    q_value = float(best_reachable_value) if visit_count > 0 and best_reachable_value is not None else float(node.value)
    bonus = float(puct_c) * float(scale) * float(prior) * math.sqrt(1.0 + float(total_visits)) / (1.0 + float(visit_count))
    return q_value + bonus


def rank_archive_nodes(
    nodes: list[LibraryNode],
    *,
    initial_ids: set[str],
    visit_counts: dict[str, int],
    best_reachable_values: dict[str, float],
    total_visits: int,
    puct_c: float,
) -> list[tuple[float, float, LibraryNode, int, float, float, float]]:
    """Return nodes sorted by discover-style PUCT score.

    Tuple layout mirrors discover's logging order:
    (score, value, node, n, Q, P, bonus).
    """
    scale = compute_scale(nodes, initial_ids=initial_ids)
    priors = rank_priors(nodes)
    scored = []
    for node in nodes:
        n_visits = int(visit_counts.get(node.id, 0))
        q_value = (
            float(best_reachable_values[node.id])
            if n_visits > 0 and node.id in best_reachable_values
            else float(node.value)
        )
        prior = float(priors.get(node.id, 0.0))
        bonus = float(puct_c) * scale * prior * math.sqrt(1.0 + float(total_visits)) / (1.0 + float(n_visits))
        score = q_value + bonus
        scored.append((score, float(node.value), node, n_visits, q_value, prior, bonus))
    scored.sort(key=lambda item: (item[0], item[1], item[2].id), reverse=True)
    return scored
