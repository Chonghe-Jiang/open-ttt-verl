from __future__ import annotations

import math

from guidance_ttt.state import LibraryNode


def puct_score(*, node: LibraryNode, parent_visits: int, puct_c: float) -> float:
    exploitation = float(node.value)
    exploration = float(puct_c) * math.sqrt(math.log(parent_visits + 1.0) / (node.visits + 1.0))
    return exploitation + exploration


def choose_child(parent: LibraryNode, children: list[LibraryNode], *, puct_c: float) -> LibraryNode:
    if not children:
        return parent
    min_value = min(child.value for child in children)
    max_value = max(child.value for child in children)
    value_span = max(max_value - min_value, 1e-12)

    def normalized_score(child: LibraryNode) -> float:
        exploitation = (child.value - min_value) / value_span
        exploration = float(puct_c) * math.sqrt(math.log(parent.visits + 1.0) / (child.visits + 1.0))
        return exploitation + exploration

    return max(children, key=lambda child: (normalized_score(child), child.id))
