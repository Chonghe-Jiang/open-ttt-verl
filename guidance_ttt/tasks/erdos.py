from __future__ import annotations

import numpy as np

from guidance_ttt.state import LibraryNode, make_root_node


ERDOS_PROBLEM_PROMPT = """You are solving the Erdos minimum overlap problem.

Find a step function h: [0, 2] -> [0, 1] that minimizes:

C5 = max_k integral h(x)(1 - h(x+k)) dx

Discretize h as n_points samples over [0, 2]. The verifier expects Python code
with:

def run(seed=42, budget_s=..., **kwargs):
    return (h_values, c5_bound, n_points)

Constraints:
- 0 <= h[i] <= 1
- sum(h) == n_points / 2
- c5_bound must match max(np.correlate(h, 1-h, mode="full") * (2.0 / n_points))

Lower raw C5 is better. The reward used for RL is 1 / (1e-8 + C5).
"""


def baseline_raw_score(n_points: int = 2) -> float:
    h_values = np.ones(n_points, dtype=np.float64) * 0.5
    dx = 2.0 / n_points
    return float(np.max(np.correlate(h_values, 1.0 - h_values, mode="full") * dx))


def create_root_node() -> LibraryNode:
    raw_score = baseline_raw_score()
    return make_root_node(problem_id="erdos", raw_score=raw_score, reward=1.0 / (1e-8 + raw_score))

