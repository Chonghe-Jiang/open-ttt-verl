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


def c5_raw_score(h_values: np.ndarray) -> float:
    h_values = np.asarray(h_values, dtype=np.float64)
    dx = 2.0 / int(h_values.shape[0])
    return float(np.max(np.correlate(h_values, 1.0 - h_values, mode="full") * dx))


def baseline_raw_score(n_points: int = 2) -> float:
    h_values = np.ones(n_points, dtype=np.float64) * 0.5
    return c5_raw_score(h_values)


def _random_perturbed_constant(rng: np.random.Generator) -> np.ndarray:
    n_points = int(rng.integers(40, 100))
    for _ in range(100):
        perturbation = rng.uniform(-0.4, 0.4, size=n_points)
        h_values = 0.5 + perturbation - float(np.mean(perturbation))
        if np.all((0.0 <= h_values) & (h_values <= 1.0)):
            return h_values.astype(np.float64)
    h_values = np.ones(n_points, dtype=np.float64) * 0.5
    perturbation = rng.uniform(-0.4, 0.4, size=n_points)
    h_values += perturbation - float(np.mean(perturbation))
    h_values = np.clip(h_values, 0.0, 1.0)
    target = n_points / 2.0
    for _ in range(100):
        diff = float(target - h_values.sum())
        if abs(diff) < 1e-12:
            break
        free = (h_values > 1e-12) & (h_values < 1.0 - 1e-12)
        if not np.any(free):
            free = np.ones_like(h_values, dtype=bool)
        h_values[free] += diff / float(np.count_nonzero(free))
        h_values = np.clip(h_values, 0.0, 1.0)
    return h_values.astype(np.float64)


def create_root_node(*, seed: int | None = None) -> LibraryNode:
    rng = np.random.default_rng(seed)
    h_values = _random_perturbed_constant(rng)
    n_points = int(h_values.shape[0])
    raw_score = c5_raw_score(h_values)
    root = make_root_node(problem_id="erdos", raw_score=raw_score, reward=1.0 / (1e-8 + raw_score))
    root.metadata.update(
        {
            "n_points": n_points,
            "h_values": h_values.tolist(),
            "c5_bound": raw_score,
            "initialization": "random_perturbed_constant",
        }
    )
    return root
