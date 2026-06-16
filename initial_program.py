"""Initial Erdos minimum-overlap candidate for OpenEvolve."""

from __future__ import annotations

import time

import numpy as np


# EVOLVE-BLOCK-START
def construct_h_values(seed=42, budget_s=1, n_points=256):
    """Return an initial balanced construction over [0, 2]."""
    _ = seed, budget_s
    return np.ones(int(n_points), dtype=np.float64) * 0.5
# EVOLVE-BLOCK-END


def _normalize_h_values(h_values, n_points):
    h = np.asarray(h_values, dtype=np.float64).reshape(-1)
    if h.size != int(n_points):
        h = np.resize(h, int(n_points))
    h = np.nan_to_num(h, nan=0.5, posinf=1.0, neginf=0.0)
    h = np.clip(h, 0.0, 1.0)
    target_sum = int(n_points) / 2.0
    current_sum = float(np.sum(h))
    if current_sum <= 0:
        h = np.ones(int(n_points), dtype=np.float64) * 0.5
    else:
        h = h * (target_sum / current_sum)
        if np.any(h > 1.0):
            h = np.clip(h, 0.0, 1.0)
            for _ in range(8):
                deficit = target_sum - float(np.sum(h))
                if abs(deficit) <= 1e-10:
                    break
                room = np.where(h < 1.0 - 1e-12)[0] if deficit > 0 else np.where(h > 1e-12)[0]
                if room.size == 0:
                    break
                h[room] += deficit / room.size
                h = np.clip(h, 0.0, 1.0)
    return h


def _compute_c5(h_values, n_points):
    h = np.asarray(h_values, dtype=np.float64)
    dx = 2.0 / int(n_points)
    return float(np.max(np.correlate(h, 1.0 - h, mode="full") * dx))


def run(seed=42, budget_s=1, n_points=256, **kwargs):
    start = time.time()
    n_points = int(kwargs.get("points", n_points))
    h_values = construct_h_values(seed=seed, budget_s=budget_s, n_points=n_points)
    h_values = _normalize_h_values(h_values, n_points)
    c5_bound = _compute_c5(h_values, n_points)
    _elapsed = time.time() - start
    return (h_values.tolist(), c5_bound, n_points)


if __name__ == "__main__":
    h, c5, n = run()
    print(f"n_points={n} c5={c5:.8f} sum={sum(h):.8f}")

