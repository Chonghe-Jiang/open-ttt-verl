ERDOS_PROBLEM_PROMPT = """You are solving the Erdos minimum overlap problem.

Find a step function h: [0, 2] -> [0, 1] that minimizes:

C5 = max_k integral h(x)(1 - h(x+k)) dx

Discretize h as n_points samples over [0, 2]. The candidate program must expose:

def run(seed=42, budget_s=..., **kwargs):
    return (h_values, c5_bound, n_points)

Constraints:
- 0 <= h[i] <= 1
- sum(h) == n_points / 2
- c5_bound must match max(np.correlate(h, 1-h, mode="full") * (2.0 / n_points))

Lower raw C5 is better. The OpenEvolve score is 1 / (1e-8 + C5).
"""


SYSTEM_MESSAGE = """You are an expert numerical search programmer working on the Erdos minimum overlap problem.

Improve only the evolved construction strategy while preserving the public run(seed=42, budget_s=..., **kwargs)
interface and verifier constraints. Good candidates return a one-dimensional h array with values in [0, 1],
sum(h) = n_points / 2, and a correctly recomputed C5 bound. Optimize for lower C5, not for shorter code.
"""

