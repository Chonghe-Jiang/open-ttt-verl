# Rollout `ff19d9c9-3cc6-40f9-a492-ee2b41573be8`

## Metadata

- `id`: `ff19d9c9-3cc6-40f9-a492-ee2b41573be8`
- `timestep`: `18`
- `parent_id`: `6b4d84cf-a029-456a-8e91-5a2436afc909`
- `verifier_status`: `valid`
- `failure_mode`: `null`
- `verifier_raw_score`: `0.38115897991766123`
- `verifier_reward`: `2.623577106802655`
- `verifier_message`: `C5 bound: 0.381159`

## Guidance

```text
Preserve the current best valid construction with raw_score = 0.38115897991766123.  
Do not restart from the constant baseline.  
Include known information directly in the guidance: current best raw score, best valid solution/profile, verifier constraints, verifier normalization, and the strict target.  

Hypothesis: Controlled local perturbations around the current best valid h can strictly improve raw C5 without losing verifier validity.  
Plan:  
1. Use the best valid h profile as initialization, not the 0.5 baseline.  
2. Search only small deterministic perturbations that preserve box, sum, verifier normalization, and symmetry constraints.  
3. Optimize only the independent half variables and immediately score each candidate with the official verifier.  

What to preserve: current best valid low-C5 structure and any mirror/complement symmetry.  
What to change to beat raw score 0.38115897991766123: make controlled local repairs/perturbations around the best valid profile.  
Expected verifier signal: strictly lower raw C5 than 0.38115897991766123.
```

## Solution

```python
import numpy as np

def project_to_box_sum(h, target):
    """Project h onto [0,1] while preserving the sum to target."""
    h = np.clip(np.asarray(h, dtype=float), 0.0, 1.0)
    for _ in range(200):
        diff = float(target - h.sum())
        if abs(diff) < 1e-12:
            break
        free = (h > 1e-12) & (h < 1.0 - 1e-12)
        if not np.any(free):
            free = np.ones_like(h, dtype=bool)
        h[free] += diff / float(np.count_nonzero(free))
        h = np.clip(h, 0.0, 1.0)
    return h

def c5_score(h):
    n = int(len(h))
    return float(np.max(np.correlate(h, 1.0 - h, mode="full") * (2.0 / n)))

def run(seed=42, budget_s=1, **kwargs):
    """
    Return a step function h on [0,2] discretized into n_points samples
    that minimizes the C5 objective.
    """
    # Start from the best known construction (raw C5 ≈ 0.38115897991766123)
    n_points = 19
    target = n_points / 2.0
    h = np.array([
        0.9999877603639307, 0.9871689928539907, 0.6538052901835218,
        0.26496040864134757, 0.7921008518715473, 0.22244708724960485,
        0.43594987520258144, 0.31249596343709657, 0.013288180619505931,
        0.10447938652930353, 0.04501365418702798, 0.3119089519044072,
        0.4358802114412871, 0.22261948541845558, 0.7918283420600218,
        0.268186249932381, 0.6427085234261372, 0.9952347090654273,
        0.9999360756124249,
    ], dtype=float)

    h = project_to_box_sum(h, target)
    best_h = h.copy()
    best_c5 = c5_score(best_h)

    # Deterministic local search with small random perturbations
    rng = np.random.default_rng(seed)
    for _ in range(2000000):
        noise = rng.normal(scale=5e-4, size=n_points)
        cand = best_h + noise
        cand = project_to_box_sum(cand, target)
        cand_c5 = c5_score(cand)
        if cand_c5 < best_c5:
            best_h = cand
            best_c5 = cand_c5

    # Optional SLSQP refinement if SciPy is available
    try:
        from scipy.optimize import minimize
        constraints = ({"type": "eq", "fun": lambda x: float(np.sum(x) - target)},)
        result = minimize(
            c5_score,
            best_h,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * n_points,
            constraints=constraints,
            options={"maxiter": 300, "ftol": 1e-13, "disp": False},
        )
        if result.success:
            cand = project_to_box_sum(result.x, target)
            cand_c5 = c5_score(cand)
            if cand_c5 < best_c5:
                best_h = cand
                best_c5 = cand_c5
    except Exception:
        pass

    best_h = project_to_box_sum(best_h, target)
    c5_bound = c5_score(best_h)
    return [float(x) for x in best_h], float(c5_bound), int(n_points)
```

## Summary

```text
Outcome hypothesis: execution did not provide summary.
Reusable idea: Preserve the current best valid construction with raw_score = 0.38115897991766123.  
Do not restart from the constant baseline.  
Include known information directly in the guidance: current best raw score, best valid solution/profile, verifier constraints, verifier normalization, and the strict targ
Risk / possible failure mode: missing_summary
What future guidance should preserve: selected library context.
What future guidance should change: request clearer executable plan.
```

## Analysis

这条 rollout 是一条有效 verifier 结果，但它没有改进当前 best。它的 `verifier_raw_score` 等于 `0.38115897991766123`，和 guidance 中写入的 current best 完全一致，所以 reward 也只是复现 best：`2.623577106802655`。这说明 execution model 主要选择了“从 best profile 初始化并局部搜索”的保守策略，而不是产生新的结构。

guidance 的核心方向是合理的：不要从常数 baseline 重启，而是围绕当前最优 profile 做小扰动，同时保持 box constraint、sum normalization 和 verifier 一致性。这个方向能显著降低 parse/error 风险，也能避免回退到高 C5 的无效探索。

solution 的行为也符合这个思路：它固定 `n_points = 19`，从一个已知低 C5 的向量开始，用 `project_to_box_sum` 保证 `0 <= h[i] <= 1` 且 `sum(h) == n_points / 2`，然后用 `np.correlate(h, 1-h, mode="full") * (2.0 / n)` 计算 C5。最后返回 `(h_values, c5_bound, n_points)`，所以 verifier 可以直接解析并验证。

主要问题是探索强度有限。局部随机扰动的 scale 是 `5e-4`，而且每次只接受严格降低 `best_c5` 的候选；如果当前点已经在这个离散 `n_points=19` 空间的局部平台附近，随机搜索很容易只复现原 score。SLSQP refinement 也未必能改进，因为目标函数是 max over correlations，非光滑，SLSQP 可能停在同一 active-constraint set。

另一个问题是 summary 质量低。entry 的 `summary` 明确写了 `execution did not provide summary` 和 `missing_summary`，所以后续 library selection 能复用的高层信息比较少。可复用的部分主要来自 `guidance` 和 `reusable_idea`，而不是 execution 自己总结出的新 insight。

后续如果要让类似 rollout 更有机会改进，可以让 guidance 明确要求 execution 输出候选 profile 的 active correlation lags，针对这些 active lags 做约束松弛或成对坐标更新，而不是只做 isotropic Gaussian perturbation。也可以要求尝试多个 `n_points` 或多种 structured perturbation，但这会增加 verifier 失败和搜索耗时风险。
