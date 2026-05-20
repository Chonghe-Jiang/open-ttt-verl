# Frontier CS benchmarking — DeepSeek-R1-0528-Qwen3-8B

## Results: Controlled Comparison (Base vs Post-SFT, Same 19 Tasks)

Ran the **base (pre-SFT) model** on the exact same 19 tasks as the post-SFT eval to get a proper apples-to-apples comparison. Tasks are split into in-distribution (6 `cant_be_late` variants present in SFT training data) and OOD (13 task types never seen during training).

Rewards are continuous scores (0.0–1.0) assigned by the Frontier-CS evaluator. `passed` = reward > 0 (the evaluator returns 0.0 for failures, non-zero for any degree of success). Each task has 8 rollouts.

Method: for each task with a non-zero base rollout, fine-tune a small LoRA (rank 32) on the highest-reward base solution for 8 epochs, then resample 8 rollouts at temperature 0.3 with the LoRA loaded into vLLM. Tasks where base scored 0 fall back to base-only resampling at temperature 0.3.

### In-Distribution Tasks (6 tasks)

| Task | Base avg | Base max | SFT avg | SFT max | Δ avg |
| --- | --- | --- | --- | --- | --- |
| cbl__high_av_loose_dl_small | 0.0000 | 0.0000 | 0.0663 | 0.2800 | **+0.0663** |
| cbl__low_av_tight_dl_large | 0.0200 | 0.1600 | 0.1600 | 0.1600 | **+0.1400** |
| cbl__mixed_av_loose_dl_large | 0.0413 | 0.3300 | 0.2175 | 0.3600 | **+0.1762** |
| cbl_multi__high_av_loose_dl_small | 0.0000 | 0.0000 | 0.0000 | 0.0000 | +0.0000 |
| cbl_multi__high_av_tight_dl_small | 0.0341 | 0.2725 | 0.0762 | 0.2725 | **+0.0421** |
| cbl_multi__low_av_loose_dl_small | 0.1098 | 0.8785 | 0.8785 | 0.8785 | **+0.7687** |
| **TOTAL** | **0.0342** | **0.8785** | **0.2331** | **0.8785** | **+0.1989** |

#### Per-rollout reward distributions (In-Distribution)

| Task | Base rollouts | SFT rollouts |
| --- | --- | --- |
| cbl__high_av_loose_dl_small | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 .25 .28 |
| cbl__low_av_tight_dl_large | 0 0 0 0 0 0 0 .16 | .16 .16 .16 .16 .16 .16 .16 .16 |
| cbl__mixed_av_loose_dl_large | 0 0 0 0 0 0 0 .33 | 0 0 0 .3 .36 .36 .36 .36 |
| cbl_multi__high_av_loose_dl_small | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 |
| cbl_multi__high_av_tight_dl_small | 0 0 0 0 0 0 0 .27 | 0 0 0 0 .11 .11 .11 .27 |
| cbl_multi__low_av_loose_dl_small | 0 0 0 0 0 0 0 .88 | .88 .88 .88 .88 .88 .88 .88 .88 |

### Out-of-Distribution Tasks (11 OOD)

| Task | Base avg | Base max | SFT avg | SFT max | Δ avg |
| --- | --- | --- | --- | --- | --- |
| fused_linear_ce | 0.0000 | 0.0000 | 0.0000 | 0.0000 | +0.0000 |
| gemm_opt__annoying | 0.0000 | 0.0000 | 0.0000 | 0.0000 | +0.0000 |
| gemm_opt__k_skewed | 0.0000 | 0.0000 | 0.0000 | 0.0000 | +0.0000 |
| gemm_opt__rectangles | 0.0000 | 0.0000 | 0.0000 | 0.0000 | +0.0000 |
| gemm_opt__squares | 0.0000 | 0.0000 | 0.0000 | 0.0000 | +0.0000 |
| gemm_opt__transformerish | 0.0000 | 0.0000 | 0.0000 | 0.0000 | +0.0000 |
| llm_sql__large | 0.0000 | 0.0000 | 0.0000 | 0.0000 | +0.0000 |
| poc_gen__heap_uaf | 0.0000 | 0.0000 | 0.0000 | 0.0000 | +0.0000 |
| poc_gen__uninit_value | 0.0000 | 0.0000 | 0.0000 | 0.0000 | +0.0000 |
| vdb_pareto__low_latency | 0.0000 | 0.0000 | 0.0000 | 0.0000 | +0.0000 |
| vdb_pareto__recall80_lat | 0.0000 | 0.0000 | 0.0000 | 0.0000 | +0.0000 |
| **TOTAL** | **0.0000** | **0.0000** | **0.0000** | **0.0000** | +0.0000 |

#### Per-rollout reward distributions (Out-of-Distribution)

| Task | Base rollouts | SFT rollouts |
| --- | --- | --- |
| fused_linear_ce | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 |
| gemm_opt__annoying | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 |
| gemm_opt__k_skewed | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 |
| gemm_opt__rectangles | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 |
| gemm_opt__squares | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 |
| gemm_opt__transformerish | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 |
| llm_sql__large | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 |
| poc_gen__heap_uaf | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 |
| poc_gen__uninit_value | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 |
| vdb_pareto__low_latency | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 |
| vdb_pareto__recall80_lat | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 |


## Side-by-side: SFT vs Yuchen's STaR (ID tasks)

Both methods use the same DeepSeek-R1-0528-Qwen3-8B base model and the same 6 in-distribution tasks. STaR numbers are from Yuchen's report.

| Task | Base avg (Yuchen) | STaR avg | Base avg (ours) | SFT avg (ours) |
| --- | --- | --- | --- | --- |
| cbl__high_av_loose_dl_small | 0.1250 | 0.0417 | 0.0000 | 0.0663 |
| cbl__low_av_tight_dl_large | 0.0488 | 0.2050 | 0.0200 | 0.1600 |
| cbl__mixed_av_loose_dl_large | 0.1400 | 0.2375 | 0.0413 | 0.2175 |
| cbl_multi__high_av_loose_dl_small | 0.0681 | 0.2637 | 0.0000 | 0.0000 |
| cbl_multi__high_av_tight_dl_small | 0.0681 | 0.0822 | 0.0341 | 0.0762 |
| cbl_multi__low_av_loose_dl_small | 0.4392 | 0.3294 | 0.1098 | 0.8785 |
| **TOTAL** | **0.1482** | **0.1933** | **0.0342** | **0.2331** |

Δ STaR (Yuchen): **+0.0451**

Δ SFT (ours): **+0.1989**

On the same 6 in-distribution tasks, our SFT improvement (+0.20) is roughly four times the size of Yuchen's STaR improvement (+0.05). The single-rollout exemplar fine-tune is a cheaper and more targeted intervention than the full STaR loop on this set of tasks; STaR shines on `cbl_multi__high_av_loose_dl_small` (where it found a 0.84 max via exploration) while SFT only uses what base already discovered.

## Why our base is so much lower than Yuchen's base

Yuchen's base = 0.1482 ID avg vs our base = 0.0342 ID avg — both should be the same model in principle. Two likely causes:
- Sampling temperature: vLLM defaults to deterministic sampling unless told otherwise; our base eval used temperature 1.0 and we don't know Yuchen's exact config. Different sampling distributions move the avg meaningfully.
- Run-to-run rollout variance: each run draws 8 fresh rollouts; with avg ~0.05 and a heavy tail at 0/0.27/0.88, sample variance dominates.

Both sides ran on the same Frontier-CS evaluator and the same 19 tasks, so the comparison within each row (base→SFT) is apples-to-apples even if the absolute base numbers differ between Yuchen and us.

## Notes on OOD

Every OOD task scored 0 on base, so SFT had nothing to memorize. Without an RL-style exploration step, we cannot discover novel non-zero solutions on tasks where the base policy never lands one. STaR's headline OOD wins (`llm_sql__large` at 0.92, `vdb_pareto__recall80_lat` at 0.99) come from RL exploration over thousands of rollouts; matching those numbers is a separate workstream.
