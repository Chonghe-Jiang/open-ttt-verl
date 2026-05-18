# Frontier-CS benchmarking — DeepSeek-R1-0528-Qwen3-8B + TTT-Discover (faithful)

## Results: Controlled Comparison (Base vs TTT-Discover, Same 19 Tasks)

Same model, same 19 Frontier-CS tasks, same evaluator as Yuchen's STaR comparison. The TTT-Discover column reports rewards from the **best step** of the TTT run (paper's argmax_i protocol, Algorithm 1 line 12).

Rewards are continuous scores (0.0–1.0) from the Frontier-CS evaluator. Each task has 8 rollouts.

Method (faithful port of arxiv 2601.16175):
- vLLM (TP=8) for sampling, HF + PEFT for LoRA gradient (rank 32), LoRA hot-reloaded into vLLM each step
- PUCT prioritization over state archive (Appendix A.2)
- Entropic objective with adaptive β solving KL(qβ‖uniform)=ln(2) by bisection (Appendix A.1)
- KL penalty against base policy, importance-sampling correction for sampler/learner mismatch
- Adam(lr=4e-5, β1=0.9, β2=0.95, ε=1e-8); 10 steps × 8 rollouts/step (paper uses 50×512; 1/640 budget)

### In-Distribution Tasks (6 tasks)

| Task | Base avg | Base max | TTT avg | TTT max | Δ avg |
| --- | --- | --- | --- | --- | --- |
| cbl__high_av_loose_dl_small | 0.0000 | 0.0000 | 0.2263 | 0.2800 | **+0.2263** |
| cbl__low_av_tight_dl_large | 0.0200 | 0.1600 | 0.0988 | 0.4000 | **+0.0788** |
| cbl__mixed_av_loose_dl_large | 0.0413 | 0.3300 | 0.2513 | 0.3600 | **+0.2100** |
| cbl_multi__high_av_loose_dl_small | 0.0000 | 0.0000 | 0.2382 | 0.2725 | **+0.2382** |
| cbl_multi__high_av_tight_dl_small | 0.0341 | 0.2725 | 0.0000 | 0.0000 | -0.0341 |
| cbl_multi__low_av_loose_dl_small | 0.1098 | 0.8785 | 0.7687 | 0.8785 | **+0.6589** |
| **TOTAL** | **0.0342** | **0.8785** | **0.2639** | **0.8785** | **+0.2297** |

#### Per-rollout reward distributions (In-Distribution)

| Task | Base rollouts | TTT-Discover rollouts (best step) |
| --- | --- | --- |
| cbl__high_av_loose_dl_small | 0 0 0 0 0 0 0 0 | 0 .13 .28 .28 .28 .28 .28 .28 |
| cbl__low_av_tight_dl_large | 0 0 0 0 0 0 0 .16 | 0 0 0 0 0 0 .39 .4 |
| cbl__mixed_av_loose_dl_large | 0 0 0 0 0 0 0 .33 | 0 0 .33 .33 .33 .33 .33 .36 |
| cbl_multi__high_av_loose_dl_small | 0 0 0 0 0 0 0 0 | 0 .27 .27 .27 .27 .27 .27 .27 |
| cbl_multi__high_av_tight_dl_small | 0 0 0 0 0 0 0 .27 | 0 0 0 0 0 0 0 0 |
| cbl_multi__low_av_loose_dl_small | 0 0 0 0 0 0 0 .88 | 0 .88 .88 .88 .88 .88 .88 .88 |

### Out-of-Distribution Tasks (11 OOD)

| Task | Base avg | Base max | TTT avg | TTT max | Δ avg |
| --- | --- | --- | --- | --- | --- |
| fused_linear_ce | 0.0000 | 0.0000 | 0.0000 | 0.0000 | +0.0000 |
| gemm_opt__annoying | 0.0000 | 0.0000 | 0.0000 | 0.0000 | +0.0000 |
| gemm_opt__k_skewed | 0.0000 | 0.0000 | 0.0000 | 0.0000 | +0.0000 |
| gemm_opt__rectangles | 0.0000 | 0.0000 | 0.0000 | 0.0000 | +0.0000 |
| gemm_opt__squares | 0.0000 | 0.0000 | 0.0000 | 0.0000 | +0.0000 |
| gemm_opt__transformerish | 0.0000 | 0.0000 | 0.0000 | 0.0000 | +0.0000 |
| llm_sql__large | 0.0000 | 0.0000 | 0.1575 | 0.6886 | **+0.1575** |
| poc_gen__heap_uaf | 0.0000 | 0.0000 | (running) | (running) | (running) |
| poc_gen__uninit_value | 0.0000 | 0.0000 | (running) | (running) | (running) |
| vdb_pareto__low_latency | 0.0000 | 0.0000 | (running) | (running) | (running) |
| vdb_pareto__recall80_lat | 0.0000 | 0.0000 | (running) | (running) | (running) |
| **TOTAL** | **0.0000** | **0.0000** | **0.0225** | **0.6886** | **+0.0225** |

#### Per-rollout reward distributions (Out-of-Distribution)

| Task | Base rollouts | TTT-Discover rollouts (best step) |
| --- | --- | --- |
| fused_linear_ce | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 |
| gemm_opt__annoying | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 |
| gemm_opt__k_skewed | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 |
| gemm_opt__rectangles | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 |
| gemm_opt__squares | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 |
| gemm_opt__transformerish | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 |
| llm_sql__large | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 .13 .44 .69 |
| poc_gen__heap_uaf | 0 0 0 0 0 0 0 0 | (no data) |
| poc_gen__uninit_value | 0 0 0 0 0 0 0 0 | (no data) |
| vdb_pareto__low_latency | 0 0 0 0 0 0 0 0 | (no data) |
| vdb_pareto__recall80_lat | 0 0 0 0 0 0 0 0 | (no data) |

## Side-by-side: TTT-Discover vs STaR (ID tasks)

| Task | Yuchen base | Yuchen STaR | Our base | Our TTT-Discover |
| --- | --- | --- | --- | --- |
| cbl__high_av_loose_dl_small | 0.1250 | 0.0417 | 0.0000 | 0.2263 |
| cbl__low_av_tight_dl_large | 0.0488 | 0.2050 | 0.0200 | 0.0988 |
| cbl__mixed_av_loose_dl_large | 0.1400 | 0.2375 | 0.0413 | 0.2513 |
| cbl_multi__high_av_loose_dl_small | 0.0681 | 0.2637 | 0.0000 | 0.2382 |
| cbl_multi__high_av_tight_dl_small | 0.0681 | 0.0822 | 0.0341 | 0.0000 |
| cbl_multi__low_av_loose_dl_small | 0.4392 | 0.3294 | 0.1098 | 0.7687 |
| **TOTAL** | **0.1482** | **0.1933** | **0.0342** | **0.2639** |

- Δ STaR (Yuchen): +0.0451
- Δ TTT-Discover (ours): **+0.2297**
