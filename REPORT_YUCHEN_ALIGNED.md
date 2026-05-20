# Frontier-CS — Yuchen-aligned comparison (Base vs STaR vs TTT-Discover)

All three approaches use the **same protocol**:
- Base model: DeepSeek-R1-0528-Qwen3-8B
- Iterative tool calling rollout, 3 turns × 16384 tokens, temperature 0.9
- Prompt includes starter code (initial_greedy.py / seed solution)
- 8 rollouts/task, score = max across turns of each rollout

**TTT-Discover setup**: trained on **4 tasks** matching Yuchen's STaR (kelp PR#11 `iterations=1, batch_size=4, seed=1` from full 66-task pool), then evaluated on all 17 ID/OOD tasks (4 train + 13 generalization).

Train tasks (matching `random.Random(1).sample(yuchen_pool_66, 4)`):
  - `cbl__mixed_av_loose_dl_large_oh` (TRAIN)
  - `cbl_multi__high_av_tight_dl_small_oh` (TRAIN)
  - `cbl_multi__low_av_loose_dl_small_oh` (TRAIN)
  - `gemm_opt__transformerish` (TRAIN)

### In-Distribution Tasks (6)

| Task | Yuchen base avg/max | Yuchen STaR avg/max | Our base avg/max | Our TTT avg/max | Δ TTT vs base |
| --- | --- | --- | --- | --- | --- |
| cbl__high_av_loose_dl_small | 0.1250/0.2500 | 0.0417/0.2500 | 0.1288/0.2800 | 0.1325/0.2800 | **+0.0038** |
| cbl__low_av_tight_dl_large | 0.0488/0.3900 | 0.2050/0.4000 | 0.2162/0.4000 | 0.0887/0.4000 | -0.1275 |
| cbl__mixed_av_loose_dl_large (TRAIN) | 0.1400/0.3800 | 0.2375/0.3800 | 0.1900/0.3800 | 0.1425/0.3800 | -0.0475 |
| cbl_multi__high_av_loose_dl_small | 0.0681/0.2725 | 0.2637/0.8435 | 0.0000/0.0000 | 0.1605/0.7396 | **+0.1605** |
| cbl_multi__high_av_tight_dl_small (TRAIN) | 0.0681/0.2725 | 0.0822/0.2725 | 0.1139/0.6386 | 0.1981/0.7675 | **+0.0842** |
| cbl_multi__low_av_loose_dl_small (TRAIN) | 0.4392/0.8785 | 0.3294/0.8785 | 0.2500/1.0000 | 0.0000/0.0000 | -0.2500 |
| **TOTAL** | **0.1482/0.8785** | **0.1933/0.8785** | **0.1498/1.0000** | **0.1204/0.7675** | **-0.0294** |

#### Per-rollout reward distributions (In-Distribution)

| Task | Our base | Our TTT-Discover (zero-shot) |
| --- | --- | --- |
| cbl__high_av_loose_dl_small | 0 0 0 0 .25 .25 .25 .28 | 0 0 0 0 .25 .25 .28 .28 |
| cbl__low_av_tight_dl_large | 0 0 0 .31 .31 .31 .4 .4 | 0 0 0 0 0 0 .31 .4 |
| cbl__mixed_av_loose_dl_large | 0 0 0 0 .38 .38 .38 .38 | 0 0 0 0 0 .38 .38 .38 |
| cbl_multi__high_av_loose_dl_small | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 .27 .27 .74 |
| cbl_multi__high_av_tight_dl_small | 0 0 0 0 0 0 .27 .64 | 0 0 0 0 .27 .27 .27 .77 |
| cbl_multi__low_av_loose_dl_small | 0 0 0 0 0 0 1 1 | 0 0 0 0 0 0 0 0 |

### Out-of-Distribution Tasks (11)

| Task | Yuchen base avg/max | Yuchen STaR avg/max | Our base avg/max | Our TTT avg/max | Δ TTT vs base |
| --- | --- | --- | --- | --- | --- |
| fused_linear_ce | n/a | n/a | 0.0000/0.0000 | 0.0000/0.0000 | +0.0000 |
| gemm_opt__annoying | n/a | n/a | 0.0000/0.0000 | 0.0000/0.0000 | +0.0000 |
| gemm_opt__k_skewed | n/a | n/a | 0.0000/0.0000 | 0.0000/0.0000 | +0.0000 |
| gemm_opt__rectangles | n/a | n/a | 0.0000/0.0000 | 0.0000/0.0000 | +0.0000 |
| gemm_opt__squares | n/a | n/a | 0.0000/0.0000 | 0.0000/0.0000 | +0.0000 |
| gemm_opt__transformerish (TRAIN) | n/a | n/a | 0.0000/0.0000 | 0.0000/0.0000 | +0.0000 |
| llm_sql__large | n/a | n/a | 0.1769/0.6329 | 0.0164/0.1312 | -0.1606 |
| poc_gen__heap_uaf | n/a | n/a | 0.0000/0.0000 | 0.0000/0.0000 | +0.0000 |
| poc_gen__uninit_value | n/a | n/a | 0.0000/0.0000 | 0.0000/0.0000 | +0.0000 |
| vdb_pareto__low_latency | n/a | n/a | 0.0000/0.0000 | 0.1430/0.5769 | **+0.1430** |
| vdb_pareto__recall80_lat | n/a | n/a | 0.3317/0.9951 | 0.0000/0.0000 | -0.3317 |
| **TOTAL** | **0.0000/0.0000** | **0.0000/0.0000** | **0.0462/0.9951** | **0.0145/0.5769** | **-0.0317** |

#### Per-rollout reward distributions (Out-of-Distribution)

| Task | Our base | Our TTT-Discover (zero-shot) |
| --- | --- | --- |
| fused_linear_ce | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 |
| gemm_opt__annoying | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 |
| gemm_opt__k_skewed | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 |
| gemm_opt__rectangles | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 |
| gemm_opt__squares | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 |
| gemm_opt__transformerish | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 |
| llm_sql__large | 0 0 .13 .13 .17 .63 | 0 0 0 0 0 0 0 .13 |
| poc_gen__heap_uaf | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 |
| poc_gen__uninit_value | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 |
| vdb_pareto__low_latency | 0 0 0 0 0 0 | 0 0 0 0 0 0 .57 .58 |
| vdb_pareto__recall80_lat | 0 0 1 | 0 0 0 0 0 0 0 0 |
