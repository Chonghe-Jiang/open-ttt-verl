# TTT-Discover on Frontier-CS — DeepSeek-R1-0528-Qwen3-8B

Replicates Yuchen's STaR table layout. Rewards are continuous scores in [0,1] (frontier-cs/100).

Base column: vanilla model, no TTT. TTT column: TTT-Discover (paper Yuksekgonul et al. 2026, arxiv 2601.16175).

`passed` = reward > 0 implicitly. Each task has 8 rollouts.


## Results

### In-Distribution Tasks (6 tasks)

| Task | Base avg | Base max | TTT avg | TTT max | Δ avg |
| --- | --- | --- | --- | --- | --- |
| cbl__high_av_loose_dl_small_oh | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| cbl__low_av_tight_dl_large_oh | 0.0200 | 0.1600 | 0.0200 | 0.1600 | 0.0000 |
| cbl__mixed_av_loose_dl_large_oh | 0.0413 | 0.3300 | 0.0413 | 0.3300 | 0.0000 |
| cbl_multi__high_av_loose_dl_small_oh | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| cbl_multi__high_av_tight_dl_small_oh | 0.0341 | 0.2725 | 0.0341 | 0.2725 | 0.0000 |
| cbl_multi__low_av_loose_dl_small_oh | 0.1098 | 0.8785 | n/a | n/a | n/a |
| **TOTAL** | **0.0342** | **0.8785** | **0.0191** | **0.3300** | **-0.0151** |

#### Per-rollout reward distributions (ID)

| Task | Base rollouts | TTT rollouts |
| --- | --- | --- |
| cbl__high_av_loose_dl_small_oh | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 |
| cbl__low_av_tight_dl_large_oh | 0 0 0 0 0 0 0 .16 | 0 0 0 0 0 0 0 .16 |
| cbl__mixed_av_loose_dl_large_oh | 0 0 0 0 0 0 0 .33 | 0 0 0 0 0 0 0 .33 |
| cbl_multi__high_av_loose_dl_small_oh | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 |
| cbl_multi__high_av_tight_dl_small_oh | 0 0 0 0 0 0 0 .27 | 0 0 0 0 0 0 0 .27 |
| cbl_multi__low_av_loose_dl_small_oh | 0 0 0 0 0 0 0 .88 | (no data) |


### Out-of-Distribution Tasks (13 OOD)

| Task | Base avg | Base max | TTT avg | TTT max | Δ avg |
| --- | --- | --- | --- | --- | --- |
| fused_linear_ce | 0.0000 | 0.0000 | n/a | n/a | n/a |
| gemm_opt__annoying | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| gemm_opt__k_skewed | 0.0000 | 0.0000 | n/a | n/a | n/a |
| gemm_opt__rectangles | 0.0000 | 0.0000 | n/a | n/a | n/a |
| gemm_opt__squares | 0.0000 | 0.0000 | n/a | n/a | n/a |
| gemm_opt__transformerish | 0.0000 | 0.0000 | n/a | n/a | n/a |
| llm_sql__large | 0.0000 | 0.0000 | n/a | n/a | n/a |
| poc_gen__heap_uaf | 0.0000 | 0.0000 | n/a | n/a | n/a |
| poc_gen__uninit_value | 0.0000 | 0.0000 | n/a | n/a | n/a |
| vdb_pareto__low_latency | 0.0000 | 0.0000 | n/a | n/a | n/a |
| vdb_pareto__recall80_lat | 0.0000 | 0.0000 | n/a | n/a | n/a |
| cbl__high_av_loose_dl_small | 0.0000 | 0.0000 | n/a | n/a | n/a |
| **TOTAL** | **0.0000** | **0.0000** | **0.0000** | **0.0000** | **0.0000** |

#### Per-rollout reward distributions (OOD)

| Task | Base rollouts | TTT rollouts |
| --- | --- | --- |
| fused_linear_ce | 0 0 0 0 0 0 0 0 | (no data) |
| gemm_opt__annoying | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 |
| gemm_opt__k_skewed | 0 0 0 0 0 0 0 0 | (no data) |
| gemm_opt__rectangles | 0 0 0 0 0 0 0 0 | (no data) |
| gemm_opt__squares | 0 0 0 0 0 0 0 0 | (no data) |
| gemm_opt__transformerish | 0 0 0 0 0 0 0 0 | (no data) |
| llm_sql__large | 0 0 0 0 0 0 0 0 | (no data) |
| poc_gen__heap_uaf | 0 0 0 0 0 0 0 0 | (no data) |
| poc_gen__uninit_value | 0 0 0 0 0 0 0 0 | (no data) |
| vdb_pareto__low_latency | 0 0 0 0 0 0 0 0 | (no data) |
| vdb_pareto__recall80_lat | 0 0 0 0 0 0 0 0 | (no data) |
| cbl__high_av_loose_dl_small | 0 0 0 0 0 0 0 0 | (no data) |

