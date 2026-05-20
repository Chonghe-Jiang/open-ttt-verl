# TTT-Discover on Frontier-CS — DeepSeek-R1-0528-Qwen3-8B

Replicates Yuchen's STaR table layout. Rewards are continuous scores in [0,1] (frontier-cs/100).

Base column: vanilla model, no TTT. 

`passed` = reward > 0 implicitly. Each task has 8 rollouts.


## Results

### In-Distribution Tasks (6 tasks)

| Task | Base avg | Base max |
| --- | --- | --- |
| cbl__high_av_loose_dl_small_oh | 0.0000 | 0.0000 |
| cbl__low_av_tight_dl_large_oh | 0.0200 | 0.1600 |
| cbl__mixed_av_loose_dl_large_oh | 0.0413 | 0.3300 |
| cbl_multi__high_av_loose_dl_small_oh | 0.0000 | 0.0000 |
| cbl_multi__high_av_tight_dl_small_oh | 0.0341 | 0.2725 |
| cbl_multi__low_av_loose_dl_small_oh | 0.1098 | 0.8785 |
| **TOTAL** | **0.0342** | **0.8785** |

#### Per-rollout reward distributions (ID)

| Task | Base rollouts |
| --- | --- |
| cbl__high_av_loose_dl_small_oh | 0 0 0 0 0 0 0 0 |
| cbl__low_av_tight_dl_large_oh | 0 0 0 0 0 0 0 .16 |
| cbl__mixed_av_loose_dl_large_oh | 0 0 0 0 0 0 0 .33 |
| cbl_multi__high_av_loose_dl_small_oh | 0 0 0 0 0 0 0 0 |
| cbl_multi__high_av_tight_dl_small_oh | 0 0 0 0 0 0 0 .27 |
| cbl_multi__low_av_loose_dl_small_oh | 0 0 0 0 0 0 0 .88 |


### Out-of-Distribution Tasks (13 OOD)

| Task | Base avg | Base max |
| --- | --- | --- |
| fused_linear_ce | 0.0000 | 0.0000 |
| gemm_opt__annoying | 0.0000 | 0.0000 |
| gemm_opt__k_skewed | 0.0000 | 0.0000 |
| gemm_opt__rectangles | 0.0000 | 0.0000 |
| gemm_opt__squares | 0.0000 | 0.0000 |
| gemm_opt__transformerish | 0.0000 | 0.0000 |
| llm_sql__large | 0.0000 | 0.0000 |
| poc_gen__heap_uaf | 0.0000 | 0.0000 |
| poc_gen__uninit_value | 0.0000 | 0.0000 |
| vdb_pareto__low_latency | 0.0000 | 0.0000 |
| vdb_pareto__recall80_lat | 0.0000 | 0.0000 |
| cbl__high_av_loose_dl_small | 0.0000 | 0.0000 |
| **TOTAL** | **0.0000** | **0.0000** |

#### Per-rollout reward distributions (OOD)

| Task | Base rollouts |
| --- | --- |
| fused_linear_ce | 0 0 0 0 0 0 0 0 |
| gemm_opt__annoying | 0 0 0 0 0 0 0 0 |
| gemm_opt__k_skewed | 0 0 0 0 0 0 0 0 |
| gemm_opt__rectangles | 0 0 0 0 0 0 0 0 |
| gemm_opt__squares | 0 0 0 0 0 0 0 0 |
| gemm_opt__transformerish | 0 0 0 0 0 0 0 0 |
| llm_sql__large | 0 0 0 0 0 0 0 0 |
| poc_gen__heap_uaf | 0 0 0 0 0 0 0 0 |
| poc_gen__uninit_value | 0 0 0 0 0 0 0 0 |
| vdb_pareto__low_latency | 0 0 0 0 0 0 0 0 |
| vdb_pareto__recall80_lat | 0 0 0 0 0 0 0 0 |
| cbl__high_av_loose_dl_small | 0 0 0 0 0 0 0 0 |

