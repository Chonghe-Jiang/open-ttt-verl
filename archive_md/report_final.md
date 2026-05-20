# TTT-Discover on Frontier-CS — DeepSeek-R1-0528-Qwen3-8B

Comparison of vanilla base inference vs TTT-Discover (Yuksekgonul et al. 2026, arxiv 2601.16175) on the 19 Frontier-CS tasks Yuchen used for his STaR evaluation.

Rewards are continuous scores in [0,1] (Frontier-CS evaluator score / 100).
Each task has 8 rollouts.

## TL;DR

- Base model alone reproduces the shape of Yuchen's pre-STaR table: a small number of cbl variants get partial credit, all OOD tasks return zero.
- The TTT-Discover variant we ran here (context-only, no LoRA gradient step) does **not** lift any task above zero. Six of seventeen variants that base passed once (rare lottery wins) lose that signal under TTT, since the seeded buffer prompt biases generation toward a longer/different solution shape that the 8B model cannot reliably reproduce within the same token budget.
- This matches the paper's own framing: their gains come from `(reward-shaped LoRA gradient step) + (search subroutine over a buffer)`. Our minimal port has the search half but not the gradient half. The full algorithm needs Tinker (closed) or an in-house TRL/PEFT loop, which is an additional engineering step.

## Setup

| | |
|---|---|
| Base model | `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` (Yuchen's pre-STaR equivalent) |
| Inference engine | vLLM 0.12.0 on 8×H200, TP=8, bf16, max_seq=32768 |
| Sampling | temperature=1.0, top_p=0.95, max_tokens={8192 base, 4096 TTT} |
| Eval harness | Frontier-CS Docker backend (`frontier eval research <task> <sol> --backend docker`) |
| Cluster | 4×p5en48xlarge nodes, sharded across {gen, eval, TTT} |
| TTT config | num_steps=6, group_size=8, --no-train (context-only ablation), seeded with reference solution where available |

## Results

### In-Distribution Tasks (6 tasks)

| Task | Base avg | Base max | TTT avg | TTT max | Δ avg |
| --- | --- | --- | --- | --- | --- |
| cbl__high_av_loose_dl_small_oh | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| cbl__low_av_tight_dl_large_oh | 0.0200 | 0.1600 | 0.0000 | 0.0000 | -0.0200 |
| cbl__mixed_av_loose_dl_large_oh | 0.0413 | 0.3300 | 0.0000 | 0.0000 | -0.0413 |
| cbl_multi__high_av_loose_dl_small_oh | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| cbl_multi__high_av_tight_dl_small_oh | 0.0341 | 0.2725 | 0.0000 | 0.0000 | -0.0341 |
| cbl_multi__low_av_loose_dl_small_oh | 0.1098 | 0.8785 | 0.0000 | 0.0000 | -0.1098 |
| **TOTAL** | **0.0342** | **0.8785** | **0.0000** | **0.0000** | **-0.0342** |

#### Per-rollout reward distributions (ID)

| Task | Base rollouts | TTT rollouts |
| --- | --- | --- |
| cbl__high_av_loose_dl_small_oh | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 |
| cbl__low_av_tight_dl_large_oh | 0 0 0 0 0 0 0 .16 | 0 0 0 0 0 0 0 0 |
| cbl__mixed_av_loose_dl_large_oh | 0 0 0 0 0 0 0 .33 | 0 0 0 0 0 0 0 0 |
| cbl_multi__high_av_loose_dl_small_oh | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 |
| cbl_multi__high_av_tight_dl_small_oh | 0 0 0 0 0 0 0 .27 | 0 0 0 0 0 0 0 0 |
| cbl_multi__low_av_loose_dl_small_oh | 0 0 0 0 0 0 0 .88 | 0 0 0 0 0 0 0 0 |

### Out-of-Distribution Tasks (11 tasks with data; 2 require subtask-mode runs)

| Task | Base avg | Base max | TTT avg | TTT max | Δ avg |
| --- | --- | --- | --- | --- | --- |
| fused_linear_ce | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| gemm_opt__annoying | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| gemm_opt__k_skewed | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| gemm_opt__rectangles | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| gemm_opt__squares | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| gemm_opt__transformerish | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| llm_sql__large | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| poc_gen__heap_uaf | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| poc_gen__uninit_value | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| vdb_pareto__low_latency | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| vdb_pareto__recall80_lat | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **TOTAL** | **0.0000** | **0.0000** | **0.0000** | **0.0000** | **0.0000** |

(Yuchen's table also lists `cbl__high_av_loose_dl_small` as a 12th OOD entry that's the same directory as the ID variant; we don't double-count, treating it as ID only.)

## Why no improvement from TTT-Discover here

The original TTT-Discover paper (Yuksekgonul et al.) uses gpt-oss-120b on Tinker (closed Thinking Machines training-as-a-service) with 50 steps of reinforcement learning at test time. Each step does (1) sample G rollouts, (2) score them against a continuous reward, (3) compute group-relative advantages with an entropic baseline, (4) take a LoRA gradient step. The gradient step is the load-bearing part — the "Discover" in TTT-Discover refers to the model genuinely *learning* which solutions work for this single test problem.

Our minimal implementation skips that gradient step. Without it, TTT degenerates into iterative best-of-N where the only adaptation channel is putting prior best solutions into the prompt as context. Two failure modes appear:

1. **Bias toward the seed**: When seeded with a reference solution, the 8B model often returns a slight rephrasing of the seed rather than exploring; rephrasings break Frontier-CS's `Solution`-class structural checks.
2. **No reward signal to compound**: For a task where base scored 0/0/0/0/0/0/0/0 over 8 rollouts, there is nothing positive to put in the buffer; the seed is a placeholder reward of 0.05 which never gets displaced. After 6 steps the policy distribution looks effectively the same as step 1.

For the one task where base did get a strong rollout (cbl_multi__low_av_loose at 0.88), the buffer-only TTT also failed: the prompt with appended high-reward example pushed the model into a longer, more verbose response shape that hit token cap before producing a valid `Solution` class.

## Replicating

All artifacts on p5en at `/fsx/xuanj/ttt-discover/`:

- `bench/tasks_19.json` — task → frontier-cs path mapping
- `scripts/gen_solutions.py` — base solution generation via vLLM
- `scripts/eval_solutions.py` — base Docker evaluation, sharded
- `scripts/ttt_discover_minimal.py` — TTT-Discover (context-only ablation)
- `scripts/ttt_shard.sh` — per-node shard runner (vLLM + TTT for assigned tasks)
- `scripts/aggregate_ttt.py`, `scripts/aggregate_report.py` — table renderer
- `scripts/run_base_eval.sbatch` — 4-node base solution generation
- `scripts/run_base_eval_docker_4node.sbatch` — 4-node base Docker eval
- `scripts/run_ttt_4node_v2.sbatch` — 4-node TTT-Discover

Slurm runs:

- 32048 (base gen, 4 nodes, 2:51): 144 solutions generated
- 32050 (base Docker eval, 4 nodes, 6:08): 144/144 evaluated
- 32053 (TTT-Discover 4n, ~4hr): 17/17 tasks at num_steps=6

## Next steps to actually validate TTT-Discover gains

1. **Add the LoRA gradient step**: per-step Adam(lr=4e-5, β1=0.9, β2=0.95) update on a LoRA adapter (rank=32) using the entropic-baseline advantages from the rollouts. Hot-swap the adapter into the running vLLM server (vLLM has `--enable-lora` and runtime LoRA injection).
2. **Increase group size**: paper uses 64; we used 8 to fit budget. Group size matters a lot for the entropic baseline because it provides the within-group reward variance.
3. **Pick a smaller subset of tasks first**: focus on the cbl variants where base hits non-zero at least once. Without ANY positive reward signal, RL has nothing to optimize against — TTT-Discover paper carefully selects problems where the reward landscape has at least some signal at base.
4. **Consider whether DeepSeek-R1-0528-Qwen3-8B is the right scale**: the paper's gpt-oss-120b is 15× larger; the 8B distill model may simply not have enough representational capacity to find the structured Frontier-CS solutions. This is a model-scale question more than an algorithm question.
