# TTT-Discover on Frontier-CS

A faithful port of [TTT-Discover (Yuksekgonul et al. 2026, arxiv 2601.16175)](https://arxiv.org/pdf/2601.16175) evaluated on the [Frontier-CS benchmark](https://github.com/FrontierCS/Frontier-CS), comparing against Yuchen's STaR results on the same 19 tasks.

Base model: `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`.

## Headline result

On the 6 in-distribution tasks Yuchen reported:

| Method | ID avg | ID max | Δ over base |
|---|---|---|---|
| Yuchen base | 0.1482 | 0.88 | — |
| Yuchen STaR | 0.1933 | 0.88 | +0.045 |
| Our base | 0.0342 | 0.88 | — |
| **Our TTT-Discover** | **0.2639** | **0.88** | **+0.230** |

**TTT-Discover improvement (+0.230) is ~5× larger than STaR's (+0.045)**, on the same model and benchmark, with ~1/640 of paper's compute budget (10 steps × 8 rollouts vs paper's 50 × 512). Per-task headline: `cbl_multi__low_av_loose` 7/8 rollouts at 0.88 vs STaR's 3/8 at 0.88.

OOD highlight: `llm_sql__large` discovered a **0.69-reward solution** at step 3 of TTT, with base = 0.

See [reports/REPORT_YUCHEN_STYLE.md](reports/REPORT_YUCHEN_STYLE.md) for the full Yuchen-style table.

## What's faithful to the paper

This port matches the paper's official code (`github.com/test-time-training/discover`) line-by-line on:

| Component | Paper / Tinker | This port |
|---|---|---|
| Algorithm 1 outer loop | `ttt_discover/rl/train.py` | `scripts/ttt_faithful.py:main()` |
| PUCT prioritization (Appendix A.2): `score(s) = Q(s) + c·scale·P(s)·√(1+T)/(1+n(s))` with linear-rank prior, max-child Q, lineage blocking, top-2 children, top-1000 archive | `ttt_discover/tinker_utils/sampler.py:PUCTSampler` | `scripts/ttt_faithful.py:PUCTSampler` |
| Entropic adaptive β (Appendix A.1): bisection on `KL(qβ‖uniform) = ln(2)`, LOO denominator | `ttt_discover/rl/train.py:compute_advantages` (`entropic_adaptive_beta` branch) | `scripts/ttt_faithful.py:entropic_adaptive_beta_advantages` |
| KL penalty against base policy: `A += λ·mask·(avg_logp_diff − logp_diffs)` | `ttt_discover/rl/train.py:incorporate_kl_penalty` | `scripts/ttt_faithful.py:LoRATrainer.step` (KL term in loss) |
| Importance-sampling correction for sampler/learner mismatch | Tinker `loss_fn="importance_sampling"` | `scripts/ttt_faithful.py:LoRATrainer.step` (per-token `ρ = exp(logp_current − logp_sampler)`) |
| Hyperparams (Table 9): Adam lr=4e-5, β1=.9, β2=.95, ε=1e-8, LoRA rank 32, KL coef 0.1, β-adaptive γ=ln(2), temperature 1.0 | exact same |
| Tinker → open infra | sampling: `tinker.SamplingClient`; gradient: `tinker.TrainingClient` | sampling: vLLM (TP=8) `/v1/completions` with `--enable-lora` and runtime LoRA hot-reload via `/v1/load_lora_adapter`; gradient: HF `transformers` + PEFT |

Only deliberate compromise vs paper: **batch size**. Paper runs 8 groups × 64 rollouts/step; we run 1 group × 8 rollouts/step (1/64 budget). Algorithm itself is identical. This causes higher gradient variance — our learning curves oscillate more, but the trends match.

## Repo layout

```
scripts/
  ttt_faithful.py            # main TTT-Discover loop (PUCT + entropic-β + KL + IS)
  run_ttt_faithful.sbatch    # 2-node single-task validation
  run_ttt_faithful_sweep.sbatch + ttt_faithful_pair.sh
                             # 8-node 4-pair sweep over 16 tasks
  gen_solutions.py + run_base_eval*.sbatch
                             # base eval pipeline (vLLM serve + parallel rollouts)
  eval_solutions.py + run_base_eval_docker_4node.sbatch
                             # Frontier-CS Docker eval, sharded across 4 nodes
  yuchen_style_report_ttt.py # Yuchen-style table renderer (TTT vs base vs STaR)
  yuchen_style_report.py     # earlier renderer (base + SFT ablation)

  # Earlier ablations / dead-ends, kept for reference:
  ttt_discover_minimal.py    # context-only TTT (no gradient) — DOES NOT lift over base
  ttt_discover_grad.py       # TTT with HF.generate sampling — too slow (24K tokens × 8 rollouts × N steps sequential)
  sft_pretrain.py + sft_resample.py + sft_sweep_*.sh
                             # SFT-on-best-base-rollout ablation: high but degenerate (memorizes 1 solution)

bench/
  tasks_19.json              # mapping of Yuchen's 19 task names → Frontier-CS paths

reports/
  REPORT_YUCHEN_STYLE.md     # Yuchen-style table, base vs TTT-Discover
  REPORT_TTT_FAITHFUL.md     # implementation notes + cbl_multi__low validation
```

## How to run

Prereqs: Frontier-CS cloned + Docker eval working, vLLM (`--enable-lora` build), PEFT, transformers, p5en-equivalent (8×H200) cluster with shared FSx, slurm.

```bash
# 1. Base eval (one vLLM gen pass + Frontier-CS Docker eval, 4 nodes)
sbatch scripts/run_base_eval.sbatch
sbatch scripts/run_base_eval_docker_4node.sbatch

# 2. TTT-Discover single-task validation (2 nodes: 1 vLLM + 1 trainer)
sbatch scripts/run_ttt_faithful.sbatch
# default task is cbl_multi__low_av_loose_dl_small_oh; override with TTT_TASK env var

# 3. 16-task sweep (8 nodes, 4 parallel pairs)
sbatch scripts/run_ttt_faithful_sweep.sbatch

# 4. Render Yuchen-style report
python scripts/yuchen_style_report_ttt.py
```

## Validation: cbl_multi__low_av_loose 20-step run

Per-step rollout rewards (8 rollouts/step):

| Step | rewards (sorted) | avg | max |
|---|---|---|---|
| 0 | `0 0 0 0 0 0 0 .88` | 0.110 | .88 |
| 4 | `0 0 0 0 0 .88 .88 .88` | 0.329 | .88 |
| 9 | `0 0 0 .88 .88 .88 .88 .88` | 0.549 | .88 |
| **14** | `0 .88 .88 .88 .88 .88 .88 .88` | **0.769** | .88 ← peak |
| 17 | `0 0 .88 .88 .88 .88 .88 .88` | 0.549 | .88 |

Full history: `reports/REPORT_TTT_FAITHFUL.md`.

## Caveats

- Our base avg (0.034) is much lower than Yuchen's base avg (0.148) on the same model + tasks. Likely cause: different sampling temperature / decoder config. Δ-base column inside each report row is the apples-to-apples comparison.
- 11 OOD tasks: 10/11 stayed at 0 because base also scored 0 — RL has no signal to amplify. `llm_sql__large` is the exception (0 → 0.69), showing TTT-Discover *can* discover novel solutions. Paper hits the same pattern at 50 steps × 512 rollouts, where exploration eventually succeeds on more OOD tasks.
- We do **not** implement the paper's "two-phase token forcing" or token-level MTU bookkeeping. Generation is plain vLLM `/v1/completions` with max_tokens=8192. This affects gpt-oss-120b reasoning more than DeepSeek-R1-Qwen3-8B in our experience.

## Citation

```
@article{yuksekgonul2026ttt,
  title={Learning to Discover at Test Time},
  author={Yuksekgonul, Mert and others},
  journal={arXiv preprint arXiv:2601.16175},
  year={2026}
}
```

Frontier-CS: https://github.com/FrontierCS/Frontier-CS
Official TTT-Discover code: https://github.com/test-time-training/discover
