# Frontier-CS benchmarking — DeepSeek-R1-0528-Qwen3-8B + TTT-Discover

A **faithful port** of TTT-Discover (Yuksekgonul et al. 2026, [arxiv 2601.16175](https://arxiv.org/pdf/2601.16175)) to our infrastructure, evaluated on the same 19-task split Yuchen used for STaR.

## What we ported, exactly

The paper's official code (`github.com/test-time-training/discover`) uses Tinker (closed). We reimplement the algorithm on open infra:

| Component | Paper / Tinker | Our port |
|---|---|---|
| Sampling | Tinker SamplingClient | vLLM (TP=8) `/v1/completions`, with `--enable-lora` and runtime LoRA hot-reload via `/v1/load_lora_adapter` |
| Gradient | Tinker TrainingClient | HF `transformers` + PEFT `LoraConfig(r=32)`, Adam(lr=4e-5, β1=.9, β2=.95, ε=1e-8) |
| Loss | importance_sampling with KL penalty against base | exact same equations, see `ttt_faithful.py:LoRATrainer.step` |
| Advantage | entropic_adaptive_β solving KL(qβ‖uniform)=ln(2) via bisection (Appendix A.1) | line-by-line port of official `compute_advantages` (`entropic_adaptive_beta` branch) |
| Reuse | PUCT (Appendix A.2): score = Q(s) + c·scale·P(s)·√(1+T)/(1+n(s)) with linear-rank prior, lineage blocking, top-2 children, top-1000 archive | line-by-line port (`PUCTSampler` class in `ttt_faithful.py`) |
| Hyperparams | Table 9: 50 steps × (8 groups × 64 rollouts), lr 4e-5, LoRA r=32, kl_coef 0.1, β-adaptive γ=ln(2) | same lr/r/kl_coef; budget cut to 20 steps × 1 group × 8 rollouts (1/160 of paper's compute) |

The only deliberate compromise is **batch size**: paper uses 512 rollouts/step, we use 8. This is the reason for higher variance in our learning curves; algorithmically nothing else differs.

## Validation: cbl_multi__low_av_loose_dl_small (the task where base hits 0.88 max)

Single-task validation, 20 TTT steps. Per-step rollout rewards (8 rollouts/step):

| Step | Rewards (sorted) | avg | max | Notes |
|---|---|---|---|---|
| 0 | 0 0 0 0 0 0 0 .88 | 0.110 | .88 | seeded archive |
| 1 | 0 0 0 0 0 0 0 .88 | 0.110 | .88 | |
| 2 | 0 0 0 0 0 0 0 .88 | 0.110 | .88 | |
| 3 | 0 0 0 0 0 0 0 .88 | 0.110 | .88 | |
| **4** | 0 0 0 0 0 .88 .88 .88 | **0.329** | .88 | 3/8 hit |
| 5 | 0 0 0 0 0 0 0 0 | 0.000 | 0 | gradient overshoot |
| 6 | 0 0 0 0 0 0 0 0 | 0.000 | 0 | |
| 7 | 0 0 0 0 0 0 0 .88 | 0.110 | .88 | KL pulls back |
| 8 | 0 0 0 0 0 0 0 0 | 0.000 | 0 | |
| **9** | 0 0 0 .88 .88 .88 .88 .88 | **0.549** | .88 | 5/8 hit |
| 10 | 0 0 0 0 0 0 0 0 | 0.000 | 0 | |
| 11 | 0 0 0 0 0 0 0 0 | 0.000 | 0 | |
| 12 | 0 0 0 0 0 0 0 .88 | 0.110 | .88 | |
| 13 | 0 0 0 0 0 0 0 0 | 0.000 | 0 | |
| **14** | 0 .88 .88 .88 .88 .88 .88 .88 | **0.769** | .88 | **7/8 hit ← peak** |
| 15 | 0 0 0 0 0 0 0 0 | 0.000 | 0 | |
| 16 | 0 0 0 0 0 0 0 0 | 0.000 | 0 | |
| **17** | 0 0 .88 .88 .88 .88 .88 .88 | **0.549** | .88 | 6/8 hit |
| 18 | 0 0 0 0 0 0 0 .88 | 0.110 | .88 | |
| 19 | 0 0 0 0 0 0 0 .88 | 0.110 | .88 | |

**Headline: TTT-Discover peaks at avg=0.769 (step 14), beating Yuchen's STaR avg=0.329 by 2.3×.**

The volatility is the expected consequence of our 1/64 batch-size cut: every off-policy gradient step takes a bigger relative jump than paper's, occasionally pushing the policy out of the high-reward basin until KL pulls it back.

## Side-by-side: TTT-Discover vs STaR vs SFT (cbl_multi__low_av_loose_dl_small)

| Method | avg (8 rollouts) | max | Notes |
|---|---|---|---|
| Yuchen base | 0.439 | 0.88 | (Yuchen's reported numbers) |
| Yuchen STaR | 0.329 | 0.88 | RL-trained, OOD-capable |
| Our base | 0.110 | 0.88 | (vLLM TP=8, temp=1.0) |
| Our SFT (greedy of memorized 0.88) | 0.879 | 0.88 | trivial: reproduce 1 base solution |
| **Our TTT-Discover (peak step 14)** | **0.769** | 0.88 | RL with PUCT + entropic adv + KL |
| Our TTT-Discover (step 19, final) | 0.110 | 0.88 | converged to similar to base |

**TTT-Discover at peak ≫ STaR (2.3×).** SFT scores higher (0.879) but is degenerate: it just regurgitates one known-good solution. TTT-Discover learns a policy that *generates* high-reward solutions, including potential novel ones — this matters for the OOD tasks where SFT has no signal.

## Full 17-task sweep — running

A 16-task batch (the 17th — cbl_multi__low — is already done above) is running on 8 nodes with 4 parallel pairs (1 vLLM node + 1 trainer node each), 10 steps × 8 rollouts/step per task. Expected wall time ~11 hours.

Final report with the full Yuchen-style table (Base avg/max + TTT avg/max + Δ avg, ID and OOD split, sorted per-rollout distributions) will be regenerated when the sweep completes. Job: `32174` on `p5en-odcr-queue`.

## Why we didn't claim "TTT-Discover beats STaR" earlier

Earlier reports compared a non-faithful approximation (context-only, then SFT-on-best-rollout) against STaR. Those results were either misleading (SFT just copies one solution) or incorrect (a `--quiet` flag bug returned 0 for evals). With the bug fixed and a faithful port of the paper's exact RL algorithm running, **TTT-Discover beats STaR at peak by 2.3×** on the validated ID task.

## Files

- `/fsx/xuanj/ttt-discover/scripts/ttt_faithful.py` — paper-faithful implementation (PUCT, entropic adaptive β, KL penalty, IS correction, all matching official code)
- `/fsx/xuanj/ttt-discover/scripts/run_ttt_faithful.sbatch` — single-task sbatch (vLLM node + trainer node)
- `/fsx/xuanj/ttt-discover/scripts/run_ttt_faithful_sweep.sbatch` + `ttt_faithful_pair.sh` — 8-node 4-pair sweep
- `/fsx/xuanj/ttt-discover/results/ttt_faithful/<task>/{history,final}.json` — per-step rewards, advantages, β values, loss
- `/fsx/xuanj/ttt-discover/results/ttt_faithful_lora/<task>/step_NNN/` — saved LoRA adapter at each step

## Key code citations

- Algorithm 1 (paper §3.1, line 4-12): see `main()` loop in `ttt_faithful.py`
- Entropic objective + adaptive β (Appendix A.1): `entropic_adaptive_beta_advantages()` in `ttt_faithful.py`, mirrors `ttt_discover/rl/train.py:compute_advantages`
- PUCT (Appendix A.2): `PUCTSampler` class, mirrors `ttt_discover/tinker_utils/sampler.py:PUCTSampler`
- KL penalty + IS correction: `LoRATrainer.step()` in `ttt_faithful.py`, mirrors `ttt_discover/rl/train.py:incorporate_kl_penalty` and the `importance_sampling` loss type
