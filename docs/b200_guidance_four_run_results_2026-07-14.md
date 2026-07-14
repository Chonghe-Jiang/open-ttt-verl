# B200 Guidance-TTT four-run snapshot

Snapshot time: 2026-07-14 10:40 EDT.

## Cancellation status

The following jobs were cancelled intentionally after preserving their logs and libraries:

| Job | Experiment | Final Slurm state | Elapsed | Allocation |
| --- | --- | --- | ---: | ---: |
| 153449 | Qwen3-8B, 8x64, `summary_only` | `CANCELLED` | 14:24:53 | 5xB200 |
| 153386 | Qwen3-8B, 8x64, `code_delta` | `CANCELLED` | 16:42:04 | 5xB200 |
| 153448 | Qwen3-14B, 8x32, `summary_only` | `CANCELLED` | 16:28:34 | 5xB200 |
| 153379 | Qwen3-14B, 8x32, `code_delta` | `CANCELLED` | 16:51:05 | 5xB200 |

The dependent continuation jobs `153450` and `153387` were also cancelled before allocation. The queued four-GPU Qwen3-8B execution-model experiment `154411` was cancelled before allocation. None of these three jobs consumed GPU runtime.

## Experiment configuration

All four experiments used four B200s for guidance-model training and one B200 for the frozen `openai/gpt-oss-120b` execution server. They used FrontierCS problem 0, eight parent groups per training step, guidance sampling temperature 0.9, greedy execution, LoRA rank/alpha 32/32, learning rate `4e-5`, and KL coefficient 0.05.

| Setting | `code_delta` | `summary_only` |
| --- | ---: | ---: |
| Guidance actor context | Complete parent code + incremental summary + score | Complete algorithm summary + score |
| Execution context | Complete parent code + guidance | Complete parent code + guidance |
| Execution output | Complete updated C++ solution + incremental summary | Complete updated C++ solution + complete summary |
| PPO mini-batch | 8 | 4 |
| Maximum actor prompt | 8192 | 4096 |
| Truncation | `error` | `middle` |

This was therefore not a clean prompt-mode-only ablation: PPO mini-batch size and prompt handling also differed.

## Final result snapshot

`Completed step` means the last step that reached the training update and appeared in the trainer log. The libraries also contain partial entries from the next interrupted step. `Archive best` includes those partial entries.

| Job | Completed steps | Partial next step | Last completed mean | Peak completed mean | Archive best | Valid entries | Format/judge failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 153449, 8B `summary_only` | 10 | 213/512 at step 11 | 66.796 | 69.220 at step 8 | 81.422 | 5197/5333 (97.45%) | 136/5333 (2.55%) |
| 153386, 8B `code_delta` | 11 | 24/512 at step 12 | 68.547 | 68.547 at step 11 | 81.173 | 5219/5656 (92.27%) | 437/5656 (7.73%) |
| 153448, 14B `summary_only` | 23 | 93/256 at step 24 | 72.553 | 74.980 at step 17 | **81.847** | 5700/5981 (95.30%) | 281/5981 (4.70%) |
| 153379, 14B `code_delta` | 23 | 60/256 at step 24 | 69.123 | 69.123 at step 23 | 81.464 | 5471/5948 (91.98%) | 477/5948 (8.02%) |

The first step starts from the bootstrap candidate and contains many zero-score attempts, so aggregate means over the entire run understate post-bootstrap quality. Excluding step 1:

| Model/mode | Rollout mean | Rollout median | Mean guidance prompt | Groups improved over parent | Groups whose best child was worse |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8B `summary_only` | 61.158 | 72.720 | 1067 tokens | 69/76 (90.8%) | 3/76 (3.9%) |
| 8B `code_delta` | 57.670 | 69.267 | 4346 tokens | 68/81 (84.0%) | 7/81 (8.6%) |
| 14B `summary_only` | **69.114** | **77.816** | 1069 tokens | 103/180 (57.2%) | 9/180 (5.0%) |
| 14B `code_delta` | 64.200 | 75.490 | 3869 tokens | 108/179 (60.3%) | 26/179 (14.5%) |

The online rollout mean is not a fixed evaluation metric. PUCT intentionally chooses different parents, so the mean can fluctuate even when the global archive best improves.

## Training-health snapshot

No run showed CUDA OOM, NCCL failure, NaN, Ray task failure, FrontierCS environment failure, or rollout abortion before cancellation.

| Job | Entropy | Rollout/train KL | Grad norm | GPU allocated | CPU memory | Last step time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 153449 | 0.300 | 0.0124 | 0.0251 | 23.58 GiB | 726.50 GiB | 6783 s |
| 153386 | 0.272 | 0.0130 | 0.0341 | 39.61 GiB | 578.91 GiB | 8375 s |
| 153448 | 0.444 | 0.0161 | 0.0174 | 41.69 GiB | 540.24 GiB | 3788 s |
| 153379 | 0.383 | 0.0150 | 0.0167 | 41.69 GiB | 362.10 GiB | 4019 s |

The policy updates were numerically stable, but the implementation did not scale operationally. Every rollout rewrote a growing `library.json`, causing very high CPU memory and increasing step time. The stopped libraries are 525-673 MiB each. Job 153449 was already using about 726.5 GiB of its 768 GiB allocation.

## Main findings

1. `summary_only` produced the strongest archive result for both actor sizes. The 14B `summary_only` run achieved the overall best score, 81.847.
2. `code_delta` still found improving children, but had a substantially heavier bad tail. Its format/compile/judge failure rate was about 7.7-8.0%, versus 2.6-4.7% for `summary_only`.
3. Both modes gave the execution model complete parent code and required a complete updated program. The meaningful mode difference was the representation shown to the smaller guidance actor: full C++ plus an incremental summary versus a compact complete summary.
4. The `code_delta` guidance prompts were approximately 3.6-4.1 times longer. This likely made the guidance actor focus more on implementation details and increased destructive or overly ambitious changes.
5. The comparison is confounded by PPO mini-batch size: `summary_only` used 4 while `code_delta` used 8, with one PPO epoch. A future ablation must hold this constant.
6. Increasing group size is not equivalent to ordinary batch scaling. At the same 5120 execution calls, 8x16 performs 40 library/policy steps, 8x32 performs 20, and 8x64 performs only 10. Larger groups therefore trade evolutionary depth for more siblings from each selected parent.

## Difference from the Discover paper

These runs did not use the exact objective and PUCT rule in [Learning to Discover at Test Time](https://test-time-training.github.io/discover.pdf).

- Active advantage estimator: standard GRPO `(reward - group mean) / group std`.
- Paper objective: adaptive entropic leave-one-out advantage, with beta chosen so `KL(q_beta || uniform) = ln(2)`.
- The repository contains `compute_entropic_adaptive_beta`, but the launch path selected `algorithm.adv_estimator=grpo`, so it was inactive.
- Active PUCT Q value after expansion: `0.8 * parent_reward + 0.2 * best_child_reward`.
- Paper PUCT Q value after expansion: `best_child_reward` directly.
- Active policy loss: `ttt_reinforce_is` with token-level sampler/learner importance correction and a separate 0.05 low-variance KL loss.

A clean paper-aligned rerun should change the advantage estimator and PUCT Q rule deliberately, then validate each change with a short smoke before launching long jobs.

## Preserved artifacts

- 8B `summary_only` library: `outputs/guidance_ttt/polyomino_b200_5gpu_group64_prompt_refinement_50step/library.json`
- 8B `code_delta` library: `outputs/guidance_ttt/polyomino_b200_5gpu_group64_50step_from_smoke_153361/library.json`
- 14B `summary_only` library: `outputs/guidance_ttt/polyomino_b200_5gpu_qwen3_14b_group32_prompt_refinement_50step_153448/library.json`
- 14B `code_delta` library: `outputs/guidance_ttt/polyomino_b200_5gpu_qwen3_14b_group32_50step_153379/library.json`
- Slurm logs are preserved under `outputs/slurm/` using the corresponding job IDs.
- The final 8B `summary_only` `global_step_10` checkpoint (approximately 17 GiB on disk) was deliberately removed after this snapshot. No old checkpoint directories remain; all four libraries and logs are preserved.

No artifact was deleted during cancellation itself. The checkpoint cleanup happened later as a separate storage-maintenance action.
