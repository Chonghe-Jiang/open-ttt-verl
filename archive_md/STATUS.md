# Sprint complete -- 2026-05-16 ~13:10 UTC

## End state

- **Base eval**: 144/144 solutions, 19 tasks x 8 rollouts. 4 cbl variants got non-zero rollouts (best 0.88 on cbl_multi__low_av_loose). All OOD tasks zero.
- **TTT-Discover** (context-only, no LoRA grad step): 17/17 tasks completed in 4:15:51 wall on 4 p5en nodes. **All TTT runs returned 0**, including the cbl variants where base had non-zero. Reasoning in report.md TL;DR.
- **Final report**: /fsx/xuanj/ttt-discover/report.md (Yuchen-style table, with narrative explaining why TTT-Discover did not lift rewards in this minimal port).
- **Blockers log**: /fsx/xuanj/ttt-discover/BLOCKERS.md (3 entries, all resolved or workaround documented).

## Key files

- /fsx/xuanj/ttt-discover/report.md         -- final narrative + table
- /fsx/xuanj/ttt-discover/results/base/eval/*.eval.json  -- per-rollout base reward
- /fsx/xuanj/ttt-discover/results/ttt/*/history.json     -- per-step TTT trajectories
- /fsx/xuanj/ttt-discover/scripts/          -- all generation/eval/aggregate code
- /fsx/xuanj/ttt-discover/logs/slurm/       -- all slurm + per-task logs

## Slurm jobs

| job   | description       | wall    | state     |
|-------|-------------------|---------|-----------|
| 32048 | base solution gen | 02:51   | COMPLETED |
| 32050 | base Docker eval  | 06:08   | COMPLETED |
| 32053 | TTT-Discover 4n   | 04:15:51 | COMPLETED |

## What this exercise tells us

1. The harness works end-to-end: vLLM -> Frontier-CS Docker eval -> JSON -> report.
2. DeepSeek-R1-0528-Qwen3-8B base on Frontier-CS reproduces the shape of Yuchen pre-STaR table -- mostly zero, occasional cbl wins.
3. **Context-only TTT-Discover (without the LoRA gradient step) does not work on Frontier-CS at this scale.** This is consistent with the paper framing -- the gradient step is the load-bearing piece. To validate TTT-Discover proper, we need to add the LoRA grad path (transformers + peft, hot-swap into vLLM).
4. To get a signal comparable to Yuchen STaR delta of +0.05 ID avg, we likely need either (a) full TTT-Discover with grad updates, or (b) larger model or tasks where base has reward to compound.

## Recommendation for next session

Implement the LoRA grad step as a small extension on top of ttt_discover_minimal.py. Test on the one cbl variant where base hits 0.88 -- if TTT can hold or improve that, the implementation is viable; then sweep.
