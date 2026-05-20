# Blockers / things to discuss when xuanj wakes

(Append-only. Newest at top. Each entry: timestamp, what, why blocking, my workaround.)

## 2026-05-16 06:55 UTC -- pip dependency conflicts (non-blocking)
- vllm 0.12.0 pins anthropic==0.71.0 strictly; frontier-cs requires anthropic>=0.74.0
- trl 0.9.6 requires numpy<2.0; vllm 0.12 wants numpy>=2; frontier-cs wants numpy==2.3.4
- numba 0.61.2 wants numpy<2.3
**workaround:** Settled on numpy 2.3.4 + anthropic 0.102.0. Frontier-cs CLI works. vllm 0.12 has a hard pin (anthropic==0.71.0) but actually still imports/runs fine -- pip resolver complaint only. trl is unused for now (we will use raw transformers + peft for LoRA training).
**revisit if:** vllm import errors on GPU node, or trl path is reactivated for SFT.

## 2026-05-16 06:55 UTC -- model name (resolved)
User said "deepseek-r1-qwen3-8b". The HF id is **deepseek-ai/DeepSeek-R1-0528-Qwen3-8B** (not DeepSeek-R1-Distill-Qwen-8B; that 8B variant does not exist on HF -- only 1.5B/7B/14B/32B). Updated all scripts.

## 2026-05-16 ~07:50 UTC -- TTT step time (resolved by tightening config)
First TTT step took ~870s (730s sample + 140s eval) for one task on 1 node TP=8. With 17 tasks, 10 steps, 4 shards => ~12 hrs which overruns the 12hr slurm time limit.

**Workaround:** lowered max-tokens 8192 -> 4096 (reasoning model still fits), num-steps 10 -> 6, eval-concurrency 4 -> 8 (one Docker eval per H200 GPU on the same node = OK). Re-running. Note: existing job 32052 keeps running at old config; we will let it finish what it can but the resubmit captures more thorough coverage.
