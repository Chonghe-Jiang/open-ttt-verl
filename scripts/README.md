# B200 launcher index

The cluster workflow separates reusable runners from Slurm allocation files:

- `setup_polyomino_b200.sh` and `slurm_setup_polyomino_b200.sbatch` prepare the
  local FrontierCS runtime and model cache.
- `run_polyomino_b200_3gpu_smoke.sh` runs the basic GPT-OSS acceptance path.
- `run_polyomino_b200_long.sh` runs five-GPU GPT-OSS experiments.
- `run_polyomino_b200_4gpu_qwen_exec_shared.sh` runs the experimental Qwen3-8B
  execution server colocated with training GPU 3.
- `validate_polyomino_b200_smoke.py` validates group accounting, prompt mode,
  PUCT accounting, and real verifier outcomes.
- `validate_qwen_shared_smoke.py` adds Qwen native-reasoning checks.

All generated models, caches, logs, checkpoints, and libraries are ignored by
Git under `models/`, `.hf_cache/`, `.runtime/`, and `outputs/`. API credentials
belong under ignored `.secrets/`; never put them in YAML or submit scripts.

## One-GPU Qwen3-8B with Evolvent GPT-5.4

This path trains Qwen3-8B on one B200 and uses the external GPT-5.4 endpoint for
execution. Create the key file with owner-only permissions, then probe a safe
request concurrency before submitting:

```bash
install -d -m 700 .secrets
${EDITOR:-vi} .secrets/evolvent_api_key
chmod 600 .secrets/evolvent_api_key

SIF_PATH=/work/mit/ppliang_mit/chonghej/open-ttt-verl/containers/open-ttt-verl-ttt-vllm.sif
apptainer exec "${SIF_PATH}" \
  python scripts/probe_evolvent_gpt54_concurrency.py \
  --config guidance_ttt/config/polyomino_b200_1gpu_qwen3_8b_evolvent_gpt54_batch8_group16_summary_only_entropic_best_child_500step.yaml \
  --output .runtime/evolvent_gpt54_concurrency
```

The probe writes the selected value to `.runtime/evolvent_gpt54_concurrency`.
The one-step smoke uses a fresh GPT-5.4 bootstrap and validates API errors,
sample accounting, verifier outcomes, and rollout/actor probability parity:

```bash
sbatch scripts/slurm_polyomino_b200_1gpu_qwen3_8b_evolvent_gpt54_group16_smoke.sbatch
```

After the smoke passes, the bounded formal run is:

```bash
sbatch scripts/slurm_polyomino_b200_1gpu_qwen3_8b_evolvent_gpt54_group16_50step_oss_seed_no_ckpt.sbatch
```

It uses the validated GPT-OSS bootstrap seed, runs 8x16 for at most 50 steps,
and writes no actor checkpoint. Stable prompt prefixes are preserved so the API
gateway can reuse cached input tokens when supported.

## Two-GPU Qwen3-8B execution matrix

These launchers use one B200 for the trainable Qwen3-8B actor and one B200 for
the frozen execution model. They submit both communication modes through an
`afterok` chain: model/runtime preparation (if required), one-step smoke,
23-hour day 1, then a 23-hour resumable day 2. The formal stages retain the
latest checkpoint only.

| Execution model | Execution final format | Submit command |
| --- | --- | --- |
| GPT-OSS-120B | explicit thinking, solution, summary | `scripts/submit_qwen3_8b_b200_2gpu_group8_two_day.sh <tag>` |
| Qwen3.6-35B-A3B | solution + summary, no thinking | `scripts/submit_qwen36_35b_exec_b200_2gpu_group8_two_day.sh <tag>` |
| Qwen3-Coder-Next-FP8 | solution + summary, no thinking | `scripts/submit_qwen3_coder_next_fp8_exec_b200_2gpu_group8_two_day.sh <tag>` |

The Qwen3.6 and Coder-Next launchers use the isolated
`.runtime/qwen36-exec-site-packages` vLLM 0.19 environment only for execution;
the guidance actor remains on the Apptainer image's verl-compatible stack. The
Coder-Next setup additionally verifies the `Qwen3NextForCausalLM` FP8 model
configuration before enabling its smoke jobs.

## Five-GPU Qwen3.6-27B guidance scaling

Use the dedicated submitters for the dense 27B experiments:

```bash
scripts/submit_qwen36_27b_guidance_b200_5gpu_group8_two_day.sh <tag>
scripts/submit_qwen36_27b_guidance_code_delta_prompt8192_b200_5gpu_group8_two_day.sh <tag>
scripts/submit_qwen36_27b_guidance_blended_b200_5gpu_group16_two_day.sh <tag>
scripts/submit_qwen36_27b_guidance_code_delta_blended_b200_5gpu_group16_two_day.sh <tag>
```

It assigns GPUs 0-3 to Qwen3.6-27B guidance training/rollout and GPU 4 to the
GPT-OSS-120B execution server. Each chain is `setup -> 1-step smoke -> 23h day1
-> 23h day2`; every dependency uses `afterok`. The default smoke and formal
shape is the validated 8x16 setting. Use `GROUP_SIZE=8 <command>` for the
fallback shape.

The formal stages share a fresh output directory, resume automatically, save
every step, and retain only the latest checkpoint. Actor and reference-policy
forwards use non-packed batches to isolate Qwen3.6 Gated DeltaNet state, while
the standard attention layers use FlashAttention 2. The smoke gate checks
rollout/actor probability parity before either formal stage can run. The setup
uses an isolated vLLM 0.19 runtime with the Qwen3.6 packed-LoRA and Gated
DeltaNet Triton allocator fixes for the colocated rollout path.

The two blended submitters are fixed 8x16 `summary_only` and `code_delta`
ablations. They change only PUCT Q to
`0.8 * own_reward + 0.2 * best_child_reward`; the actor, execution model,
objective, mode-specific prompts, sampling temperature, checkpoint policy,
and smoke-gated two-day continuation match their direct-best-child runs.

For one bounded 23-hour comparison with no checkpoints or continuation chain,
submit the complete 2x2 matrix at once:

```bash
scripts/submit_qwen36_27b_g16_four_no_ckpt.sh <tag>
```

This submits `summary_only`/`code_delta` crossed with `best_child`/`blended`.
Each task requests five B200s, overrides the shape to 8x16, allows at most 500
steps, and verifies that no checkpoint directory was created. The setup marker
and rollout/actor parity checks remain mandatory.

## Paper-aligned five-GPU experiments

The generic launcher requires explicit environment values. Use a fresh output
directory for every independent experiment.

For a direct 14B run:

```bash
CONFIG=guidance_ttt/config/polyomino_b200_5gpu_qwen3_14b_gpt_oss_120b_batch8_group32_code_delta_8192_entropic_best_child_50step.yaml
OUTPUT_DIR=outputs/guidance_ttt/entropic_best_child_14b_g32_code_delta_$(date +%Y%m%d_%H%M%S)

sbatch \
  --export=ALL,CONFIG="$CONFIG",OUTPUT_DIR="$OUTPUT_DIR",EXPECTED_GROUP_SIZE=32,EXPECTED_PROMPT_MODE=code_delta,STAGE=direct \
  scripts/slurm_polyomino_b200_5gpu_entropic_best_child_50step.sbatch
```

The 8B 8x64 runs are staged at step 20 for the 24-hour partition wall time:

```bash
CONFIG=guidance_ttt/config/polyomino_b200_5gpu_gpt_oss_120b_batch8_group64_code_delta_8192_entropic_best_child_50step.yaml
OUTPUT_DIR=outputs/guidance_ttt/entropic_best_child_8b_g64_code_delta_$(date +%Y%m%d_%H%M%S)
SCRIPT=scripts/slurm_polyomino_b200_5gpu_entropic_best_child_50step.sbatch

FIRST=$(sbatch --parsable \
  --export=ALL,CONFIG="$CONFIG",OUTPUT_DIR="$OUTPUT_DIR",EXPECTED_GROUP_SIZE=64,EXPECTED_PROMPT_MODE=code_delta,STAGE=first \
  "$SCRIPT")
sbatch --dependency="afterok:$FIRST" \
  --export=ALL,CONFIG="$CONFIG",OUTPUT_DIR="$OUTPUT_DIR",EXPECTED_GROUP_SIZE=64,EXPECTED_PROMPT_MODE=code_delta,STAGE=final \
  "$SCRIPT"
```

Substitute the matching `prompt_refinement` recipe and
`EXPECTED_PROMPT_MODE=summary_only` for the summary-only condition.

Each five-GPU task currently requests 96 CPUs and 768 GiB of host memory. The
agent loop now shares one locked in-process archive across concurrent
trajectories, removing the previous per-trajectory `library.json` duplication.
The allocation remains conservative for model workers, FrontierCS, and Ray; it
can wait longer than GPU count alone would suggest.

## Acceptance jobs

```bash
sbatch scripts/slurm_polyomino_b200_3gpu_smoke.sbatch
sbatch scripts/slurm_polyomino_b200_5gpu_group64_smoke.sbatch
sbatch scripts/slurm_polyomino_b200_5gpu_qwen3_14b_group32_smoke.sbatch
sbatch scripts/slurm_polyomino_b200_4gpu_qwen_exec_shared_group32_smoke.sbatch
```

The shared-Qwen smoke accepts `COLOCATION_PROFILE=primary` or `fallback`. It
starts Qwen3-8B with vLLM's `qwen3` reasoning parser; the prompt requests native
thinking and only `<solution>` plus `<summary>` in the final answer.
