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
Git under `models/`, `.hf_cache/`, `.runtime/`, and `outputs/`.

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

Each five-GPU task currently requests 96 CPUs and 768 GiB of host memory because
large JSON libraries and concurrent verifier work approached the previous
memory limit. This resource shape can wait longer than GPU count alone would
suggest; reduce it only after the library persistence path is more efficient.

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
