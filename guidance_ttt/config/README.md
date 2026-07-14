# Guidance-TTT recipe index

The top-level directory contains active Polyomino Packing recipes. Historical
Modal, Erdos, smoke, and debugging recipes live in `backup/`. Recipe filenames
encode the platform, allocation, actor/execution models, batch/group shape,
prompt mode, and run length.

## Recommended B200 long runs

The current paper-aligned experiments all use four training B200s and one
GPT-OSS-120B execution B200:

| Guidance actor | Shape | Prompt mode | Recipe suffix |
| --- | --- | --- | --- |
| Qwen3-8B | 8x64 | `code_delta` | `group64_code_delta_8192_entropic_best_child_50step.yaml` |
| Qwen3-8B | 8x64 | `summary_only` | `group64_prompt_refinement_entropic_best_child_50step.yaml` |
| Qwen3-14B | 8x32 | `code_delta` | `group32_code_delta_8192_entropic_best_child_50step.yaml` |
| Qwen3-14B | 8x32 | `summary_only` | `group32_prompt_refinement_entropic_best_child_50step.yaml` |

Each recipe explicitly sets:

```yaml
run:
  adv_estimator: entropic_adaptive_beta
  save_freq: 10
ttt:
  puct_q_mode: best_child
```

`best_child` means an expanded archive node uses its best observed child reward
directly as Q. Recipes without these fields retain the backward-compatible GRPO
and `0.8 * parent + 0.2 * best_child` behavior.

## B200 acceptance and comparison recipes

- `polyomino_b200_3gpu_*_smoke.yaml`: two training GPUs plus one GPT-OSS-120B
  execution GPU, 8x16, one step.
- `polyomino_b200_5gpu_*_smoke.yaml`: four training GPUs plus one execution
  GPU, covering 8x32, 8x64, Qwen3-8B, and Qwen3-14B acceptance shapes.
- Matching non-`entropic_best_child` 50-step recipes preserve the earlier GRPO
  and blended-PUCT baseline.
- `polyomino_b200_4gpu_qwen3_8b_exec_shared_*`: Qwen3-8B execution with native
  thinking, colocated on training GPU 3. The `fallback` smoke lowers execution
  concurrency and GPU memory pressure.

## Modal recipes

The active Modal/H200 family remains at the top level for compatibility with
`scripts/modal_polyomino_h200_smoke.py`. The recommended code-delta pair is:

- `polyomino_modal_h200_3gpu_gpt_oss_120b_batch8_group16_code_delta_8192_smoke.yaml`
- `polyomino_modal_h200_3gpu_gpt_oss_120b_batch8_group16_code_delta_8192_50step.yaml`

Older recipes referenced by legacy launch modes are preserved under `backup/`.

## Shared execution settings

The GPT-OSS-120B recipes do not artificially shorten execution:

```yaml
llm:
  execution:
    max_tokens: null
    phase1_max_tokens: null
```

The code-delta recipes cap concurrent execution requests and FrontierCS judge
workers at 16. One agent-loop worker owns the global execution semaphore, so
the configured concurrency is not multiplied by additional worker processes.
If GPU count, group size, agent workers, or execution concurrency changes,
update the YAML recipe, Slurm launcher, and acceptance expectations together.

Seeded runs copy
`guidance_ttt/seeds/polyomino_packing/gpt_oss_120b_bootstrap_library.json` into
a fresh output directory. Runtime fields such as rollout group size and PUCT Q
mode are then persisted into the copied library and validated on resume.
