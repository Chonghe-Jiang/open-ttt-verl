# Guidance-TTT recipe index

The top-level directory contains active Polyomino Packing recipes. Historical
Modal, Erdos, smoke, and debugging recipes live in `backup/`. Recipe filenames
encode the platform, allocation, actor/execution models, batch/group shape,
prompt mode, and run length.

## Established B200 long runs

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

## Two-GPU Qwen3-8B execution recipes

The `polyomino_b200_2gpu_qwen3_8b_*_batch8_group8_*_500step.yaml` recipes are
the current lightweight, resumable B200 family. Each uses one training GPU, one
execution GPU, 8 groups x 8 samples, `save_freq: 1`,
`entropic_adaptive_beta`, `puct_q_mode: best_child`, auto-resume, and a
single retained checkpoint.

| Execution family | Recipe filename fragment | Prompt style |
| --- | --- | --- |
| GPT-OSS-120B | `qwen3_8b_gpt_oss_120b` | explicit execution thinking |
| Qwen3.6-35B-A3B | `qwen3_8b_qwen36_35b_exec` | `qwen_no_thinking` |
| Qwen3-Coder-Next-FP8 | `qwen3_8b_qwen3_coder_next_fp8_exec` | `qwen_no_thinking` |

Each family has both `code_delta` and `summary_only` files. The no-thinking
styles set `chat_template_kwargs.enable_thinking: false` and require exactly
`<solution>` plus `<summary>` from execution. They do not impose an execution
token cap (`max_tokens: null`). Use the paired submitters in
[`scripts/README.md`](../../scripts/README.md) rather than submitting these
YAML files directly, so the smoke and continuation dependencies are preserved.

The two modes use different actor token budgets. `summary_only` keeps the
original 4096-token prompt and 8192-token response limits. `code_delta` carries
the selected candidate's complete source code, so it reserves 12288 tokens for
the prompt and 4096 for the guidance response, preserving the same 16384-token
total context. Dynamic agent-loop prompts are bounded before rollout generation
with `truncation: middle`; this keeps the prompt seen during sampling identical
to the prompt used for training and prevents mixed-width batch assembly when a
candidate grows beyond the configured budget.

All active training recipes set
`actor_rollout_ref.rollout.free_cache_engine=True`. The rollout engine releases
its KV cache before actor log-probability and update phases, avoiding overlap
between a large vLLM cache allocation and the training model's temporary
logits/entropy tensors. Historical files under `backup/` are left unchanged.

## Qwen3.6-27B dense guidance scaling

The current dense actor-scale recipes are:

- `polyomino_b200_5gpu_qwen36_27b_gpt_oss_120b_batch8_group8_summary_only_entropic_best_child_500step.yaml`
- `polyomino_b200_5gpu_qwen36_27b_gpt_oss_120b_batch8_group8_code_delta_prompt8192_entropic_best_child_500step.yaml`
- `polyomino_b200_5gpu_qwen36_27b_gpt_oss_120b_batch8_group8_summary_only_entropic_blended_500step.yaml`

Both assign four B200s to Qwen3.6-27B FSDP training/rollout and keep
GPT-OSS-120B isolated on the fifth B200. They use LoRA rank 32, guidance
temperature 0.9, `entropic_adaptive_beta`, direct best-child PUCT,
`ttt_reinforce_is`, per-step saves, auto-resume, and one retained checkpoint.
The launchers default to the validated 8x16 shape by overriding the recipes'
conservative 8x8 fallback value.

The blended `summary_only` recipe is a controlled search ablation: after a
node has been expanded, its PUCT Q value is `0.8 * own_reward + 0.2 *
best_child_reward`. All non-search settings match the direct-best-child
`summary_only` recipe, and its dedicated launcher fixes the shape at 8x16.

`summary_only` uses a 4096-token prompt and 8192-token response budget.
`code_delta` uses 8192 + 8192 and a 16384-token rollout context so the selected
parent code remains available to guidance.

Qwen3.6 is a multimodal Gated DeltaNet checkpoint. The recipe therefore
excludes and freezes the visual tower and asks vLLM to load only the language
model. Use the two smoke-gated submitters documented in
[`scripts/README.md`](../../scripts/README.md); do not bypass their one-step
compatibility and memory tests.

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
