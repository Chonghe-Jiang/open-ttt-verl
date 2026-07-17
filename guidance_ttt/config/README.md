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

## Discover-compatible search and RL core

New experiments can opt into the official Discover search/training semantics
without changing the Guidance-to-Execution communication architecture:

```yaml
run:
  adv_estimator: entropic_adaptive_beta
  learning_rate: 4.0e-5
  temperature: 1.0
  use_kl_loss: false
ttt:
  discover_compat: true
  groups_per_batch: 8
  group_size: 16
  puct_c: 1.0
  puct_q_mode: best_child
  max_buffer_size: 1000
  topk_children: 2
```

The compatibility path keeps batch/group shape configurable, but fixes the
remaining mechanics: one PUCT visit per rollout (including failures), failed
rollouts excluded from the candidate archive, batch-end top-2 filtering,
independent seed-root lineages, direction-normalized raw score for PUCT,
adaptive-entropic LOO advantages, constant-reward-group removal, centered
base-policy KL in the advantage with coefficient `0.1`, untruncated token-level
importance sampling, one optimizer epoch, and the official Adam
hyperparameters.

Compatibility is persisted in `library.json` and validated on resume. Do not
enable it in an existing output directory produced by the legacy per-group
PUCT accounting path; start from a fresh output directory instead.

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

The dense actor-scale matrix contains four matched recipes:

- `polyomino_b200_5gpu_qwen36_27b_gpt_oss_120b_batch8_group8_summary_only_entropic_best_child_500step.yaml`
- `polyomino_b200_5gpu_qwen36_27b_gpt_oss_120b_batch8_group8_code_delta_prompt8192_entropic_best_child_500step.yaml`
- `polyomino_b200_5gpu_qwen36_27b_gpt_oss_120b_batch8_group8_summary_only_entropic_blended_500step.yaml`
- `polyomino_b200_5gpu_qwen36_27b_gpt_oss_120b_batch8_group8_code_delta_prompt8192_entropic_blended_500step.yaml`

All four assign four B200s to Qwen3.6-27B FSDP training/rollout and keep
GPT-OSS-120B isolated on the fifth B200. They use LoRA rank 32, guidance
temperature 0.9, `entropic_adaptive_beta`, and `ttt_reinforce_is`. The launchers
default to the validated 8x16 shape by overriding the recipes' conservative 8x8
fallback value.

The two direct recipes use the best observed child reward as an expanded node's
PUCT Q. The two blended ablations use `0.8 * own_reward + 0.2 *
best_child_reward`. Prompt mode and Q mode are therefore independently varied.

`summary_only` uses a 4096-token prompt and 8192-token response budget.
`code_delta` uses 8192 + 8192 and a 16384-token rollout context so the selected
parent code remains available to guidance.

Qwen3.6 is a multimodal Gated DeltaNet checkpoint. The recipe therefore
excludes and freezes the visual tower and asks vLLM to load only the language
model. Actor and reference forwards use non-packed batches, while standard
attention layers use FlashAttention 2. Use the smoke-gated, resumable launchers
for compatibility testing and continued runs. For a bounded comparison that
must not write checkpoints, use the four-way 23-hour launcher documented in
[`scripts/README.md`](../../scripts/README.md).

## One-GPU Qwen3-8B with Evolvent GPT-5.4 execution

These recipes keep the trainable Qwen3-8B actor on one B200 and call GPT-5.4
through an OpenAI-compatible Evolvent endpoint, so no local execution GPU is
needed:

- `polyomino_b200_1gpu_qwen3_8b_evolvent_gpt54_batch8_group16_summary_only_entropic_best_child_500step.yaml`
- `polyomino_b200_1gpu_qwen3_8b_evolvent_gpt54_batch8_group16_summary_only_entropic_best_child_50step_oss_seed_no_ckpt.yaml`

Both use 8 groups x 16 samples, `summary_only`, `entropic_adaptive_beta`, and
direct best-child PUCT. The first is the smoke recipe and can bootstrap a fresh
GPT-5.4 seed. The second is the bounded formal recipe: it starts from the
validated GPT-OSS seed, runs at most 50 steps, and sets `save_freq: -1`.

The API prompt asks GPT-5.4 to think carefully, then return exactly `<solution>`
and `<summary>`; it does not require a separate execution-thinking block. The
API key is read from ignored file `.secrets/evolvent_api_key`. Concurrency is
selected by `scripts/probe_evolvent_gpt54_concurrency.py` and stored in ignored
runtime file `.runtime/evolvent_gpt54_concurrency`.

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

For GPT-OSS high-reasoning runs, setting a non-null
`phase1_max_tokens` enables Discover's token-exact two-phase completion over
the vLLM completions endpoint. The value is a total prompt-plus-phase-1 budget,
not a standalone output limit. If phase 1 exhausts this budget, the client
inserts the Harmony final-channel prefill and uses the remaining context for
the answer:

```yaml
llm:
  execution:
    reasoning_effort: high
    max_tokens: null
    phase1_max_tokens: 22000
    context_window: 32768
    context_buffer: 50
```

The two-phase path requires a vLLM server that supports token-ID prompts and
`return_token_ids` on `/v1/completions` (vLLM 0.17 in the B200 image). Leaving
`phase1_max_tokens: null` retains the single-request chat-completions path.
Use `prompt_style: qwen_native_thinking` with GPT-OSS two-phase high-reasoning
runs. Phase 1 already occupies the native reasoning channel, so the final
answer should contain only `<solution>` and `<summary>` blocks. The legacy
explicit `<execution_thinking>` final-answer contract can make phase 2 stop
after a short natural-language explanation, producing verifier parse errors
because no C++17 code block was emitted.

The active two-GPU high-reasoning recipe is:

- `polyomino_b200_2gpu_qwen3_8b_gpt_oss_120b_high_reasoning_batch8_group16_summary_only_discover_two_phase_1day_no_ckpt.yaml`

It pairs Qwen3-8B guidance on GPU 0 with GPT-OSS-120B execution on GPU 1,
uses 8 groups x 16 rollouts, disables checkpoint writing, and is launched by
`scripts/slurm_polyomino_b200_2gpu_qwen3_8b_gpt_oss_120b_high_reasoning_discover_two_phase_1day.sbatch`.

The code-delta recipes cap concurrent execution requests and FrontierCS judge
workers at 16. One agent-loop worker owns the global execution semaphore, so
the configured concurrency is not multiplied by additional worker processes.
If GPU count, group size, agent workers, or execution concurrency changes,
update the YAML recipe, Slurm launcher, and acceptance expectations together.

Seeded runs copy
`guidance_ttt/seeds/polyomino_packing/gpt_oss_120b_bootstrap_library.json` into
a fresh output directory. Runtime fields such as rollout group size and PUCT Q
mode are then persisted into the copied library and validated on resume.
