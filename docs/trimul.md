# TriMul Guidance-TTT task

## Contract

`trimul` optimizes the outgoing AlphaFold3 Triangle Multiplicative Update from
TTT-Discover GPU Mode. A candidate is one complete Python `submission.py` that
defines `custom_kernel(data)`, includes at least one `@triton.jit` kernel,
supports every published shape/mask/distribution case, and returns float32.

The task-facing material is pinned in the repository:

| Artifact | Role |
| --- | --- |
| `guidance_ttt/tasks/trimul_prompt.py` | Original TTT-Discover task prompt |
| `guidance_ttt/seeds/trimul/glm52_scratch_bootstrap_library.json` | Default fixed GLM-5.2 scratch seed |
| `guidance_ttt/tasks/assets/trimul/baseline_solution.py` | Archived published TTT-Discover result; not used by default |
| `guidance_ttt/tasks/assets/trimul_evaluator/` | Original `task.yml`, reference, utilities, and evaluator |

The five evaluator files are vendored verbatim under the source repository's
MIT license. Tests pin their SHA-256 digests so a later edit cannot silently
change the benchmark.

## Scratch seed provenance

The main experiments do not bootstrap from the published TTT-Discover result.
GLM-5.2 generated the tracked root once from the public task statement under an
explicit `scratch` contract. Its prompt contained no parent code, prior
solution, selected summary, guidance, verifier score, search history, or
Discover optimization result. The accepted implementation follows the public
PyTorch equations and uses one functionally required flat Triton kernel for the
output-gate multiplication.

The official H100 evaluator reported:

- 18 of 18 correctness cases passed, each with zero reported maximum error;
- seven of seven leaderboard cases passed;
- geometric-mean runtime `10177.396849081848 us`;
- reward `0.14738542893071152`, computed as `1500 / runtime_us`;
- solution SHA-256
  `49485cfe6f1e0f5d2ea40df2bead246b59ab8774cff3a79ab03cba4cf08995a9`.

The tracked JSON preserves the full accepted prompt, response, source code, raw
summary, API metadata, and verifier artifacts. Both normal smoke configs copy
this pristine library before the run, so they make no bootstrap LLM request.
The separate `generate_scratch_seed` action reproduces the provenance workflow
into the Modal run volume but never overwrites the tracked seed automatically.

## Guidance-TTT data flow

The task supports matched `code_delta` and `summary_only` modes:

1. PUCT selects a library node.
2. In `code_delta`, Qwen3-8B receives the original task prompt plus the selected
   full `submission.py`, its raw delta summary, and verified runtime/reward. In
   `summary_only`, it receives a self-contained raw candidate summary and the
   same score fields, but no parent source code.
3. The guidance response proposes one executable H100/Triton optimization.
4. Evolvent GLM-5.2 receives the same task prompt, selected full parent code,
   and guidance in both modes, then returns a complete replacement
   `submission.py`. It returns a raw delta summary for `code_delta` or a
   self-contained full-candidate summary for `summary_only`.
5. The Modal H100 endpoint runs the official 18 correctness tests. Only a
   passing candidate proceeds to the official seven-case leaderboard run.
6. Code, the mode-specific raw summary, metric, reward, verifier artifacts, and
   parent edge are written to `library.json`; only guidance tokens receive the
   RL update.

The framework transports parent state in `<parent_code>` instead of duplicating
TTT-Discover's dynamic `State.to_prompt` code block. The public task statement,
rules, evaluator implementation, cases, numerical tolerances, and timing loop
are unchanged.

The LLM-response parser is intentionally framework-specific. TTT-Discover
extracts a bare fenced Python block, whereas Guidance-TTT asks execution for
`<solution>` and `<summary>` blocks and extracts `submission.py` from the
solution block. After extraction, the same Triton/identity prechecks are
applied, with one additional explicit `custom_kernel(data)` interface check,
and the extracted file is passed unchanged to the official evaluator.

## Metric and reward

For each of the seven ranked cases, the official evaluator reports mean kernel
latency in nanoseconds. The raw score is:

```text
raw_score_us = geometric_mean(case_mean_ns / 1000)
```

Lower is better. The maximize-oriented RL reward exactly matches
TTT-Discover:

```text
reward = 1500 / raw_score_us
```

A format failure, missing Triton kernel, correctness failure, runtime failure,
or evaluator failure receives no candidate score and zero reward. Evaluator
infrastructure failures are labeled `environment_error` rather than accepted
as ordinary benchmark failures.

## Modal environment

The official evaluator endpoint uses:

- one strict `H100!` allocation per request;
- CUDA `12.8.0` devel on Ubuntu 24.04;
- Python 3.13;
- PyTorch `>=2.7,<2.8` from the CUDA 12.8 index;
- Triton `3.3.1`;
- no network access while candidate code runs.

The original shared GPU-mode image also installed JAX, TinyGrad, cuPyNumeric,
CUTLASS DSL, and CUDA Python for unrelated task backends. TriMul and its
official evaluator do not import those packages, so this task-specific image
omits them. This avoids depending on mutable auxiliary wheels while preserving
the CUDA, compiler, Python, PyTorch, Triton, reference, tolerances, cases, and
timing environment that can affect TriMul correctness or score.

Each endpoint container accepts one request at a time so correctness and timing
measurements never share a GPU. The combined H200 smoke launcher defaults to
four evaluator containers. The task-specific `scripts/modal_trimul_judge.py`
defaults to 16 for the 8x16 B200 recipe. `TRIMUL_JUDGE_MAX_CONTAINERS` can
override either autoscaling cap at deployment time; individual recipes must
keep `task.verifier.concurrency` at or below the deployed cap.
Transient HTTP 408/429/5xx and network failures are retried at the transport
layer; evaluator-produced correctness failures are never retried or converted
into valid results.

## Credentials

Create a Modal secret without putting credentials in YAML or source code:

```bash
modal secret create guidance-ttt-trimul-secrets \
  EVOLVENT_API_KEY='...' \
  TRIMUL_JUDGE_TOKEN="$(openssl rand -hex 32)"
```

Select it at launch:

```bash
export GUIDANCE_TTT_TRIMUL_SECRET=guidance-ttt-trimul-secrets
```

The task-specific judge defaults to `guidance-ttt-trimul-secrets`. For
compatibility with the existing workspace, the combined H200 launcher defaults
to `guidance-ttt-vliw-secrets` and accepts its legacy `VLIW_JUDGE_TOKEN` as the
bearer value. Candidate subprocesses receive neither API key nor bearer token.

## One-step acceptance run

Install only launcher-side dependencies locally:

```bash
pip install -r requirements-modal.txt
```

The normal checks can be run separately:

```bash
modal run scripts/modal_trimul_h200_smoke.py --action runtime_probe
modal run scripts/modal_trimul_h200_smoke.py --action judge_probe
modal run scripts/modal_trimul_h200_smoke.py --action bootstrap --prompt-mode summary_only
modal run scripts/modal_trimul_h200_smoke.py --action train --prompt-mode summary_only
```

Here `bootstrap` copies and audits the tracked fixed seed. It does not call
GLM-5.2. The one-off scratch-generation path is separate:

```bash
modal run scripts/modal_trimul_h200_smoke.py \
  --action generate_scratch_seed \
  --prompt-mode summary_only
```

Or run the complete flow:

```bash
modal run scripts/modal_trimul_h200_smoke.py --action smoke --prompt-mode summary_only
```

Omit `--prompt-mode` or set it to `code_delta` to run the original matched
recipe. The two modes use separate run directories, so one smoke cannot
overwrite the other.

### Local B200 high-thinking cache smoke

For the small local-B200 acceptance run, deploy the combined evaluator with
four containers and submit the group-of-four recipe:

```bash
TRIMUL_JUDGE_MAX_CONTAINERS=4 modal deploy scripts/modal_trimul_h200_smoke.py
sbatch scripts/slurm_trimul_b200_1gpu_evolvent_glm52_thinking_cache_group4_smoke.sbatch
```

For the Polyomino-matched 8x16 run, deploy the task-specific evaluator and
probe all 16 isolated H100 containers before requesting one two-hour B200 job:

```bash
TRIMUL_JUDGE_MAX_CONTAINERS=16 modal deploy scripts/modal_trimul_judge.py
python scripts/probe_trimul_judge_concurrency.py --concurrency 16 --requests 16
sbatch scripts/slurm_trimul_b200_1gpu_evolvent_glm52_thinking_cache_batch8_group16_smoke.sbatch
```

The 8x16 config uses one local B200 and matches the run, PPO, LoRA, vLLM, and
Discover parameters of the 91.89074282 Polyomino run. Only task-specific
execution, verifier, one-step duration, and no-checkpoint smoke controls differ.
It permits 32 concurrent external GLM requests so 128 long high-thinking
generations fit inside the two-hour allocation; H100 evaluator concurrency
remains 16.

This path requests GLM-5.2 `reasoning_effort: high` without changing its prompt
or temperature. Evolvent currently runs an older LiteLLM DashScope capability
table, so the request also sends
`allowed_openai_params: [reasoning_effort]`; otherwise the proxy rejects the
parameter before it reaches the provider. Do not combine this with
`enable_thinking: true`, which is only a binary thinking switch and does not
identify the high effort tier. It requests an explicit 81920-token output
budget, matching the observed effective ceiling of the earlier OpenRouter
GLM-5.2 run instead of relying on a provider-specific default. This differs
from the old YAML's `null` only by making that effective ceiling explicit.
Execution responses use SSE streaming so long reasoning runs do not leave an
idle HTTP connection. Uninterrupted streams reconstruct final reasoning,
content, usage, and cached-token accounting exactly. If the upstream stream
boundary arrives before a final answer, the client preserves the reasoning
trace as assistant `reasoning_content` and makes a thinking-disabled final-only
continuation with a separate 65536-token budget instead of discarding the
reasoning and resampling. Recovery metadata records both phases, but token
usage remains conservative when the interrupted phase did not emit a final
usage event. Before the candidate requests, the client sends one high-thinking
request capped at one output token, waits four seconds for gateway cache
propagation, and then releases the full-budget requests.
The primer is not treated as a candidate; every candidate retains the same
prompt, sampling settings, and 81920-token budget. This avoids waiting for a
full high-thinking response long enough for the prefix cache to expire.

The 2026-08-09 acceptance run (`SLURM_JOB_ID=330467`) completed in 19m23s.
All four full-budget candidates retained non-empty reasoning and parsed final
answers; three passed the official H100 evaluator. The three uninterrupted
responses cached 12032 of 12752 prompt tokens (94.35%), while one incomplete
upstream stream required final-only recovery and did not expose its initial
cache usage. Including that recovery prompt, the conservative validator total
was 12032 of 32156 tokens (37.42%). The best child ran in 9084.976 us versus
the 10177.397 us frozen root. The one-step actor update completed without OOM
or checkpoint output.

The evaluator cap can be raised independently of the training recipe. On
2026-08-09, the fixed valid seed passed a 16-request probe in 33.135 s: all 16
requests used `NVIDIA H100 80GB HBM3`, passed correctness, and ran in distinct
single-input containers. A separate 16-request Evolvent probe completed in
2.542 s after a one-token primer; all responses contained reasoning and 15 of
16 reported cached input tokens. A subsequent 32-request probe ran while 16
full generations were active: all 32 retained thinking output, none returned a
capacity error, 27 reported cached input, and wall time was 2.965 s.
Cache reuse is therefore best-effort across gateway backend shards, while
request quality settings remain unchanged.

The matched 8x16 acceptance run (`SLURM_JOB_ID=331053`) completed on one B200
in 47m35s with exit code 0. Its archive contains eight valid roots, eight
finalized groups, 128 submitted children, and `puct_T=128`. All 128 children
retained reasoning, parsed solution and summary fields, streamed successfully,
and ended with `finish_reason=stop`; 85 were H100-valid, 40 were invalid, and
three were model-level parse errors. Nine interrupted streams completed through
final-only recovery. The gateway reported cache hits for 88 children and
324992 cached prompt tokens out of 832617 reported prompt tokens (39.03%).

The best child ran in 3330.475 us versus the 10177.397 us frozen root, a 67.27%
runtime reduction. The actor update completed at `training/global_step=1` with
`actor/pg_loss=0.0555206`, `actor/grad_norm=0.096341`, learning rate `4e-5`,
651203 training tokens, a non-empty optimization mask, and zero aborted
responses. No OOM or checkpoint output occurred. A DataLoader worker warning
appeared only during temporary-directory cleanup after training reached 100%;
both validators and Slurm still completed with exit code 0.

TTT-Discover's default configuration creates eight groups of 64 rollouts (512
solutions per step). It starts all groups concurrently in
[`train.py`](https://github.com/test-time-training/discover/blob/6c40e82dab9d5de7416ac873ad5cd3106084aaed/ttt_discover/rl/train.py)
and all rollouts inside each group concurrently in
[`rollouts.py`](https://github.com/test-time-training/discover/blob/6c40e82dab9d5de7416ac873ad5cd3106084aaed/ttt_discover/rl/rollouts.py).
Its GPU Mode Modal functions do not set `max_containers`, so actual evaluator
fan-out is bounded by Modal autoscaling and workspace quota rather than by an
application-side semaphore. Our explicit cap makes cost and failure behavior
predictable while retaining the same asynchronous evaluation structure.

### Existing Modal H200 acceptance smoke

The smoke recipe uses one H200 for Qwen3-8B, deterministic GLM-5.2 execution,
and one group of two guidance rollouts. It accepts the run only when bootstrap
loads one officially valid frozen root, the training command completes, exactly two
children and one finalized group are persisted, at least one child contains
parseable code plus a raw summary, at least one child passes the official H100
evaluator, and PUCT records one group-wise update.

Earlier 2026-08-09 integration smokes used the archived Discover result while
the task transport and evaluator were being validated. Those runs completed the
full one-step GRPO path, but their root scores are not results for the current
scratch-seed protocol. The fixed GLM-5.2 root above is now the common starting
point for both prompt-mode comparisons.

Persistent output is stored on the `guidance-ttt-runs` Volume at:

```text
/runs/guidance_ttt/trimul_modal_h200_1gpu_evolvent_glm52_group2_1step/
/runs/guidance_ttt/trimul_modal_h200_1gpu_evolvent_glm52_group2_summary_only_1step/
```

The final audit file is `modal_smoke_result.json`; complete prompts, responses,
solutions, summaries, scores, and per-case verifier artifacts remain in
`library.json` and the normal Guidance-TTT rollout dumps.
