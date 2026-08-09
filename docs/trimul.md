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

Each endpoint container accepts one request at a time. The smoke app permits at
most two evaluator containers, matching `task.verifier.concurrency: 2`.
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

For compatibility with the existing workspace, the launcher defaults to
`guidance-ttt-vliw-secrets` and accepts its legacy `VLIW_JUDGE_TOKEN` as the
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
