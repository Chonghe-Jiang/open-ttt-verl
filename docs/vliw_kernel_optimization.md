# EdgeBench VLIW Kernel Optimization

Guidance-TTT exposes the official EdgeBench `vliw_kernel_optimization` task as
`task.id: vliw_kernel_optimization`. The trainable Qwen3-8B model proposes a
high-level optimization direction, while a frozen execution model rewrites the
selected parent `solution.py`. Every generated candidate is evaluated by the
official hidden-case runner.

## Candidate interface

The submitted artifact is a complete Python file defining
`KernelBuilder.build_kernel(self, forest_height, n_nodes, batch_size, rounds)`.
The method emits instruction bundles for EdgeBench's custom VLIW/SIMD machine.
The hidden judge checks the final indices and values against the reference
implementation and measures simulator cycles.

The library stores the complete candidate source, its raw delta summary, judge
status, cycle count, reward, guidance, and parent relationship. In
`code_delta` mode:

```text
PUCT-selected parent code + delta summary + cycles
-> Qwen3-8B guidance
-> GLM-5.2 execution
-> complete solution.py + delta summary
-> official EdgeBench hidden-case judge
-> cycles/reward + new library child
-> GRPO update on guidance tokens
```

Bootstrap starts from the exact official starter. GLM-5.2 is asked for one
conservative, judgeable improvement; when it cannot make a safe optimization,
it may retain the verified starter and state that no change was made. The
bootstrap response is parsed strictly and must pass the official judge before
training starts.

## Metric and reward

`score_cycles` is the authoritative raw task metric and **lower is better**. It
is the maximum cycle count across the judge's performance cases after all
correctness cases pass. Any incorrect, malformed, or timed-out candidate is
invalid.

The EdgeBench reporting score is retained in verifier artifacts. That mapping
clips candidates above its baseline threshold to zero, including the official
starter and many early valid optimizations. The one-step training recipe
therefore uses this dense maximize-oriented reward:

```text
reward = 1,000,000 / score_cycles
```

Invalid candidates receive zero. The raw cycle count remains in
`verifier_raw_score`, and the library converts the minimize-oriented raw metric
before PUCT comparison. This reward shaping affects RL credit assignment; it
does not replace the official cycle metric used for reporting.

## Verifier environments

The implementation supports two equivalent verifier transports:

- `provider: docker` runs the pinned official judge image locally with no
  network, bounded CPU/memory, and the generated `solution.py` mounted into the
  judge workspace.
- `provider: modal_http` sends the source to an authenticated Modal CPU
  endpoint built directly from the same pinned judge image. Each endpoint
  container accepts one evaluation at a time, and Modal scales to 16 containers.

The GPU training container does not need a Docker daemon. It only calls the
isolated CPU verifier service. The active smoke recipe limits both execution
API calls and verifier calls to 16 concurrent requests. Each judge run has a
60-second runner limit; the surrounding request and agent-loop limits are 90
and 120 seconds respectively.

## Modal smoke

Install the local launcher dependency and configure Modal:

```bash
python -m pip install -r requirements-modal.txt
modal token new
```

Create one Modal secret. Use the Evolvent key for `EVOLVENT_API_KEY` and an
independent random bearer token for `VLIW_JUDGE_TOKEN`; do not put either value
in YAML or git:

```bash
modal secret create guidance-ttt-vliw-secrets \
  EVOLVENT_API_KEY='<evolvent-api-key>' \
  VLIW_JUDGE_TOKEN='<random-verifier-token>'
```

Run the inexpensive CPU acceptance checks first:

```bash
modal run scripts/modal_vliw_kernel_h200_smoke.py --action judge_probe
modal run scripts/modal_vliw_kernel_h200_smoke.py --action runtime_probe
```

Run bootstrap and the complete smoke together:

```bash
modal run scripts/modal_vliw_kernel_h200_smoke.py --action smoke
```

For operational debugging, the two phases can also be launched separately.
Bootstrap is CPU-only and commits the validated root to the shared Volume;
`train` requires exactly that one root and no existing children:

```bash
modal run scripts/modal_vliw_kernel_h200_smoke.py --action bootstrap
modal run scripts/modal_vliw_kernel_h200_smoke.py --action train
```

The active recipe is
`guidance_ttt/config/vliw_kernel_modal_h200_2gpu_glm52_batch8_group16_1step.yaml`:

| Setting | Value |
| --- | --- |
| Guidance model | `Qwen/Qwen3-8B`, LoRA rank/alpha `32/32` |
| Execution model | Evolvent `glm-5.2`, greedy, `enable_thinking=false` requested |
| Training GPUs | `2 x H200` |
| Training steps | `1` |
| Groups | `8` |
| Rollouts per group | `16` (`128` total) |
| Guidance sampling | temperature `0.9`, top-p `0.95` |
| Prompt/response limits | `8192` / `8192` tokens |
| Execution/verifier concurrency | `16` / `16` |
| Judge runner timeout | `60` seconds per candidate |

Persistent artifacts are written to the `guidance-ttt-runs` Modal Volume at:

```text
guidance_ttt/vliw_kernel_modal_h200_2gpu_glm52_batch8_group16_1step/
```

The smoke validator requires one valid root, exactly 128 children, all eight
groups finalized, group-wise `puct_T=8`, at least one parsed and officially
valid child with a raw summary, at least one positive reward, complete API
finish-reason accounting, GLM-5.2 provenance for every child, and a completed
trainer update.

The accepted one-step run on 2026-08-08 produced 128 children, finalized all
eight groups, and completed one GRPO actor update. Four children passed the
official hidden-case judge; the best reduced the generated seed from 114,966 to
28,439 cycles. Of the 128 GLM-5.2 responses, 75 ended with `stop` and 53 reached
the configured 8,192-token completion limit. Two verifier-valid children also
contained complete raw summaries, so the full guidance, execution, summary,
verification, library, and training path was exercised. The high truncation
rate is execution-model behavior to account for when turning this smoke recipe
into a longer search, not a verifier or GRPO orchestration failure.

## Pinned official artifacts

The integration pins these images rather than depending on mutable tags:

```text
seededge/edgebench.work.vliw_kernel_optimization:9fa380a0ebef
seededge/edgebench.judge.vliw_kernel_optimization:5cdef0021634
```

The checked-in starter and attribution details are documented in
`guidance_ttt/tasks/assets/vliw_kernel_optimization/SOURCE.md`. Hidden cases and
judge implementation files are never copied into this repository.
