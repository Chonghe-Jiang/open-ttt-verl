# Guidance + Execution TTT

This repository implements a self-contained Guidance-TTT prototype on top of
`open-ttt-verl`. It includes a local `verl/` tree, so the guidance recipe no
longer depends on `/reference/open-ttt-verl` at runtime.

The rollout loop is:

```text
PUCT library node
-> trainable guidance model
-> guidance / idea
-> execution LLM
-> execution thinking + solution code + summary
-> verifier reward
-> new library child node
-> RL update on guidance tokens only
```

Supported tasks:

- `erdos_min_overlap`: execution returns a Python `run()` candidate; lower raw
  C5 is better.
- `polyomino_packing`: execution returns a complete C++17 program; scoring is
  delegated to the external FrontierCS/go-judge/Docker evaluator; higher
  FrontierCS score is better.

Execution uses `MockLLMClient` by default so the package can be tested without a
real API. Real execution can use either an OpenAI-compatible endpoint or a local
Transformers model with `llm.execution.provider=local`.

## Repository Layout

```text
verl/                 # local verl runtime and trainer config
guidance_ttt/         # Guidance-TTT agent loop, library, prompts, verifier
docs/                 # prompt architecture and task notes
scripts/              # convenience launchers
tests/                # lightweight Guidance-TTT tests
```

The default verl config directory is `verl/trainer/config` in this repository.
`run.verl_config_dir=...` can still override it for debugging.

## Prepare a Smoke Run

Erdos:

```bash
python -m guidance_ttt.main_erdos \
  --config guidance_ttt/config/erdos_smoke.yaml \
  --prepare-only
```

This writes:

```text
outputs/guidance_ttt/erdos_smoke/library.json
outputs/guidance_ttt/erdos_smoke/ttt_slots.parquet
outputs/guidance_ttt/erdos_smoke/agent_loop.yaml
```

Polyomino:

```bash
python -m guidance_ttt.main_erdos \
  --config guidance_ttt/config/polyomino_frontiercs.yaml \
  --prepare-only
```

`main_erdos` is kept as the compatibility entrypoint; internally it now prepares
the task selected by `task.id`.

## Polyomino / FrontierCS Setup

Polyomino evaluation always goes through FrontierCS. There is no local smoke
evaluator and no approximate fallback score.

Install the Python package:

```bash
python -m pip install 'git+https://github.com/FrontierCS/Frontier-CS.git'
```

Clone the benchmark data outside this repo. The default config expects this
layout, with `reference/` next to `guidance/`:

```bash
mkdir -p ../reference
git clone https://github.com/FrontierCS/Frontier-CS.git ../reference/Frontier-CS
```

Requirements:

```bash
python -c "import frontier_cs"
docker --version
```

The FrontierCS algorithmic runner auto-starts go-judge with Docker Compose from
`task.frontiercs.base_dir/algorithmic`. If your checkout lives elsewhere,
override:

```bash
python -m guidance_ttt.main_erdos \
  --config guidance_ttt/config/polyomino_frontiercs.yaml \
  --prepare-only \
  task.frontiercs.base_dir=/path/to/Frontier-CS
```

Installing FrontierCS into a shared training environment can downgrade or pin
packages such as `protobuf`, `click`, and `cryptography`. Use a dedicated env if
those conflicts matter for other workloads.

## Modal Polyomino Runs

The Modal entrypoint is:

```bash
python -m pip install modal
modal token new
```

Then launch from the repository root:

```bash
modal run scripts/modal_polyomino_h200_smoke.py --action verifier
modal run scripts/modal_polyomino_h200_smoke.py --action train
modal run scripts/modal_polyomino_h200_smoke.py --action train_history
modal run scripts/modal_polyomino_h200_smoke.py --action train_single_summary
```

For the one-summary Modal run, use:

```bash
scripts/run_modal_polyomino_single_summary.sh
```

The launcher uses `modal run --detach` so the remote training function is not
stopped if the local client disconnects during model loading.

The launcher requires `.env` to contain `WORKSPACE=<modal-workspace-name-or-id>`.
Before submitting the H200 job, it checks `modal token info` and exits unless
the active Modal token is connected to that exact workspace name or workspace
id. If `WORKSPACE` also matches a local Modal profile name, the launcher exports
it as `MODAL_PROFILE` before running Modal.

If `.env` contains `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`, the launcher exports
both before calling Modal, so those credentials take precedence over the active
profile. `MODAL_SECRET_KEY` is also accepted as a compatibility alias for
`MODAL_TOKEN_SECRET`. If `.env` contains only `MODAL_TOKEN_ID`, the launcher
looks for a matching token id in `~/.modal.toml` and uses that saved token
secret. A token id with no matching secret is rejected because Modal API token
auth requires both values.

Actions:

- `verifier`: starts the FrontierCS/go-judge service inside Modal and evaluates
  a baseline C++ solution. This checks that the remote judge path works before
  training.
- `train`: runs the 4xH200 one-step Polyomino Guidance-TTT smoke config at
  `guidance_ttt/config/polyomino_modal_h200_4gpu_smoke.yaml`.
- `train_history`: runs the 2xH200 one-step history extraction smoke config at
  `guidance_ttt/config/polyomino_modal_h200_2gpu_history_smoke.yaml`. This is
  the cheaper check for guidance parsing, raw execution summary extraction,
  canonical library summary writing, and execution text capture.
- `train_single_summary`: runs a 2xH200 one-step Polyomino config with
  `Qwen/Qwen3-8B` as the guidance actor and `openai/gpt-oss-20b` as the local
  vLLM execution model. It uses one slot with the minimum two rollouts required
  by verl, prunes the output to the best guided entry, then validates that
  `library.json` contains exactly one valid entry with a summary.
- `prune_single_summary`: re-validates and prunes an existing
  `polyomino_modal_h200_2gpu_single_summary` output directory without rerunning
  the GPU training step.

The Modal script builds a remote image with FrontierCS and go-judge, copies this
repository to `/root/guidance`, and uses the sparse FrontierCS checkout at
`/opt/Frontier-CS`. It mounts:

```text
guidance-ttt-runs  -> /runs
guidance-ttt-cache -> /cache
```

Training outputs are written to:

```text
/runs/guidance_ttt/polyomino_modal_h200_4gpu_smoke
/runs/guidance_ttt/polyomino_modal_h200_2gpu_history_smoke
/runs/guidance_ttt/polyomino_modal_h200_2gpu_single_summary
```

The smoke configs intentionally keep only the training shape small: `num_steps`,
`groups_per_batch`, `group_size`, PPO mini-batch size, and GPU count. Execution
generation is not artificially shortened in these configs:

```yaml
llm:
  execution:
    max_tokens: null
    phase1_max_tokens: null
```

FrontierCS/go-judge concurrency is capped at 8 in
`scripts/modal_polyomino_h200_smoke.py`; the local vLLM execution batch settings
are aligned with the selected smoke size. If you change GPU count or group size,
update the YAML config and the Modal script constants together.

## Local gpt-oss-20b Execution

Download the local execution model:

```bash
scripts/download_gpt_oss_20b.sh
```

Run the 5-step Erdos minimum-overlap recipe. The trainable guidance actor uses
`Qwen/Qwen3-8B`; execution uses the downloaded `models/gpt-oss-20b` through the
local Transformers client.

```bash
scripts/run_erdos_gpt_oss_20b_5step.sh
```

After a completed run, write or refresh the params, prompt/output, and per-step
minimum C5 score summary:

```bash
python -m guidance_ttt.run_summary \
  outputs/guidance_ttt/erdos_gpt_oss_20b_5step \
  --config guidance_ttt/config/erdos_gpt_oss_20b_5step.yaml
```

## Tests

```bash
pytest -q tests
python -m compileall -q guidance_ttt verl
```

## Design Notes

- PUCT selects one library node before prompt assembly.
- Guidance and execution prompts attach raw execution-model summaries plus
  verifier score/status for the PUCT-selected entry, not the full canonical
  summary or global best summary.
- Execution prompt also attaches the parsed `<guidance>`.
- Execution LLM returns `<execution_thinking>`, one task-specific `<solution>`
  code block, and `<summary>` in a single response.
- The verifier reward is assigned only to guidance model response tokens.
- Execution failures are environment outcomes and become library entries with reward `0.0`.
- The execution-provided summary is stored with verifier reward/status as structured library metadata; the summary should not claim verifier success before verification runs.
