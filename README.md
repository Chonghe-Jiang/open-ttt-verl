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

## Current Approach

The current Guidance-TTT design trains only the guidance actor. The execution
model is treated as a frozen solver that converts a high-level guidance idea plus
the selected library summary into one concrete candidate.

At rollout time:

1. The JSON library uses PUCT to select one visible node.
2. The guidance actor receives the problem statement and the selected node's raw
   execution summary plus verifier score/status. It returns exactly one
   `<guidance>` block.
3. The execution model receives the same selected summary/score context plus the
   parsed guidance. It returns `<execution_thinking>`, `<solution>`, and
   `<summary>` in one response.
4. The verifier scores the solution. For Polyomino, this always goes through
   FrontierCS/go-judge.
5. The new entry is written as a child of the selected PUCT node. Its raw model
   summary, verifier score/status, execution text, and solution are stored in
   `library.json`.
6. GRPO/verl assigns the verifier reward to the guidance response tokens only;
   execution-model tokens are not trained.

For current Polyomino Modal experiments, the recommended path is the 3xH200
GPT-OSS-120B execution recipe:

```bash
WORKSPACE=<modal-workspace-name-or-id> \
modal run --detach scripts/modal_polyomino_h200_smoke.py \
  --action train_gpt_oss_120b_3gpu_group16_h200_tuned
```

This recipe trains the Qwen3-8B guidance actor for 50 steps, saves guidance
actor checkpoints every 5 steps, and uses a local Modal-hosted
`openai/gpt-oss-120b` vLLM server as the frozen execution model. It initializes
the run directory from
`guidance_ttt/seeds/polyomino_packing/openrouter_gpt55_bootstrap_library.json`
instead of spending an execution call on a fresh bootstrap candidate.

### Library Settings

`library.json` is not just history; if it contains saved library config, that
config is treated as the source of truth when `GuidanceLibrary` reloads. In
practice:

- A fresh run without an existing `library.json` uses the YAML values for
  `ttt.group_size`, `ttt.puct_c`, `ttt.max_buffer_size`, and
  `ttt.topk_children`.
- A resumed run with an existing `library.json` reloads `config.rollout_n`,
  `config.puct_c`, `config.max_buffer_size`, and `config.topk_children` from the
  library file. This prevents a resumed search tree from silently changing its
  PUCT semantics.
- The static Polyomino seed library intentionally does not store a `config`
  block. Therefore the seeded smoke still takes rollout/group settings from the
  active YAML config.

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

## Install

For local development and tests:

```bash
python -m pip install -r requirements-ttt.txt
python -m pip install -r requirements-test.txt
```

For launching Modal Polyomino runs from your local machine:

```bash
python -m pip install -r requirements-modal.txt
modal token new
```

The Modal script builds the GPU runtime remotely. It installs FrontierCS,
go-judge, vLLM, PyTorch dependencies, and the H200-compatible `flash_attn`
wheel inside the Modal image. Local CUDA-specific packages are therefore not
required just to submit the job.

## Prepare a Smoke Run

Erdos:

```bash
python -m guidance_ttt.main_erdos \
  --config guidance_ttt/config/backup/erdos_smoke.yaml \
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
  --config guidance_ttt/config/backup/polyomino_frontiercs.yaml \
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
  --config guidance_ttt/config/backup/polyomino_frontiercs.yaml \
  --prepare-only \
  task.frontiercs.base_dir=/path/to/Frontier-CS
```

Installing FrontierCS into a shared training environment can downgrade or pin
packages such as `protobuf`, `click`, and `cryptography`. Use a dedicated env if
those conflicts matter for other workloads.

## Modal Polyomino Runs

The Modal entrypoint is:

```bash
python -m pip install -r requirements-modal.txt
modal token info
```

Then launch from the repository root:

```bash
modal run scripts/modal_polyomino_h200_smoke.py --action verifier
modal run scripts/modal_polyomino_h200_smoke.py --action train
modal run scripts/modal_polyomino_h200_smoke.py --action train_history
modal run scripts/modal_polyomino_h200_smoke.py --action bootstrap_single_summary
modal run scripts/modal_polyomino_h200_smoke.py --action train_single_summary
modal run scripts/modal_polyomino_h200_smoke.py --action train_single_summary_openrouter_gpt55_seeded
WORKSPACE=<modal-workspace-name-or-id> modal run --detach scripts/modal_polyomino_h200_smoke.py --action train_gpt_oss_120b_3gpu_group16_h200_tuned
```

For the one-summary Modal run, use:

```bash
scripts/run_modal_polyomino_single_summary.sh
```

The launcher first runs `bootstrap_single_summary` synchronously, which asks the
execution model for one baseline candidate, verifies it, and attaches its summary
to the root library node. It then uses `modal run --detach` for
`train_single_summary` so the remote training function is not stopped if the
local client disconnects during model loading.

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
  `guidance_ttt/config/backup/polyomino_modal_h200_4gpu_smoke.yaml`.
- `train_history`: runs the 2xH200 one-step history extraction smoke config at
  `guidance_ttt/config/backup/polyomino_modal_h200_2gpu_history_smoke.yaml`. This is
  the cheaper check for guidance parsing, raw execution summary extraction,
  canonical library summary writing, and execution text capture.
- `bootstrap_single_summary`: resets the
  `polyomino_modal_h200_2gpu_single_summary` output directory, runs the
  execution-only bootstrap pass, starts FrontierCS/go-judge for verification,
  and writes a root-attached library entry before training.
- `train_single_summary`: runs a 2xH200 one-step Polyomino config with
  `Qwen/Qwen3-8B` as the guidance actor and `openai/gpt-oss-20b` as the local
  vLLM execution model. It requires the bootstrap entry to exist, uses one slot
  with the minimum two rollouts required by verl, prunes the output to the best
  guided entry, then validates that `library.json` contains exactly one valid
  entry with a summary.
- `prune_single_summary`: re-validates and prunes an existing
  `polyomino_modal_h200_2gpu_single_summary` output directory without rerunning
  the GPU training step.
- `train_single_summary_openrouter_gpt55_seeded`: runs the 2xH200 one-step
  Polyomino config with `Qwen/Qwen3-8B` as the guidance actor and
  `openai/gpt-5.5` through OpenRouter as the execution model. The run resets its
  output directory and initializes `library.json` from the static seed library at
  `guidance_ttt/seeds/polyomino_packing/openrouter_gpt55_bootstrap_library.json`,
  so it does not call the bootstrap-only API path before training. This action
  requires a Modal secret named `openrouter-api-key` with `OPENROUTER_API_KEY`.
- `bootstrap_gpt_oss_120b_seed`: runs a 1xH200 bootstrap-only Polyomino pass
  with local `openai/gpt-oss-120b` execution and FrontierCS verification. It
  writes a fresh seed candidate to
  `/runs/guidance_ttt/polyomino_modal_h200_gpt_oss_120b_bootstrap_seed/library.json`;
  copy that file to
  `guidance_ttt/seeds/polyomino_packing/gpt_oss_120b_bootstrap_library.json`
  when refreshing the static seed used by the recommended 120B recipe.
- `train_gpt_oss_120b_3gpu_group16_h200_tuned`: current recommended 50-step
  Polyomino training recipe. It uses 2 H200s for the Qwen3-8B guidance actor
  and one H200 for the local `openai/gpt-oss-120b` execution server. The active
  config is
  `guidance_ttt/config/polyomino_modal_h200_3gpu_gpt_oss_120b_group16_h200_tuned.yaml`.
  The run uses `groups_per_batch: 4`, `group_size: 16`, execution concurrency
  8, `max_prompt_length: 4096`, `max_response_length: 8192`,
  `filter_overlong_prompts: false`, and `truncation: middle`.
- `train_gpt_oss_120b_3gpu_batch8_group8_temp09`: exploratory 20-step
  Polyomino recipe with the same 2 H200 guidance actor plus 1 H200 local
  `openai/gpt-oss-120b` execution layout. The active config is
  `guidance_ttt/config/polyomino_modal_h200_3gpu_gpt_oss_120b_batch8_group8_temp09.yaml`.
  It uses `groups_per_batch: 8`, `group_size: 8`, guidance rollout
  `temperature: 0.9`, `top_p: 0.95`, greedy execution, execution concurrency
  8, `max_prompt_length: 4096`, and `max_response_length: 8192`.

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
/runs/guidance_ttt/polyomino_modal_h200_2gpu_openrouter_gpt55_single_summary
/runs/guidance_ttt/polyomino_modal_h200_gpt_oss_120b_bootstrap_seed
/runs/guidance_ttt/polyomino_modal_h200_3gpu_gpt_oss_120b_group16_h200_tuned_50step
/runs/guidance_ttt/polyomino_modal_h200_3gpu_gpt_oss_120b_batch8_group8_temp09_20step
```

For the 50-step GPT-OSS-120B recipe, guidance actor checkpoints are written to:

```text
/runs/guidance_ttt/polyomino_modal_h200_3gpu_gpt_oss_120b_group16_h200_tuned_50step/checkpoints/global_step_5/actor
/runs/guidance_ttt/polyomino_modal_h200_3gpu_gpt_oss_120b_group16_h200_tuned_50step/checkpoints/global_step_10/actor
...
```

Modal currently limits a single function timeout to 24 hours. The 50-step run
can exceed that wall-clock time because execution/verifier rollout dominates
each step, so checkpointing every 5 steps is intentional.

The smoke configs intentionally keep only the training shape small: `num_steps`,
`groups_per_batch`, `group_size`, PPO mini-batch size, and GPU count. They are
not meant to be quality-oriented training recipes; they are end-to-end checks
that the guidance prompt, execution prompt, verifier, library writeback, and
GRPO update still work together.

Execution generation is not artificially shortened in these configs:

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

For seeded smoke runs, the initial candidate comes from the static library file,
but the smoke shape still comes from the active YAML. The seed file does not
carry `rollout_n` or PUCT config. The recommended GPT-OSS-120B recipe uses
`guidance_ttt/seeds/polyomino_packing/gpt_oss_120b_bootstrap_library.json`;
the older OpenRouter GPT-5.5 seeded smoke keeps using
`guidance_ttt/seeds/polyomino_packing/openrouter_gpt55_bootstrap_library.json`.

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
  --config guidance_ttt/config/backup/erdos_gpt_oss_20b_5step.yaml
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
