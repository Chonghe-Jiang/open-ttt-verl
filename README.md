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
- Guidance and execution prompts attach raw library summaries selected by PUCT,
  not node metadata or verifier artifacts.
- Execution prompt also attaches the parsed `<guidance>` and the current visible
  best valid summary.
- Execution LLM returns `<execution_thinking>`, one task-specific `<solution>`
  code block, and `<summary>` in a single response.
- The verifier reward is assigned only to guidance model response tokens.
- Execution failures are environment outcomes and become library entries with reward `0.0`.
- The execution-provided summary is stored with verifier reward/status as structured library metadata; the summary should not claim verifier success before verification runs.
