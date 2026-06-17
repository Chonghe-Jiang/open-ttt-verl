# Erdos OpenEvolve Pipeline

This branch is a standalone OpenEvolve experiment for the Erdos minimum overlap
problem. It is intentionally separate from the `guidance_ttt` / `verl` training
pipeline: the trainable object here is a Python candidate program, not a policy
model.

The goal is to let OpenEvolve repeatedly edit `initial_program.py`, run each
candidate through `evaluator.py`, and keep candidates that reduce the Erdos C5
overlap score. The score exposed to OpenEvolve is:

```text
combined_score = 1 / (1e-8 + C5)
```

So a lower verified C5 produces a higher evolution score.

The Erdos prompt and verifier behavior are adapted from the local
`Guidance-ttt/guidance` project, but this folder is self-contained at runtime:
it does not import `guidance_ttt`.

OpenEvolve itself is used from the local `./openevolve` checkout. In this branch
it is recorded as a git submodule pointing at
`https://github.com/algorithmicsuperintelligence/openevolve`.

## What This Folder Does

1. Stores a baseline Erdos candidate in `initial_program.py`.
2. Lets OpenEvolve mutate only the marked evolve block in that program.
3. Executes each mutated candidate in a timeout-limited sandbox.
4. Verifies the candidate satisfies the Erdos constraints.
5. Reports OpenEvolve metrics and artifacts, including validity, C5, reward, and
   failure messages.
6. Provides no-API smoke tests now, plus OpenAI-compatible config for future real
   API runs.

## Layout

```text
openevolve/                    # Local clone of algorithmicsuperintelligence/openevolve
initial_program.py             # Candidate program evolved by OpenEvolve
evaluator.py                   # OpenEvolve evaluator entrypoint
erdos_evolve/                  # Problem prompt, sandbox, verifier
configs/                       # No-API and OpenAI-compatible config templates
scripts/                       # Local validation and run helpers
tests/                         # No-API unit tests
```

## Clone This Branch

Because `openevolve/` is a submodule, initialize it after checkout:

```bash
git clone --branch openevolve --recurse-submodules \
  https://github.com/Chonghe-Jiang/open-ttt-verl.git
cd open-ttt-verl
```

If the repository was cloned without `--recurse-submodules`, run:

```bash
git submodule update --init --recursive
```

## Local Setup

```bash
python -m pip install -r requirements.txt
bash scripts/run_local_checks.sh
```

`requirements.txt` installs `./openevolve` in editable mode. The local checks do not require
an LLM API key.

## Candidate Contract

OpenEvolve edits only the `# EVOLVE-BLOCK-START` section in `initial_program.py`.
The public interface must remain:

```python
def run(seed=42, budget_s=1, n_points=256, **kwargs):
    return (h_values, c5_bound, n_points)
```

The verifier requires:

- `h_values` is one-dimensional and length `n_points`
- every value is finite and in `[0, 1]`
- `sum(h_values) == n_points / 2`
- `c5_bound` matches `max(np.correlate(h, 1-h, mode="full") * (2.0 / n_points))`

Lower raw C5 is better. The OpenEvolve `combined_score` is `1 / (1e-8 + C5)`.

## Run With An API Later

Set either OpenEvolve-style env vars:

```bash
export OPENAI_API_KEY="..."
export OPENAI_API_BASE="https://your-openai-compatible-endpoint/v1"
```

or guidance-style env vars:

```bash
export API_KEY="..."
export ENDPOINT="https://your-openai-compatible-endpoint/v1"
```

Then run:

```bash
bash scripts/run_openevolve.sh --iterations 20
```

For a no-API OpenEvolve smoke test that only evaluates the initial program:

```bash
bash scripts/run_openevolve.sh --smoke --output /tmp/erdos_openevolve_smoke
```

The script calls:

```bash
python openevolve/openevolve-run.py initial_program.py evaluator.py \
  --config configs/erdos_openai_compatible.example.yaml
```

## Local Qwen3-8B via vLLM

To run OpenEvolve with local Qwen3-8B inference instead of an external API,
start a vLLM OpenAI-compatible server in one terminal:

```bash
bash scripts/start_vllm_qwen3_8b.sh
```

By default this runs:

```text
vllm serve Qwen/Qwen3-8B --host 127.0.0.1 --port 8000 --served-model-name Qwen/Qwen3-8B
```

Then run OpenEvolve in another terminal:

```bash
bash scripts/run_openevolve.sh --local-qwen-vllm --iterations 20
```

The local mode uses `configs/erdos_qwen3_8b_vllm.yaml`, checks
`http://127.0.0.1:8000/v1/models` before starting evolution, and sends requests
only to the localhost vLLM server. If `VLLM_BASE_URL` or `VLLM_MODEL` is set,
the run script passes those values through to OpenEvolve with `--api-base` and
`--primary-model`.

Useful overrides:

```bash
VLLM_MODEL=Qwen/Qwen3-8B \
VLLM_PORT=8000 \
VLLM_BASE_URL=http://127.0.0.1:8000/v1 \
VLLM_MAX_MODEL_LEN=8192 \
VLLM_GPU_MEMORY_UTILIZATION=0.90 \
VLLM_TENSOR_PARALLEL_SIZE=1 \
bash scripts/start_vllm_qwen3_8b.sh
```

## Relationship To Guidance-TTT

The `guidance` project uses a guidance model, an execution model, and a verifier
inside a `verl`/TTT rollout loop. This branch keeps only the Erdos problem and
verifier contract, then maps it to OpenEvolve:

```text
OpenEvolve edits candidate code
-> evaluator executes run(...)
-> verifier recomputes C5
-> OpenEvolve receives combined_score and artifacts
```

This makes the branch useful for quick program-evolution experiments before
connecting any real model API.
