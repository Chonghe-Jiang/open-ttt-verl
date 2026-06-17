# TTT-Discover on verl

This package implements the TTT-Discover/Erdos recipe while delegating the heavy
RL execution to verl: rollout, logprobs, rollout-importance correction, batching,
advantage dispatch, actor updates, LoRA weight sync, checkpointing, and Ray
runtime.

## Modules

- `main_erdos.py`: CLI entry point. It prepares the archive, slot parquet, agent
  loop config, then composes verl's trainer config through Hydra overrides. The
  underlying verl entry point and several config keys still use historical
  `ppo` names, but the algorithm is selected by overrides such as
  `algorithm.adv_estimator=grpo`.
- `archive.py`: JSON-backed dynamic Discovery archive. It owns states, PUCT
  counters, group bindings, submitted children, and the best state.
- `data.py`: writes a static slot parquet. Each row is a verl group slot with a
  stable `uid`; it is not the source of TTT state.
- `agent_loop.py`: verl agent loop for Erdos. It binds `global_step:uid` to an
  archive state, asks the model for candidate code, evaluates it, and submits
  children back to the archive.
- `erdos_env.py`: Erdos construction and score utilities.
- `sandbox.py`: small execution helper for generated candidate code.
- `state.py`: typed state helpers used by the archive and agent loop.
- `verl_ext.py`: registers the TTT agent loop, reward function, and
  `ttt_reinforce_is` policy loss with verl.
- `config/`: ready-to-run smoke and scale configs.

## Data Ownership

The slot parquet is static. It gives verl a fixed number of group slots and
stable `uid` values for grouping rollouts and advantages.

`archive.json` is dynamic. It stores Discovery states, PUCT statistics, group
bindings, submitted children, and the current best state. During training, the
agent loop reads and updates this archive; the parquet file is not rewritten.

The mapping is:

```text
verl batch row uid
  -> global_step:uid group key
  -> archive state selected by PUCT
  -> rollout.n candidate children
  -> archive update and selected verl actor update
```

## Prepare Only

```bash
python -m verl_ttt_discover.main_erdos \
  --config verl_ttt_discover/config/erdos_smoke.yaml \
  --prepare-only
```

This writes the run scaffolding:

- `outputs/ttt_erdos/smoke/archive.json`
- `outputs/ttt_erdos/smoke/best_state.json`
- `outputs/ttt_erdos/smoke/ttt_slots.parquet`
- `outputs/ttt_erdos/smoke/agent_loop.yaml`

## Training

For a two-GPU GPT-OSS BF16 smoke:

```bash
GPUS=0,1 \
HF_HOME=/path/to/large/cache/huggingface \
scripts/ttt_discover/run_erdos_gptoss_bf16_2gpu.sh
```

For a local model snapshot:

```bash
GPUS=0,1 \
MODEL_PATH=/path/to/model/snapshot \
scripts/ttt_discover/run_erdos_gptoss_bf16_2gpu.sh
```

For the intended large Erdos run on four B200 GPUs:

```bash
GPUS=0,1,2,3 \
HF_HOME=/path/to/large/cache/huggingface \
scripts/ttt_discover/run_erdos_gptoss_bf16_4gpu_b200.sh
```

This uses `verl_ttt_discover/config/erdos_4gpu_b200_gptoss20b_bf16_official.yaml`:

- `model_path=unsloth/gpt-oss-20b-BF16`
- LoRA rank/alpha 32
- `groups_per_batch=8`, `group_size=64`
- `num_steps=50`, matching official TTT's `num_epochs=50` on the
  single-problem Erdos dataset
- official `phase1_max_tokens=26000`
- 32k rollout context with dynamic per-prompt generation budget
- vLLM tensor parallel size 4
- actor/ref dtype `bf16`
- actor/ref `attn_implementation=flash_attention_2`

The original GPT-OSS model release is not the recommended verl train target
because its quantized/MXFP-style layout does not behave like a normal trainable
BF16 actor/ref checkpoint. Use the Unsloth BF16 conversion for LoRA RL.

The paper also reports the Erdos setting with `Qwen/Qwen3-8B`. Use the same
Docker image and switch only the run config:

```bash
IMAGE_TAG=open-ttt-verl:ttt-vllm \
HF_HOME=/path/to/large/cache/huggingface \
scripts/ttt_discover/docker_run_ttt_vllm.sh run-qwen8b
```

This uses `verl_ttt_discover/config/erdos_4gpu_b200_qwen3_8b_official.yaml`,
with 50 steps, `groups_per_batch=8`, `group_size=64`, LoRA rank/alpha 32,
`max_response_length=26000`, `phase1_max_tokens=26000`, and rollout
`max_model_len=32768`. The Docker build stage does not download the model;
`Qwen/Qwen3-8B` is fetched or loaded from `HF_HOME` during the run.

For the current 8xA800 Qwen GRPO run, keep the 32k context window but use the
26k generation budget:

```bash
script -q -c "IMAGE_TAG=open-ttt-verl:ttt-vllm \
HF_HOME=/path/to/cache \
OUTPUT_DIR=/path/to/output_root \
MODEL_PATH=/path/to/Qwen3-8B \
GPUS=0,1,2,3,4,5,6,7 \
scripts/ttt_discover/docker_run_ttt_vllm.sh run-qwen8b \
run.n_gpus_per_node=8 \
run.tensor_model_parallel_size=8 \
run.max_response_length=26000 \
ttt.phase1_max_tokens=26000 \
ttt.groups_per_batch=8 \
ttt.group_size=32 \
algorithm.adv_estimator=grpo \
actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
actor_rollout_ref.rollout.load_format=auto \
actor_rollout_ref.rollout.max_model_len=32768 \
actor_rollout_ref.rollout.agent.num_workers=32" /dev/null
```

`actor_rollout_ref.rollout.load_format=auto` is required for local Qwen
snapshots; otherwise vLLM can fall back to a dummy load path and produce empty
or corrupt rollouts. In non-interactive SSH/API sessions, wrapping the Docker
launcher with `script -q -c ... /dev/null` avoids TTY handling failures.

If a Blackwell transformers/flash-attn stack errors on the GPT-OSS attention
kernel, keep vLLM rollout enabled and override only actor/ref attention:

```bash
GPUS=0,1,2,3 \
ATTN_IMPL=eager \
scripts/ttt_discover/run_erdos_gptoss_bf16_4gpu_b200.sh \
  actor_rollout_ref.rollout.enforce_eager=True
```

The same entry point accepts final Hydra overrides:

```bash
python -m verl_ttt_discover.main_erdos \
  --config verl_ttt_discover/config/erdos_2gpu_smoke_flash.yaml \
  actor_rollout_ref.model.path=/path/to/local/model \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45
```

## Config Notes

- `groups_per_batch` controls how many archive states are sampled per trainer
  step; it is the TTT batch size, not the total number of generated rollouts.
- `group_size` maps to `actor_rollout_ref.rollout.n`, the number of rollout
  candidates sampled per state. Total rollout responses per trainer step are
  `groups_per_batch * group_size`.
- The original TTT-Discover Erdos shape is `groups_per_batch=8` and
  `group_size=64`; memory-constrained A800 GRPO runs can use
  `group_size=32`.
- The original official run length is `num_epochs=50`. Since the official
  single-problem dataset has one batch per epoch, this verl recipe maps it to
  `run.num_steps=50`.
- On Tinker, the `gpt-oss-120b` context window is limited to 32768 tokens. A
  rollout stops when this window is exhausted or the model emits EOS. For most
  domains, limit the prompt plus thinking budget to 26000 tokens, leaving room
  for the final response, including longer generated algorithm code. In this
  repo that means `run.max_response_length=26000`,
  `ttt.phase1_max_tokens=26000`, and
  `actor_rollout_ref.rollout.max_model_len=32768`.
- The default TTT recipe path uses `algorithm.adv_estimator=entropic_adaptive_beta`
  with `actor_rollout_ref.actor.policy_loss.loss_mode=ttt_reinforce_is`. The
  current Qwen GRPO run overrides this to `algorithm.adv_estimator=grpo` and
  `actor_rollout_ref.actor.policy_loss.loss_mode=vanilla`.
- verl retains `ppo_*` batch-size key names for actor updates even when the
  selected algorithm is GRPO; those names do not imply PPO is being used.
- Best-so-far progress is recorded in the run output directory:
  `best_state.json`, `puct_stats.json`, `archive_snapshots/step_*.json`, and
  `rollout_debug.jsonl`.
- With static batch sizing, `groups_per_batch * group_size` must be divisible
  by the number of FSDP ranks/GPUs.
- `save_freq=-1` is recommended for smoke tests with 20B models to avoid large
  checkpoints.
- `actor_rollout_ref.rollout.checkpoint_engine.backend=naive` is used for
  colocated runs; verl's NCCL checkpoint engine is better suited to
  disaggregated trainer/rollout placement.

## Validation

```bash
pytest -q tests/ttt_discover
```
