# TTT-Discover on verl

This package implements the TTT-Discover/Erdos recipe while delegating the heavy
RL execution to verl: rollout, logprobs, rollout-importance correction, batching,
advantage dispatch, actor updates, LoRA weight sync, checkpointing, and Ray
runtime.

## Modules

- `main_erdos.py`: CLI entry point. It prepares the archive, slot parquet, agent
  loop config, then composes verl's PPO trainer config through Hydra overrides.
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
  -> archive update and PPO/REINFORCE training batch
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

This uses `verl_ttt_discover/config/erdos_4gpu_b200_gptoss20b_bf16_16k.yaml`:

- `model_path=unsloth/gpt-oss-20b-BF16`
- LoRA rank/alpha 32
- `groups_per_batch=8`, `group_size=64`
- 16k rollout context, split as 8192 prompt and 8192 response tokens
- vLLM tensor parallel size 4
- actor/ref dtype `bf16`
- actor/ref `attn_implementation=flash_attention_2`

The original GPT-OSS model release is not the recommended verl train target
because its quantized/MXFP-style layout does not behave like a normal trainable
BF16 actor/ref checkpoint. Use the Unsloth BF16 conversion for LoRA RL.

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
  step.
- `group_size` maps to `actor_rollout_ref.rollout.n`, the number of rollout
  candidates sampled per state.
- The original TTT-Discover Erdos shape is `groups_per_batch=8` and
  `group_size=64`.
- On 2-GPU FSDP with static batch sizing, `groups_per_batch * group_size` must
  be divisible by 2.
- `save_freq=-1` is recommended for smoke tests with 20B models to avoid large
  checkpoints.
- `actor_rollout_ref.rollout.checkpoint_engine.backend=naive` is used for
  colocated runs; verl's NCCL checkpoint engine is better suited to
  disaggregated trainer/rollout placement.

## Validation

```bash
pytest -q tests/ttt_discover
```
