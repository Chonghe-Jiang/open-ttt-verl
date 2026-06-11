# Open-TTT Slime Erdos

This branch wires Slime, Megatron-LM, SGLang, and local TTT-Discover utilities
to reproduce the Erdos minimum-overlap experiment from
`test-time-training/discover` as closely as possible inside the Slime training
stack.

Keep this checkout separate from the original `open-ttt-verl` checkout. The
expected local layout is:

```text
/work/mit/ppliang_mit/lsy/open-ttt-verl
/work/mit/ppliang_mit/lsy/open-ttt-verl-slime-ttt
```

## What This Branch Implements

- `erdos_slime/ttt_slime.py`: the single source of truth for Slime custom
  generation, reward scoring, reward post-processing, token advantage
  broadcast, and custom REINFORCE loss.
- `erdos_slime/erdos_generate.py`, `erdos_slime/erdos_rm.py`, and
  `erdos_slime/entropic_advantage.py`: compatibility wrappers that point back
  to `ttt_slime.py`.
- `erdos_slime/ttt_discover`: the Erdos task implementation, sandbox,
  verifier, persistent PUCT archive, best-state export, and prompt builder.
- Launch scripts: Docker and Apptainer entrypoints that use the unified hooks,
  `custom_loss`, rollout logprobs, entropic leave-one-out advantages, and
  TTT-Discover-style defaults.

The archive writes:

```text
archive.json
best_state.json
puct_stats.json
```

## Important Defaults

- model: `Qwen/Qwen3-8B`
- fine-tuning: LoRA rank/alpha `32`
- base optimizer: Adam, lr `4e-5`, beta1 `0.9`, beta2 `0.99`, eps `1e-8`
- 8-GPU B200 wrapper: beta2 `0.95`
- KL loss coefficient: `0.1`
- reward normalization: disabled
- target C5: `0.38080`
- sandbox timeout: `1000` seconds
- sandbox CPUs: `2`
- reasoning effort: `high`
- importance-sampling clip: `0`, which disables clipping
- entropic target KL: `ln 2`

Override these with environment variables such as `LORA_RANK`,
`LORA_ALPHA`, `LR`, `KL_LOSS_COEF`, `TTT_TARGET_C5`,
`TTT_SANDBOX_TIMEOUT_S`, `TTT_SANDBOX_CPUS`, and
`TTT_ENTROPIC_TARGET_KL`.

## MIT Apptainer Flow

The cluster-local helper defaults are in:

```bash
scripts/lsy_apptainer_env.sh
```

They point to:

```text
REPO_DIR=/work/mit/ppliang_mit/lsy/open-ttt-verl-slime-ttt
WORKSPACE=/work/mit/ppliang_mit/lsy/open-ttt-workspace
SIF_PATH=/work/mit/ppliang_mit/lsy/open-ttt-workspace/images/slime_latest.sif
```

Build or inspect the Slime image:

```bash
sbatch scripts/lsy_build_slime_sif.slurm
bash scripts/inspect_slime_container.sh
```

Prepare Qwen3-8B and convert it to Megatron torch-dist:

```bash
sbatch scripts/lsy_download_convert_qwen3_8b_2gpu.slurm
```

Run the 2-GPU colocated debug/training profile:

```bash
sbatch scripts/lsy_run_qwen3_8b_erdos_rl_2gpu.slurm
```

Generic Apptainer scripts also use script-relative `REPO_DIR` defaults, so they
can be launched from this checkout without accidentally using a different repo:

```bash
sbatch scripts/download_convert_qwen3_8b_2gpu.slurm
sbatch scripts/run_qwen3_8b_erdos_rl_2gpu.slurm
sbatch scripts/run_qwen3_8b_erdos_rl_4gpu.slurm
```

Use the 4-GPU non-colocated script if the 2-GPU colocated job OOMs during actor
or rollout startup.

## Docker Flow

See [docker/README.md](docker/README.md) for Docker details. The wrapper mounts
this repo at `/root/workspace/erdos` and a persistent workspace at
`/root/workspace`.

Build the image:

```bash
docker/build_image.sh
```

Prepare Qwen3-8B on a clean workspace:

```bash
docker/run_open_ttt_slime.sh scripts/download_qwen3_8b.sh
docker/run_open_ttt_slime.sh scripts/convert_qwen3_8b_in_container.sh
```

Run smoke, one-step, and full 8-GPU B200 flows:

```bash
docker/smoke_qwen3_8b_8b200.sh
docker/run_qwen3_8b_8b200_one_step.sh
docker/run_qwen3_8b_8b200_rl.sh
```

For 8xA800 80GB machines, use:

```bash
docker/run_qwen3_8b_a800_rl.sh
```

## Training Hooks

All main launch scripts pass these Slime custom hooks:

```text
--custom-generate-function-path erdos_slime.ttt_slime.generate
--custom-rm-path erdos_slime.ttt_slime.reward
--custom-reward-post-process-path erdos_slime.ttt_slime.ttt_reward_post_process
--custom-advantage-function-path erdos_slime.ttt_slime.ttt_advantages
--loss-type custom_loss
--custom-loss-function-path erdos_slime.ttt_slime.ttt_reinforce_loss
```

Generation acquires one archive parent per prompt group. Reward execution runs
the generated Python in the sandbox, verifies `(h_values, c5_bound, n_points)`,
submits valid children back to the archive, and records raw C5 scores in sample
metadata. Reward post-processing converts per-group scores into entropic
leave-one-out advantages for REINFORCE training.

## GPT-OSS-20B

The GPT-OSS-20B path uses the same hooks and workspace layout:

```text
/root/workspace/models/gpt-oss-20b
/root/workspace/models/gpt-oss-20b-bf16
/root/workspace/ckpt/gpt-oss-20b_torch_dist
```

Download and convert:

```bash
docker/run_open_ttt_slime.sh scripts/download_gpt_oss_20b.sh
docker/run_open_ttt_slime.sh scripts/convert_gpt_oss_20b_in_container.sh
```

Run one training-step validation:

```bash
docker/run_gpt_oss_20b_one_step.sh
```

## Validation

Useful local checks:

```bash
python -m py_compile \
  erdos_slime/ttt_slime.py \
  erdos_slime/erdos_generate.py \
  erdos_slime/erdos_rm.py \
  erdos_slime/entropic_advantage.py \
  erdos_slime/ttt_discover/erdos_env.py \
  erdos_slime/ttt_discover/archive.py \
  erdos_slime/ttt_discover/sandbox.py \
  tests/test_ttt_discover_erdos.py

bash -n run_qwen3_8b_erdos.sh run_gpt_oss_20b_erdos.sh scripts/*.sh scripts/*.slurm
```

Inside the Apptainer image, the non-torch test subset can be run with:

```bash
apptainer exec \
  --bind /work/mit/ppliang_mit/lsy:/work/mit/ppliang_mit/lsy \
  /work/mit/ppliang_mit/lsy/open-ttt-workspace/images/slime_latest.sif \
  /usr/bin/bash -lc 'cd /work/mit/ppliang_mit/lsy/open-ttt-verl-slime-ttt && PYTHONPATH=$PWD:$PWD/slime /usr/bin/python3 -m pytest tests/test_ttt_discover_erdos.py -k "not ttt_advantages" -q'
```

The final `ttt_advantages` test imports torch and can be slow or memory-heavy
depending on the container and node allocation.
