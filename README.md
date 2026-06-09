# Open-TTT Slime Erdos

This repo wires Slime, Megatron-LM, SGLang, and local Erdos task utilities for
TTT-Discover-style RL on the Erdos minimum overlap task.

## Recommended Entry Point: 8-GPU Docker + Qwen3-8B

Build the Docker image:

```bash
docker/build_image.sh
```

If the workspace already contains the raw model and converted checkpoint, these
paths are used directly:

```text
/root/workspace/models/Qwen3-8B
/root/workspace/ckpt/Qwen3-8B_torch_dist
```

On a clean workspace, prepare them with:

```bash
docker/run_open_ttt_slime.sh scripts/download_qwen3_8b.sh
docker/run_open_ttt_slime.sh scripts/convert_qwen3_8b_in_container.sh
```

Then run the 8-GPU B200 smoke test. This verifies model init and actor-to-rollout
weight sync, but exits before rollout generation:

```bash
docker/smoke_qwen3_8b_8b200.sh
```

To verify a real training update with a small batch/group while keeping 30k max
output tokens:

```bash
docker/run_qwen3_8b_8b200_one_step.sh
```

After smoke and one-step pass, run the full 8-GPU B200 training flow:

```bash
docker/run_qwen3_8b_8b200_rl.sh
```

By default, generated models/checkpoints/logs are stored under `./workspace`,
which is ignored by git.

## Docker

See [docker/README.md](docker/README.md). The launcher mounts this repo at
`/root/workspace/erdos` and a persistent workspace at `/root/workspace`.

## Slurm / Apptainer on Engaging

For the current cluster setup:

```bash
sbatch scripts/download_convert_qwen3_8b_2gpu.slurm
sbatch scripts/run_qwen3_8b_erdos_rl_2gpu.slurm
```

The 2-GPU RL script uses colocated Slime actor and SGLang rollout on H200-class
GPUs. If this colocated job is OOM-killed during actor initialization, use the
4-GPU non-colocated layout:

```bash
sbatch scripts/run_qwen3_8b_erdos_rl_4gpu.slurm
```

The 4-GPU script keeps the Megatron actor on 2 GPUs and runs two dedicated
SGLang rollout engines on the remaining GPUs, avoiding colocate offload pressure.

For the paper-aligned 8×B200 Docker smoke test:

```bash
docker/smoke_qwen3_8b_8b200.sh
```

The smoke test builds the image by default, checks 8 visible CUDA devices, checks
the raw model and converted checkpoint paths, starts Ray/SGLang/Megatron, applies
LoRA, syncs actor weights to rollout, and exits before rollout generation.

For the full paper-aligned 8×B200 Docker run:

```bash
docker/run_qwen3_8b_8b200_rl.sh
```

This uses 8 GPUs, 50 training steps, 512 rollouts per step as 8 groups × 64 rollouts,
LoRA rank/alpha 32, Adam lr `4e-5`, β1 `0.9`, β2 `0.95`, ε `1e-8`, KL
coefficient `0.01`, 30k maximum rollout response tokens, PUCT reuse, and
entropic target KL `ln 2`.

## 8-GPU B200 Training Defaults

- model: `Qwen/Qwen3-8B`
- raw model: `/root/workspace/models/Qwen3-8B`
- converted checkpoint: `/root/workspace/ckpt/Qwen3-8B_torch_dist`
- converted checkpoint layout: `TP=1`, `PP=2`
- actor GPUs: `4`
- rollout GPUs: `4`
- rollout groups per step: `8`
- rollouts per group: `64`
- global batch size: `512`
- micro batch size: `1`
- fine-tuning: LoRA rank/alpha `32`
- training steps: `50`
- max rollout response length: `30000`
- SGLang context length: `32768`
- max tokens per GPU: `8192`
- Megatron-to-HF weight conversion mode: `bridge`
- initial actor-to-rollout weight sync timeout: `900` seconds
- reasoning effort: `high`
- task/archive/sandbox utilities: `erdos_slime/ttt_discover`

The packaged 2-GPU scripts are kept only as low-resource smoke/debug helpers.
For paper-aligned Erdos TTT-Discover runs, use the 8-GPU B200 scripts above.

## GPT-OSS-20B

The 20B path uses the same workspace layout:

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

Run one training step with small batch/group defaults:

```bash
docker/run_gpt_oss_20b_one_step.sh
```

The 20B one-step defaults are `NUM_ROLLOUT=1`, `ROLLOUT_BATCH_SIZE=1`,
`N_SAMPLES_PER_PROMPT=2`, `GLOBAL_BATCH_SIZE=2`, and
`ROLLOUT_MAX_RESPONSE_LEN=30000`. Override those environment variables when a
larger 20B run is needed.

## Core Frameworks

- Slime and Megatron-LM: RL actor training.
- SGLang: rollout generation.
- `erdos_slime`: custom Slime generate/reward/advantage/loss hooks.
- `erdos_slime/ttt_discover`: archive, PUCT state sampling, Erdos
  sandbox/scoring, and TTT task utilities.
