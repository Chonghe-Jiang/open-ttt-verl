# Open-TTT Slime Erdos

This repo wires Slime, Megatron-LM, SGLang, and local Erdos task utilities for
TTT-Discover-style RL on the Erdos minimum overlap task.

## Recommended Entry Point: Docker + Qwen3-8B

Build the Docker image:

```bash
docker/build_image.sh
```

Run the 2-GPU Qwen3-8B flow step by step:

```bash
docker/run_open_ttt_slime.sh scripts/download_qwen3_8b.sh
docker/run_open_ttt_slime.sh scripts/convert_qwen3_8b_in_container.sh
docker/run_open_ttt_slime.sh scripts/run_qwen3_8b_erdos_rl_in_container.sh
```

Or run the full 2-GPU sequence:

```bash
docker/run_open_ttt_slime.sh scripts/docker_qwen3_8b_2gpu_all.sh
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
GPUs. Each SGLang engine uses 1 GPU.

## Training Defaults

- model: `Qwen/Qwen3-8B`
- converted checkpoint layout: `TP=1`, `PP=2`
- fine-tuning: LoRA rank/alpha `64`
- training steps: `50`
- default rollouts per step: `2 * 8 = 16`
- max rollout response length: `30000`
- SGLang context length: `16384`
- reasoning effort: `high`
- task/archive/sandbox utilities: `erdos_slime/ttt_discover`

The packaged 2-GPU defaults are intentionally conservative after OOMs on
colocated actor/rollout jobs. Larger batches can be tested by overriding
`ROLLOUT_BATCH_SIZE`, `GLOBAL_BATCH_SIZE`, `LORA_RANK`, and `LORA_ALPHA` at
submission time.

## Core Frameworks

- Slime and Megatron-LM: RL actor training.
- SGLang: rollout generation.
- `erdos_slime`: custom Slime generate/reward/advantage/loss hooks.
- `erdos_slime/ttt_discover`: archive, PUCT state sampling, Erdos
  sandbox/scoring, and TTT task utilities.
