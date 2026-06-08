# Open-TTT Slime Docker

This repo runs on top of the Slime image. The Docker wrapper mounts the repo at
`/root/workspace/erdos` and a persistent workspace at `/root/workspace`.

## Build

```bash
docker/build_image.sh
```

Override the image names if needed:

```bash
BASE_IMAGE=slimerl/slime:latest IMAGE=open-ttt-slime:latest docker/build_image.sh
```

## Workspace Layout

By default, `docker/run_open_ttt_slime.sh` stores generated files under
`./workspace`, which is ignored by git:

- models: `workspace/models`
- Megatron checkpoints: `workspace/ckpt`
- logs: `workspace/logs`
- TTT archive and best state: `workspace/data`
- temporary sandbox files: `workspace/tmp`

Use another host path with:

```bash
WORKSPACE_HOST=/data/open-ttt-workspace docker/run_open_ttt_slime.sh
```

## Qwen3-8B 2-GPU Flow

The default packaged path is Qwen3-8B on 2 GPUs with `TP=1`, `PP=2`, LoRA, and
colocated Slime actor/SGLang rollout. Each SGLang engine uses 1 GPU.

Run step by step:

```bash
docker/run_open_ttt_slime.sh scripts/download_qwen3_8b.sh
docker/run_open_ttt_slime.sh scripts/convert_qwen3_8b_in_container.sh
docker/run_open_ttt_slime.sh scripts/run_qwen3_8b_erdos_rl_in_container.sh
```

Or run all three:

```bash
docker/run_open_ttt_slime.sh scripts/docker_qwen3_8b_2gpu_all.sh
```

If Hugging Face auth is required:

```bash
HF_TOKEN=... docker/run_open_ttt_slime.sh scripts/download_qwen3_8b.sh
```

## Important Defaults

- model: `Qwen/Qwen3-8B`
- raw model: `/root/workspace/models/Qwen3-8B`
- converted checkpoint: `/root/workspace/ckpt/Qwen3-8B_torch_dist`
- LoRA output: `/root/workspace/ckpt/Qwen3-8B_erdos_lora_2gpu`
- training steps: `50`
- default rollouts per step: `1 * 8 = 8`
- LoRA rank/alpha: `64`
- max rollout response length: `30000`
- SGLang context length: `16384`
- reasoning effort: `high`

The default batch is conservative for colocated 2-GPU runs. Increase
`ROLLOUT_BATCH_SIZE` and `GLOBAL_BATCH_SIZE` only after confirming the job is
not OOM-limited on the target node.

## Open a Shell

```bash
docker/run_open_ttt_slime.sh
```

Inside the container:

```bash
scripts/download_qwen3_8b.sh
scripts/convert_qwen3_8b_in_container.sh
scripts/run_qwen3_8b_erdos_rl_in_container.sh
```
