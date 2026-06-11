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

## Qwen3-8B 8-GPU B200 Flow

The recommended path is Qwen3-8B on 8 B200 GPUs with `TP=1`, `PP=2`, LoRA, 4
actor GPUs, and 4 rollout GPUs. Each SGLang rollout engine uses 1 GPU.

The default in-container paths are:

```text
/root/workspace/models/Qwen3-8B
/root/workspace/ckpt/Qwen3-8B_torch_dist
```

If those already exist in the mounted workspace, skip directly to training.
On a clean workspace, download and convert first:

```bash
docker/run_open_ttt_slime.sh scripts/download_qwen3_8b.sh
docker/run_open_ttt_slime.sh scripts/convert_qwen3_8b_in_container.sh
```

Then run the 8-GPU smoke test:

```bash
docker/smoke_qwen3_8b_8b200.sh
```

Then verify one real training update with small batch/group defaults and 30k max
response tokens:

```bash
docker/run_qwen3_8b_8b200_one_step.sh
```

After smoke passes, start the full 50-step run:

```bash
docker/run_qwen3_8b_8b200_rl.sh
```

On 8xA800 80GB machines, start the lighter 50-step run:

```bash
docker/run_qwen3_8b_a800_rl.sh
```

The A800 profile keeps the same model, optimizer, LoRA, KL, actor/rollout GPU
split, and 30k maximum response length, but uses actor `TP=2`, `PP=2`,
`ROLLOUT_BATCH_SIZE=1`, `N_SAMPLES_PER_PROMPT=4`, and `GLOBAL_BATCH_SIZE=4`.

The legacy 2-GPU scripts remain available for low-resource debugging, but they
are not the paper-aligned configuration.

## Qwen3-8B 8×B200 Smoke

After the model has been downloaded and converted in the mounted workspace, run:

```bash
docker/smoke_qwen3_8b_8b200.sh
```

The smoke command builds the image by default, checks that Docker exposes 8 CUDA
devices, checks `/root/workspace/models/Qwen3-8B` and
`/root/workspace/ckpt/Qwen3-8B_torch_dist`, starts Ray/SGLang/Megatron, applies
LoRA, syncs actor weights to rollout, and exits before rollout generation.

If the image is already built:

```bash
BUILD_IMAGE=0 docker/smoke_qwen3_8b_8b200.sh
```

## Qwen3-8B 8×B200 Paper-Aligned RL

After smoke passes, run the full training entrypoint:

```bash
docker/run_qwen3_8b_8b200_rl.sh
```

Both smoke and full training use a non-colocated Docker run with 4 actor GPUs
and 4 rollout GPUs.
It uses 50 training steps, 512 rollouts per step as 8 groups × 64 rollouts,
LoRA rank/alpha 32, Adam lr `4e-5`, β1 `0.9`, β2 `0.95`, ε `1e-8`, KL
coefficient `0.1`, 30k maximum rollout response tokens, PUCT reuse, and
entropic target KL `ln 2`.

## Qwen3-8B 8×A800 RL

For 8 A800 80GB GPUs:

```bash
docker/run_qwen3_8b_a800_rl.sh
```

Defaults: 50 training steps, 4 actor GPUs as `TP=2, PP=2`, 4 rollout GPUs,
1 rollout group, 4 rollouts per group, global batch size 4, micro batch size 1,
and 30k maximum rollout response tokens.

If Hugging Face auth is required:

```bash
HF_TOKEN=... docker/run_open_ttt_slime.sh scripts/download_qwen3_8b.sh
```

## Important Defaults

- model: `Qwen/Qwen3-8B`
- raw model: `/root/workspace/models/Qwen3-8B`
- converted checkpoint: `/root/workspace/ckpt/Qwen3-8B_torch_dist`
- LoRA output: `/root/workspace/ckpt/Qwen3-8B_erdos_lora_8b200`
- training steps: `50`
- actor GPUs: `4`
- rollout GPUs: `4`
- rollout groups per step: `8`
- rollouts per group: `64`
- global batch size: `512`
- micro batch size: `1`
- LoRA rank/alpha: `32`
- max rollout response length: `30000`
- SGLang context length: `32768`
- max tokens per GPU: `8192`
- Megatron-to-HF weight conversion mode: `bridge`
- initial actor-to-rollout weight sync timeout: `900` seconds
- reasoning effort: `high`

The 8-GPU B200 scripts intentionally keep the 8B200 rollout/group/batch
parameters fixed. Override them only for debugging non-paper-aligned runs.

## GPT-OSS-20B One-Step Flow

The 20B path uses:

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

Run a one-step training validation:

```bash
docker/run_gpt_oss_20b_one_step.sh
```

Defaults: `NUM_ROLLOUT=1`, `ROLLOUT_BATCH_SIZE=1`,
`N_SAMPLES_PER_PROMPT=2`, `GLOBAL_BATCH_SIZE=2`,
`ROLLOUT_MAX_RESPONSE_LEN=30000`.

## Open a Shell

```bash
docker/run_open_ttt_slime.sh
```

Inside the container:

```bash
scripts/download_qwen3_8b.sh
scripts/convert_qwen3_8b_in_container.sh
scripts/convert_gpt_oss_20b_in_container.sh
scripts/docker_qwen3_8b_8b200_one_step.sh
scripts/docker_qwen3_8b_8b200_smoke.sh
scripts/docker_qwen3_8b_8b200_rl.sh
```
