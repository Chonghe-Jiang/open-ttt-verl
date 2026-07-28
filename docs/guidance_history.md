# Guidance-TTT History Archive

The private history archive is hosted as a Hugging Face dataset:

<https://huggingface.co/datasets/LeoJiangOR/guidance-ttt-history>

The current snapshot contains 33 meaningful runs (22.4 GB) across GLM-5.2,
GPT-OSS-120B, Qwen3/Qwen3.6, Evolvent GPT-5.4, and related experiment
families. It includes history JSON files, associated Slurm logs, and the source
configs and launch scripts needed to identify each run.

Checkpoints, model weights, caches, credentials, preflight/setup artifacts, and
short smoke runs are excluded. A run is retained when it reached at least 10
timesteps or when its name identifies a substantive formal run such as
`1day`, `two_day`, `50step`, or `discover`.

## Layout

```text
manifest.json
runs/<family>/<run>/
  history/
  logs/
source/
  configs/
  scripts/
```

`manifest.json` records the archive selection reason, inferred experiment
family, maximum timestep, source path, byte size, and SHA-256 digest for every
payload file. The verified Hugging Face snapshot commit is
`2c5f9cef4d1159cd497b443b2b218493c5e686c9`.

## Rebuild

Use the deterministic archiver from the repository root:

```bash
python scripts/archive_guidance_history.py \
  --repo-root . \
  --staging-dir .tmp/guidance_history_hf
```

Review the generated manifest before publishing a new private dataset
revision. Upload credentials must remain outside the repository and archive.
