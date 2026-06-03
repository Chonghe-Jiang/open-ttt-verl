# TTT CI

This fork intentionally keeps GitHub Actions focused on the TTT-Discover
recipe. Upstream verl workflows for full CPU/GPU/NPU, Ascend, Megatron,
SGLang, vLLM, Docker, documentation, and scorecard coverage are not run here by
default.

The default workflow is `ttt-ci.yml`. It checks:

- TTT package compilation.
- TTT license headers.
- CPU-safe TTT unit tests that do not require vLLM, FlashAttention, or GPU
  runtime packages.
- `--prepare-only` for the official 4xB200 GPT-OSS BF16 Erdos config.

Large GPU training is validated through explicit smoke/experiment runs, not
GitHub-hosted CI.
