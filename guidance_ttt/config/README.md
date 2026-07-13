# Guidance TTT Configs

The top-level config directory keeps the current Polyomino Packing Modal
recipes and seed utility. The recommended training config is
`polyomino_modal_h200_3gpu_gpt_oss_120b_batch8_group16_code_delta_8192_50step.yaml`;
its one-step acceptance counterpart is the matching `_smoke.yaml` file.

- `polyomino_modal_h200_3gpu_gpt_oss_120b_group16_h200_tuned.yaml`
- `polyomino_modal_h200_3gpu_gpt_oss_120b_batch8_group8_temp09.yaml`
- `polyomino_modal_h200_3gpu_gpt_oss_120b_batch8_group16_concurrency16_1step.yaml`
- `polyomino_modal_h200_3gpu_gpt_oss_120b_batch8_group16_puct_fix_1step.yaml`
- `polyomino_modal_h200_3gpu_gpt_oss_120b_batch8_group16_concurrency16_50step.yaml`
- `polyomino_modal_h200_3gpu_gpt_oss_120b_batch8_group16_prompt_refinement_50step.yaml`
- `polyomino_modal_h200_3gpu_gpt_oss_120b_batch8_group16_code_delta_8192_smoke.yaml`
- `polyomino_modal_h200_3gpu_gpt_oss_120b_batch8_group16_code_delta_8192_50step.yaml`
- `polyomino_modal_h200_gpt_oss_120b_bootstrap_seed.yaml`

Older smoke, debug, and historical experiment recipes are preserved under
`backup/`. Existing scripts that still target those recipes should point to the
backup path explicitly.
