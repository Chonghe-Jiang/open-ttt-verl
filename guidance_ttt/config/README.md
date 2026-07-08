# Guidance TTT Configs

The top-level config directory keeps only the current recommended Polyomino
Packing Modal recipe:

- `polyomino_modal_h200_3gpu_gpt_oss_120b_group16_h200_tuned.yaml`

Older smoke, debug, and historical experiment recipes are preserved under
`backup/`. Existing scripts that still target those recipes should point to the
backup path explicitly.
