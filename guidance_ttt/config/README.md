# Guidance TTT Configs

The top-level config directory contains the current runnable Polyomino
recipes, including guidance→execution, direct-discover, tiny smoke, and
inference-only workflows.

All `polyomino_packing` runs use the guidance-ttt Discover-compatible search
profile: per-rollout PUCT accounting, `best_child` Q, `puct_c: 1.0`, a
1000-node buffer, and top-2 children. Formal recipes use 8 groups of 16
rollouts; tiny/smoke recipes may reduce the geometry without changing those
semantics.

Older smoke, debug, and historical experiment recipes are preserved under
`backup/`. Existing scripts that still target those recipes should point to the
backup path explicitly. Their YAML is retained for provenance, but the shared
Polyomino runtime still enforces the Discover-compatible profile when they are
launched. Libraries created with the former group-level accounting are rejected
instead of migrated; use a new output directory.
