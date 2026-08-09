# Documentation index

## Current implementation

- `current_pipeline.md`: end-to-end Guidance-TTT data flow.
- `guidance_ttt_prompts.md`: guidance and execution prompt contracts.
- `model_prompt_summary.md`: model-facing prompt overview.
- `polyomino_packing.md`: FrontierCS Polyomino Packing task integration.
- `vliw_kernel_optimization.md`: EdgeBench VLIW task, official verifier,
  reward semantics, and Modal 2xH200 smoke workflow.
- `trimul.md`: TTT-Discover TriMul prompt/evaluator provenance, H100 metric,
  reward mapping, Modal setup, and one-step acceptance workflow.

## B200 experiments

- `b200_guidance_four_run_results_2026-07-14.md`: preserved baseline snapshot
  for the stopped Qwen3-8B/Qwen3-14B, code-delta/summary-only runs.

Raw libraries, checkpoints, Slurm logs, and generated summaries live under the
ignored `outputs/` tree and are intentionally not committed.

## Modal experiment artifacts

Files prefixed with `polyomino_modal_`, `modal_`, or `rollout_` are retained
prompt/answer dumps and analyses from earlier Modal experiments. JSON files are
machine-readable companions to the corresponding Markdown reports.
