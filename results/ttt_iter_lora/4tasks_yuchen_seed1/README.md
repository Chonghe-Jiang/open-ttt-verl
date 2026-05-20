# TTT-Discover LoRA checkpoints (Yuchen-aligned, 4 train tasks)

Trained on 4 tasks via random.Random(1).sample(yuchen_pool_66, 4):
- cant_be_late_multi__low_availability_loose_deadline_small_overhead
- cant_be_late__mixed_availability_loose_deadline_large_overhead
- gemm_optimization__transformerish
- cant_be_late_multi__high_availability_tight_deadline_small_overhead

Training run cancelled at step 4/5 due to cluster preemption.
Best fully-trained ckpt: **step_003**.

Per-step training metrics (from job log):

| Step | avg | max | best_so_far | loss | n_valid |
|------|-----|-----|------------|------|---------|
| 1    | 0.1748 | 0.6386 | 0.6386 | 0.1318 | 7 |
| 2    | 0.2254 | 0.8785 | 0.8785 | -0.0241 | 8 |
| 3    | 0.1573 | 0.8785 | 0.8785 | 0.0581 | 8 |

Note: LoRA weight files (adapter_model.safetensors, ~118MB each) are
excluded from git. They live at /fsx/xuanj/ttt-discover/results/ttt_iter_lora/4tasks_yuchen_seed1/
on the p5en cluster.

Adapter configs (adapter_config.json) are kept for reproducibility.
