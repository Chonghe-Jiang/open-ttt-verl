"""Yuchen-aligned final report.

Compares:
- Yuchen's reported base avg/max + STaR avg/max (hardcoded from her report)
- Our base eval (iterative tool calling, max_turns=4, max_tokens=4096, temp=0.9)
- Our TTT-Discover trained on 1 task, eval'd zero-shot on 18 others

Outputs Yuchen-style table with ID/OOD split + per-rollout reward distribution.
"""
import argparse
import json
from pathlib import Path


YUCHEN_DISPLAY = {
    "cbl__high_av_loose_dl_small_oh": "cbl__high_av_loose_dl_small",
    "cbl__low_av_tight_dl_large_oh": "cbl__low_av_tight_dl_large",
    "cbl__mixed_av_loose_dl_large_oh": "cbl__mixed_av_loose_dl_large",
    "cbl_multi__high_av_loose_dl_small_oh": "cbl_multi__high_av_loose_dl_small",
    "cbl_multi__high_av_tight_dl_small_oh": "cbl_multi__high_av_tight_dl_small",
    "cbl_multi__low_av_loose_dl_small_oh": "cbl_multi__low_av_loose_dl_small",
    "fused_linear_ce": "fused_linear_ce",
    "gemm_opt__annoying": "gemm_opt__annoying",
    "gemm_opt__k_skewed": "gemm_opt__k_skewed",
    "gemm_opt__rectangles": "gemm_opt__rectangles",
    "gemm_opt__squares": "gemm_opt__squares",
    "gemm_opt__transformerish": "gemm_opt__transformerish",
    "llm_sql__large": "llm_sql__large",
    "poc_gen__heap_uaf": "poc_gen__heap_uaf",
    "poc_gen__uninit_value": "poc_gen__uninit_value",
    "vdb_pareto__low_latency": "vdb_pareto__low_latency",
    "vdb_pareto__recall80_lat": "vdb_pareto__recall80_lat",
}

ID_TASKS = [
    "cbl__high_av_loose_dl_small_oh",
    "cbl__low_av_tight_dl_large_oh",
    "cbl__mixed_av_loose_dl_large_oh",
    "cbl_multi__high_av_loose_dl_small_oh",
    "cbl_multi__high_av_tight_dl_small_oh",
    "cbl_multi__low_av_loose_dl_small_oh",
]

OOD_TASKS = [
    "fused_linear_ce",
    "gemm_opt__annoying",
    "gemm_opt__k_skewed",
    "gemm_opt__rectangles",
    "gemm_opt__squares",
    "gemm_opt__transformerish",
    "llm_sql__large",
    "poc_gen__heap_uaf",
    "poc_gen__uninit_value",
    "vdb_pareto__low_latency",
    "vdb_pareto__recall80_lat",
]

# Yuchen's reported numbers
YUCHEN_BASE = {
    "cbl__high_av_loose_dl_small_oh":      (0.1250, 0.2500),
    "cbl__low_av_tight_dl_large_oh":       (0.0488, 0.3900),
    "cbl__mixed_av_loose_dl_large_oh":     (0.1400, 0.3800),
    "cbl_multi__high_av_loose_dl_small_oh":(0.0681, 0.2725),
    "cbl_multi__high_av_tight_dl_small_oh":(0.0681, 0.2725),
    "cbl_multi__low_av_loose_dl_small_oh": (0.4392, 0.8785),
}
YUCHEN_STAR = {
    "cbl__high_av_loose_dl_small_oh":      (0.0417, 0.2500),
    "cbl__low_av_tight_dl_large_oh":       (0.2050, 0.4000),
    "cbl__mixed_av_loose_dl_large_oh":     (0.2375, 0.3800),
    "cbl_multi__high_av_loose_dl_small_oh":(0.2637, 0.8435),
    "cbl_multi__high_av_tight_dl_small_oh":(0.0822, 0.2725),
    "cbl_multi__low_av_loose_dl_small_oh": (0.3294, 0.8785),
}


def collect(eval_dir: Path, task: str) -> list[float] | None:
    rs = []
    for f in sorted(eval_dir.glob(f"{task}.r*.eval.json")):
        try:
            j = json.loads(f.read_text())
            rs.append(max(0.0, j.get("reward_01", 0)))
        except Exception:
            continue
    return rs if rs else None


def fmt_dist(rs):
    if not rs:
        return "(no data)"
    sr = sorted(rs)
    parts = []
    for r in sr:
        if r == 0:
            parts.append("0")
        else:
            s = f"{r:.2f}".rstrip("0").rstrip(".")
            if s.startswith("0."):
                s = s[1:]
            parts.append(s)
    return " ".join(parts)


def avg_max(rs):
    if not rs:
        return None, None
    return sum(rs)/len(rs), max(rs)


TRAIN_TASKS_4 = {
    "cbl_multi__low_av_loose_dl_small_oh",
    "cbl__mixed_av_loose_dl_large_oh",
    "gemm_opt__transformerish",
    "cbl_multi__high_av_tight_dl_small_oh",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--our-base-dir", default="/fsx/xuanj/ttt-discover/results/base_iter/eval")
    ap.add_argument("--our-ttt-dir", default="/fsx/xuanj/ttt-discover/results/ttt_iter_post_eval/eval")
    ap.add_argument("--output", default="/fsx/xuanj/ttt-discover/REPORT_YUCHEN_ALIGNED.md")
    ap.add_argument("--train-task", default="4tasks_yuchen_seed1",
                    help="Display string for the train task config")
    args = ap.parse_args()
    base = Path(args.our_base_dir)
    ttt = Path(args.our_ttt_dir)

    lines = []
    lines.append("# Frontier-CS — Yuchen-aligned comparison (Base vs STaR vs TTT-Discover)")
    lines.append("")
    lines.append("All three approaches use the **same protocol**:")
    lines.append("- Base model: DeepSeek-R1-0528-Qwen3-8B")
    lines.append("- Iterative tool calling rollout, 3 turns × 16384 tokens, temperature 0.9")
    lines.append("- Prompt includes starter code (initial_greedy.py / seed solution)")
    lines.append("- 8 rollouts/task, score = max across turns of each rollout")
    lines.append("")
    lines.append("**TTT-Discover setup**: trained on **4 tasks** matching Yuchen's STaR (kelp PR#11 `iterations=1, batch_size=4, seed=1` from full 66-task pool), then evaluated on all 17 ID/OOD tasks (4 train + 13 generalization).")
    lines.append("")
    lines.append("Train tasks (matching `random.Random(1).sample(yuchen_pool_66, 4)`):")
    for t in sorted(TRAIN_TASKS_4):
        lines.append(f"  - `{t}` (TRAIN)")
    lines.append("")

    for section, task_list, is_id in [
        ("In-Distribution Tasks (6)", ID_TASKS, True),
        ("Out-of-Distribution Tasks (11)", OOD_TASKS, False),
    ]:
        lines.append(f"### {section}")
        lines.append("")
        lines.append("| Task | Yuchen base avg/max | Yuchen STaR avg/max | Our base avg/max | Our TTT avg/max | Δ TTT vs base |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        sums = {"yb_a": 0, "yb_m": 0, "ys_a": 0, "ys_m": 0,
                "ob_a": 0, "ob_m": 0, "ot_a": 0, "ot_m": 0,
                "ob_n": 0, "ot_n": 0, "yb_n": 0, "ys_n": 0}
        sub_rows = []
        for t in task_list:
            display = YUCHEN_DISPLAY.get(t, t)
            yb = YUCHEN_BASE.get(t)
            ys = YUCHEN_STAR.get(t)
            our_b = collect(base, t)
            our_t = collect(ttt, t)
            ob_a, ob_m = avg_max(our_b)
            ot_a, ot_m = avg_max(our_t)
            yb_str = f"{yb[0]:.4f}/{yb[1]:.4f}" if yb else "n/a"
            ys_str = f"{ys[0]:.4f}/{ys[1]:.4f}" if ys else "n/a"
            ob_str = f"{ob_a:.4f}/{ob_m:.4f}" if ob_a is not None else "(running)"
            ot_str = f"{ot_a:.4f}/{ot_m:.4f}" if ot_a is not None else "(pending)"
            if ob_a is not None and ot_a is not None:
                d = ot_a - ob_a
                d_str = f"**{d:+.4f}**" if d > 0.001 else f"{d:+.4f}"
            else:
                d_str = "(pending)"
            train_marker = " (TRAIN)" if t in TRAIN_TASKS_4 else ""
            lines.append(f"| {display}{train_marker} | {yb_str} | {ys_str} | {ob_str} | {ot_str} | {d_str} |")
            if yb: sums["yb_a"] += yb[0]; sums["yb_m"] = max(sums["yb_m"], yb[1]); sums["yb_n"] += 1
            if ys: sums["ys_a"] += ys[0]; sums["ys_m"] = max(sums["ys_m"], ys[1]); sums["ys_n"] += 1
            if ob_a is not None:
                sums["ob_a"] += ob_a; sums["ob_m"] = max(sums["ob_m"], ob_m); sums["ob_n"] += 1
            if ot_a is not None:
                sums["ot_a"] += ot_a; sums["ot_m"] = max(sums["ot_m"], ot_m); sums["ot_n"] += 1
            sub_rows.append((display, yb, ys, our_b, our_t))

        # Total row
        def avg(s, n): return s/n if n else 0
        lines.append(f"| **TOTAL** | "
                     f"**{avg(sums['yb_a'],sums['yb_n']):.4f}/{sums['yb_m']:.4f}** | "
                     f"**{avg(sums['ys_a'],sums['ys_n']):.4f}/{sums['ys_m']:.4f}** | "
                     f"**{avg(sums['ob_a'],sums['ob_n']):.4f}/{sums['ob_m']:.4f}** | "
                     f"**{avg(sums['ot_a'],sums['ot_n']):.4f}/{sums['ot_m']:.4f}** | "
                     f"**{avg(sums['ot_a'],sums['ot_n']) - avg(sums['ob_a'],sums['ob_n']):+.4f}** |")
        lines.append("")
        lines.append(f"#### Per-rollout reward distributions ({section.split()[0]})")
        lines.append("")
        lines.append("| Task | Our base | Our TTT-Discover (zero-shot) |")
        lines.append("| --- | --- | --- |")
        for display, yb, ys, ob, ot in sub_rows:
            lines.append(f"| {display} | {fmt_dist(ob)} | {fmt_dist(ot)} |")
        lines.append("")

    Path(args.output).write_text("\n".join(lines))
    print(f"[report] wrote {args.output}")


if __name__ == "__main__":
    main()
