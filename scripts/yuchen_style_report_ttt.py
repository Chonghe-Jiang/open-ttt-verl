"""
Render Yuchen-style report comparing Base vs TTT-Discover (faithful port).

Each task: report best step's 8 rollouts as the "TTT-Discover rollouts" column,
matching paper's argmax_i protocol (line 12 of Algorithm 1).
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


def collect_base_rewards(eval_dir: Path, task: str):
    rewards = []
    for f in sorted(eval_dir.glob(f"{task}.r*.eval.json")):
        try:
            j = json.loads(f.read_text())
            rewards.append(max(0.0, j.get("reward_01", 0)))
        except Exception:
            continue
    return rewards if rewards else None


def collect_ttt_best_step_rewards(ttt_dir: Path, task: str):
    """For TTT-Discover: pick the step with max avg reward across the run, return its 8 rollouts."""
    final = ttt_dir / task / "final.json"
    history_p = ttt_dir / task / "history.json"
    if not final.exists() and not history_p.exists():
        return None
    try:
        if final.exists():
            j = json.loads(final.read_text())
            history = j.get("history", [])
        else:
            history = json.loads(history_p.read_text())
    except Exception:
        return None
    if not history:
        return None
    # paper's argmax_i protocol: pick step with highest max reward in its rollouts
    # (ties broken by avg reward, then step idx)
    best_step = max(history, key=lambda h: (h.get("max", 0), h.get("avg", 0), h.get("step", 0)))
    rewards = [max(0.0, r) for r in best_step.get("rewards", [])]
    if len(rewards) < 8:
        rewards += [0.0] * (8 - len(rewards))
    return rewards[:8]


def avg_max(rs):
    if not rs:
        return None, None
    return sum(rs) / len(rs), max(rs)


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
                s = s[1:]  # ".25" instead of "0.25"
            parts.append(s)
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-eval-dir", default="/fsx/xuanj/ttt-discover/results/base/eval")
    ap.add_argument("--ttt-dir", default="/fsx/xuanj/ttt-discover/results/ttt_faithful")
    ap.add_argument("--output", default="/fsx/xuanj/ttt-discover/REPORT_YUCHEN_STYLE.md")
    args = ap.parse_args()
    base_dir = Path(args.base_eval_dir)
    ttt_dir = Path(args.ttt_dir)

    lines = []
    lines.append("# Frontier-CS benchmarking — DeepSeek-R1-0528-Qwen3-8B + TTT-Discover (faithful)")
    lines.append("")
    lines.append("## Results: Controlled Comparison (Base vs TTT-Discover, Same 19 Tasks)")
    lines.append("")
    lines.append("Same model, same 19 Frontier-CS tasks, same evaluator as Yuchen's STaR comparison. The TTT-Discover column reports rewards from the **best step** of the TTT run (paper's argmax_i protocol, Algorithm 1 line 12).")
    lines.append("")
    lines.append("Rewards are continuous scores (0.0–1.0) from the Frontier-CS evaluator. Each task has 8 rollouts.")
    lines.append("")
    lines.append("Method (faithful port of arxiv 2601.16175):")
    lines.append("- vLLM (TP=8) for sampling, HF + PEFT for LoRA gradient (rank 32), LoRA hot-reloaded into vLLM each step")
    lines.append("- PUCT prioritization over state archive (Appendix A.2)")
    lines.append("- Entropic objective with adaptive β solving KL(qβ‖uniform)=ln(2) by bisection (Appendix A.1)")
    lines.append("- KL penalty against base policy, importance-sampling correction for sampler/learner mismatch")
    lines.append("- Adam(lr=4e-5, β1=0.9, β2=0.95, ε=1e-8); 10 steps × 8 rollouts/step (paper uses 50×512; 1/640 budget)")
    lines.append("")

    for section, task_list in [("In-Distribution Tasks (6 tasks)", ID_TASKS),
                                ("Out-of-Distribution Tasks (11 OOD)", OOD_TASKS)]:
        lines.append(f"### {section}")
        lines.append("")
        lines.append("| Task | Base avg | Base max | TTT avg | TTT max | Δ avg |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        base_avgs, base_maxs, ttt_avgs, ttt_maxs = [], [], [], []
        sub_rows = []
        for t in task_list:
            display = YUCHEN_DISPLAY.get(t, t)
            base_r = collect_base_rewards(base_dir, t)
            ttt_r = collect_ttt_best_step_rewards(ttt_dir, t)
            ba, bm = avg_max(base_r)
            ta, tm = avg_max(ttt_r)
            ba_s = f"{ba:.4f}" if ba is not None else "n/a"
            bm_s = f"{bm:.4f}" if bm is not None else "n/a"
            ta_s = f"{ta:.4f}" if ta is not None else "(running)"
            tm_s = f"{tm:.4f}" if tm is not None else "(running)"
            if ba is not None and ta is not None:
                d = ta - ba
                d_str = f"**{d:+.4f}**" if d > 0.001 else f"{d:+.4f}"
            else:
                d_str = "(running)"
            lines.append(f"| {display} | {ba_s} | {bm_s} | {ta_s} | {tm_s} | {d_str} |")
            if ba is not None:
                base_avgs.append(ba); base_maxs.append(bm)
            if ta is not None:
                ttt_avgs.append(ta); ttt_maxs.append(tm)
            sub_rows.append((display, base_r, ttt_r))

        if base_avgs:
            ba_t = sum(base_avgs) / len(base_avgs)
            bm_t = max(base_maxs)
            ta_t = sum(ttt_avgs) / len(ttt_avgs) if ttt_avgs else 0
            tm_t = max(ttt_maxs) if ttt_maxs else 0
            d_t = ta_t - ba_t
            d_s = f"**{d_t:+.4f}**" if d_t > 0.001 else f"{d_t:+.4f}"
            lines.append(f"| **TOTAL** | **{ba_t:.4f}** | **{bm_t:.4f}** | **{ta_t:.4f}** | **{tm_t:.4f}** | {d_s} |")
        lines.append("")
        lines.append(f"#### Per-rollout reward distributions ({section.split()[0]})")
        lines.append("")
        lines.append("| Task | Base rollouts | TTT-Discover rollouts (best step) |")
        lines.append("| --- | --- | --- |")
        for display, br, tr in sub_rows:
            lines.append(f"| {display} | {fmt_dist(br)} | {fmt_dist(tr)} |")
        lines.append("")

    # Side-by-side with Yuchen's STaR
    lines.append("## Side-by-side: TTT-Discover vs STaR (ID tasks)")
    lines.append("")
    lines.append("| Task | Yuchen base | Yuchen STaR | Our base | Our TTT-Discover |")
    lines.append("| --- | --- | --- | --- | --- |")
    yuchen_rows = [
        ("cbl__high_av_loose_dl_small", 0.1250, 0.0417),
        ("cbl__low_av_tight_dl_large", 0.0488, 0.2050),
        ("cbl__mixed_av_loose_dl_large", 0.1400, 0.2375),
        ("cbl_multi__high_av_loose_dl_small", 0.0681, 0.2637),
        ("cbl_multi__high_av_tight_dl_small", 0.0681, 0.0822),
        ("cbl_multi__low_av_loose_dl_small", 0.4392, 0.3294),
    ]
    yu_base_t, yu_star_t = sum(r[1] for r in yuchen_rows)/6, sum(r[2] for r in yuchen_rows)/6
    our_b_avgs, our_t_avgs = [], []
    for display, yb, ys in yuchen_rows:
        tk = next((k for k, v in YUCHEN_DISPLAY.items() if v == display), None)
        ob_r = collect_base_rewards(base_dir, tk) if tk else None
        ot_r = collect_ttt_best_step_rewards(ttt_dir, tk) if tk else None
        ob = avg_max(ob_r)[0] if ob_r else 0
        ot = avg_max(ot_r)[0] if ot_r else 0
        our_b_avgs.append(ob); our_t_avgs.append(ot)
        lines.append(f"| {display} | {yb:.4f} | {ys:.4f} | {ob:.4f} | {ot:.4f} |")
    ob_t = sum(our_b_avgs) / 6
    ot_t = sum(our_t_avgs) / 6
    lines.append(f"| **TOTAL** | **{yu_base_t:.4f}** | **{yu_star_t:.4f}** | **{ob_t:.4f}** | **{ot_t:.4f}** |")
    lines.append("")
    lines.append(f"- Δ STaR (Yuchen): {yu_star_t - yu_base_t:+.4f}")
    lines.append(f"- Δ TTT-Discover (ours): **{ot_t - ob_t:+.4f}**")
    lines.append("")
    Path(args.output).write_text("\n".join(lines))
    print(f"[report] wrote {args.output}")


if __name__ == "__main__":
    main()
