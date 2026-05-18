"""
Aggregate per-rollout reward JSONs into Yuchen-style table:

In-Distribution (6 cbl variants) and OOD (13 OOD tasks) tables, with
- Base avg, Base max, [TTT avg, TTT max, Delta] columns
- Per-rollout reward distribution (sorted, formatted as "0 0 0 0 .25 .25 .25 .25")

USAGE:
  # Just base
  python aggregate_report.py --base-eval-dir /fsx/xuanj/ttt-discover/results/base/eval --output report.md

  # With TTT
  python aggregate_report.py --base-eval-dir <base> --ttt-eval-dir <ttt> --output report.md
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict


def load_evals(eval_dir: Path) -> dict[str, dict[int, float]]:
    """Return {yuchen_task_name: {rollout_idx: reward_01}}"""
    if eval_dir is None or not eval_dir.exists():
        return {}
    out = defaultdict(dict)
    for p in eval_dir.glob("*.eval.json"):
        try:
            j = json.loads(p.read_text())
            t = j["task"]
            r = j["rollout"]
            # reward_01 in [0,1], can be 0 if score was 0 or negative
            reward = j.get("reward_01")
            if reward is None and "score_100" in j:
                reward = max(0.0, j["score_100"]) / 100.0
            out[t][r] = reward
        except Exception:
            continue
    return out


def fmt_reward(r: float) -> str:
    if r == 0:
        return "0"
    if r < 0.005:
        return f"{r:.3f}"  # tiny
    return f".{int(round(r*100)):02d}" if r < 1.0 else f"{r:.2f}"


def fmt_rollouts(rewards: list[float]) -> str:
    return " ".join(fmt_reward(r) for r in sorted(rewards))


def build_row(task_name: str, base: dict, ttt: dict | None, num_rollouts: int) -> dict:
    base_rewards = [base.get(i, 0.0) for i in range(num_rollouts)]
    base_avail = [base[i] for i in range(num_rollouts) if i in base]
    n_base = len(base_avail)

    if ttt:
        ttt_rewards = [ttt.get(i, 0.0) for i in range(num_rollouts)]
        ttt_avail = [ttt[i] for i in range(num_rollouts) if i in ttt]
        n_ttt = len(ttt_avail)
    else:
        ttt_rewards = []
        ttt_avail = []
        n_ttt = 0

    return {
        "task": task_name,
        "base_avg": (sum(base_avail) / n_base) if n_base else None,
        "base_max": max(base_avail) if n_base else None,
        "base_rollouts": base_rewards if n_base else None,
        "n_base": n_base,
        "ttt_avg": (sum(ttt_avail) / n_ttt) if n_ttt else None,
        "ttt_max": max(ttt_avail) if n_ttt else None,
        "ttt_rollouts": ttt_rewards if n_ttt else None,
        "n_ttt": n_ttt,
    }


def fmt_cell(v):
    if v is None:
        return "n/a"
    return f"{v:.4f}"


def fmt_delta(base_avg, ttt_avg):
    if base_avg is None or ttt_avg is None:
        return "n/a"
    d = ttt_avg - base_avg
    return f"**+{d:.4f}**" if d > 0 else f"{d:.4f}"


def render_table(rows: list[dict], title: str, has_ttt: bool) -> str:
    lines = [f"### {title}\n"]
    if has_ttt:
        lines.append("| Task | Base avg | Base max | TTT avg | TTT max | Δ avg |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
    else:
        lines.append("| Task | Base avg | Base max |")
        lines.append("| --- | --- | --- |")

    sum_base, sum_base_max = 0.0, 0.0
    sum_ttt, sum_ttt_max = 0.0, 0.0
    n_base, n_ttt = 0, 0

    for row in rows:
        if has_ttt:
            lines.append(f"| {row['task']} | {fmt_cell(row['base_avg'])} | {fmt_cell(row['base_max'])} | {fmt_cell(row['ttt_avg'])} | {fmt_cell(row['ttt_max'])} | {fmt_delta(row['base_avg'], row['ttt_avg'])} |")
        else:
            lines.append(f"| {row['task']} | {fmt_cell(row['base_avg'])} | {fmt_cell(row['base_max'])} |")
        if row["base_avg"] is not None:
            sum_base += row["base_avg"]
            sum_base_max = max(sum_base_max, row["base_max"])
            n_base += 1
        if row["ttt_avg"] is not None:
            sum_ttt += row["ttt_avg"]
            sum_ttt_max = max(sum_ttt_max, row["ttt_max"])
            n_ttt += 1

    avg_base = sum_base / n_base if n_base else None
    avg_ttt = sum_ttt / n_ttt if n_ttt else None
    if has_ttt:
        delta = (avg_ttt - avg_base) if (avg_base is not None and avg_ttt is not None) else None
        lines.append(f"| **TOTAL** | **{fmt_cell(avg_base)}** | **{fmt_cell(sum_base_max)}** | **{fmt_cell(avg_ttt)}** | **{fmt_cell(sum_ttt_max)}** | **{fmt_delta(avg_base, avg_ttt)}** |")
    else:
        lines.append(f"| **TOTAL** | **{fmt_cell(avg_base)}** | **{fmt_cell(sum_base_max)}** |")
    return "\n".join(lines)


def render_rollout_table(rows: list[dict], title: str, has_ttt: bool) -> str:
    lines = [f"\n#### Per-rollout reward distributions ({title})\n"]
    if has_ttt:
        lines.append("| Task | Base rollouts | TTT rollouts |")
        lines.append("| --- | --- | --- |")
    else:
        lines.append("| Task | Base rollouts |")
        lines.append("| --- | --- |")
    for row in rows:
        b = fmt_rollouts(row["base_rollouts"]) if row["base_rollouts"] else "(no data)"
        if has_ttt:
            t = fmt_rollouts(row["ttt_rollouts"]) if row["ttt_rollouts"] else "(no data)"
            lines.append(f"| {row['task']} | {b} | {t} |")
        else:
            lines.append(f"| {row['task']} | {b} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-json", default="/fsx/xuanj/ttt-discover/bench/tasks_19.json")
    ap.add_argument("--base-eval-dir", default="/fsx/xuanj/ttt-discover/results/base/eval")
    ap.add_argument("--ttt-eval-dir", default=None)
    ap.add_argument("--num-rollouts", type=int, default=8)
    ap.add_argument("--output", default="/fsx/xuanj/ttt-discover/report.md")
    args = ap.parse_args()

    data = json.loads(Path(args.tasks_json).read_text())
    base = load_evals(Path(args.base_eval_dir))
    ttt = load_evals(Path(args.ttt_eval_dir)) if args.ttt_eval_dir else {}
    has_ttt = bool(ttt)

    id_rows = [build_row(t["yuchen_name"], base.get(t["yuchen_name"], {}), ttt.get(t["yuchen_name"], {}) if has_ttt else None, args.num_rollouts)
               for t in data["in_distribution"] if not t.get("yuchen_name", "").startswith("_")]
    ood_rows = [build_row(t["yuchen_name"], base.get(t["yuchen_name"], {}), ttt.get(t["yuchen_name"], {}) if has_ttt else None, args.num_rollouts)
                for t in data["out_of_distribution"] if not t.get("yuchen_name", "").startswith("_")]

    md = []
    md.append(f"# TTT-Discover on Frontier-CS — DeepSeek-R1-0528-Qwen3-8B\n")
    md.append("Replicates Yuchen's STaR table layout. Rewards are continuous scores in [0,1] (frontier-cs/100).\n")
    md.append(f"Base column: vanilla model, no TTT. {('TTT column: TTT-Discover (paper Yuksekgonul et al. 2026, arxiv 2601.16175).' if has_ttt else '')}\n")
    md.append("`passed` = reward > 0 implicitly. Each task has 8 rollouts.\n")

    md.append("\n## Results\n")
    md.append(render_table(id_rows, "In-Distribution Tasks (6 tasks)", has_ttt))
    md.append(render_rollout_table(id_rows, "ID", has_ttt))
    md.append("\n")
    md.append(render_table(ood_rows, "Out-of-Distribution Tasks (13 OOD)", has_ttt))
    md.append(render_rollout_table(ood_rows, "OOD", has_ttt))
    md.append("\n")

    Path(args.output).write_text("\n".join(md))
    print(f"[report] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
