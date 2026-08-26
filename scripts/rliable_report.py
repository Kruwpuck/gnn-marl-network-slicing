"""
IQM + stratified-bootstrap-CI comparison of GNN-MARL variants vs their matched
baselines, per training-budget family (DQN 200K steps vs PPO 1M steps are never
pooled — different budgets make a cross-family reward/KPI comparison meaningless).

Reads results/eval/*_eval.csv (per-episode KPI rows written by
scripts/evaluate_checkpoints.py), one file per algo x seed. Each seed's episodes
are averaged to one scalar per seed (rliable's "task" axis is degenerate here —
there's only one environment — so scores are shape (n_seeds, 1)); IQM + bootstrap
CI are then computed over seeds. Needs >=2 seeds per algo to be meaningful;
designed for the 5-seed wave.

Usage:
  python scripts/rliable_report.py --eval-dir results/eval --out results/RLIABLE.md
  python scripts/rliable_report.py --tag _floornone --out results/RLIABLE_floornone.md
"""
from __future__ import annotations
import argparse
import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from rliable import library as rly
    from rliable import metrics as rly_metrics
except ImportError:
    print("rliable not installed — `pip install rliable` (needs absl-py, arch).", file=sys.stderr)
    raise

MATCHED_BASELINES = {
    "gnn-madqn_gat": ("dqn", ["idqn", "central-dqn"]),
    "gnn-madqn_sage": ("dqn", ["idqn", "central-dqn"]),
    "gnn-mappo_gat": ("ppo", ["ippo", "central-ppo"]),
    "gnn-mappo_sage": ("ppo", ["ippo", "central-ppo"]),
}

# (column, higher_is_better)
KPIS = [
    ("timely_throughput_mbps", True),
    ("sla_satisfaction_pct", True),
    ("embb_p5_mbps", True),
    ("jains_fairness", True),
    ("urllc_delay_p99", False),
]


def parse_run_name(stem: str) -> tuple[str, str, int]:
    """'gnn-madqn_gat_floornone_seed42' -> ('gnn-madqn_gat', '_floornone', 42)."""
    # Backbone names are listed longest-first: a v6 run like 'gnn-mappo_gatres_seed42' does
    # not fail without them, it mis-parses silently -- 'gatres' matches the *tag* group, the
    # algo reads as bare 'gnn-mappo', and three separate arms collapse into one row.
    m = re.match(
        r"^(?P<algo>[a-z\-]+(?:_(?:gatres-edge|gatres|gatedge|gat|sage))?)"
        r"(?P<tag>_[a-z0-9]+)?_seed(?P<seed>\d+)$", stem)
    if not m:
        return stem, "", -1
    return m.group("algo"), (m.group("tag") or ""), int(m.group("seed"))


def primary_suffix(algo_key: str) -> str:
    """Which readout gates, per budget family (goal1.md, two separate determinations).

    PPO: sampled actions (P3, frozen 2026-08-08) -- the argmax carries ~0.2 of the action
    mass and misreads a policy by tens of Mbps.
    DQN: argmax (determination of 2026-08-16). Measured, not assumed: the pre-registered
    degeneracy test gave sd_greedy/sd_(eps=0.05) = 1.01-1.10 against a 2.0 threshold, where
    the degenerate PPO cases sat at 3.17 and 3.62. The old DQN "stochastic" column was
    epsilon=1.0, i.e. uniform random actions, and is discarded as an instrument fault.
    """
    return "_eval" if "dqn" in algo_key else "_eval_stoch"


def load_eval_dir(eval_dir: Path, suffix: str = "_eval") -> pd.DataFrame:
    """suffix='primary' picks the gating readout per algorithm; anything else is literal."""
    patterns = ["_eval", "_eval_stoch"] if suffix == "primary" else [suffix]
    frames = []
    seen = set()
    for pat in patterns:
        for path in sorted(glob.glob(str(eval_dir / f"*{pat}.csv"))):
            stem = Path(path).stem
            if pat == "_eval" and stem.endswith("_eval_stoch"):
                continue  # the *_eval glob also matches *_eval_stoch
            stem = stem[: -len(pat)]
            algo_key, tag, seed = parse_run_name(stem)
            if suffix == "primary" and pat != primary_suffix(algo_key):
                continue
            if (algo_key, tag, seed) in seen:
                continue
            seen.add((algo_key, tag, seed))
            df = pd.read_csv(path)
            df["algo_key"] = algo_key
            df["tag"] = tag
            df["seed_parsed"] = seed
            df["readout"] = pat
            frames.append(df)
    if not frames:
        raise SystemExit(f"No {'|'.join(patterns)} .csv files found under {eval_dir}")
    return pd.concat(frames, ignore_index=True)


def per_seed_means(df: pd.DataFrame, algo_key: str, tag: str, kpi: str) -> np.ndarray | None:
    sub = df[(df.algo_key == algo_key) & (df.tag == tag)]
    if sub.empty:
        return None
    seeds = sorted(sub.seed_parsed.unique())
    means = [sub[sub.seed_parsed == s][kpi].mean() for s in seeds]
    if len(means) < 2:
        return None
    return np.asarray(means, dtype=np.float64).reshape(-1, 1)  # (n_seeds, 1) — 1 degenerate "task"


def compare(df: pd.DataFrame, proposed: str, baselines: list[str], tag: str, reps: int) -> list[str]:
    lines = []
    for kpi, higher_better in KPIS:
        score_dict = {}
        for algo in [proposed] + baselines:
            arr = per_seed_means(df, algo, tag, kpi)
            if arr is None:
                continue
            score_dict[algo] = arr if higher_better else -arr
        if proposed not in score_dict or len(score_dict) < 2:
            lines.append(f"- **{kpi}**: skipped (need >=2 seeds of eval data for "
                         f"`{proposed}` and at least one baseline)")
            continue

        aggregate_func = lambda x: np.array([rly_metrics.aggregate_iqm(x)])
        point_est, interval_est = rly.get_interval_estimates(score_dict, aggregate_func, reps=reps)

        lines.append(f"- **{kpi}** ({'higher' if higher_better else 'lower'} is better):")
        for algo in [proposed] + baselines:
            if algo not in point_est:
                continue
            lo, hi = interval_est[algo][0][0], interval_est[algo][1][0]
            val = point_est[algo][0]
            lo_r, hi_r = (lo, hi) if higher_better else (-hi, -lo)
            val_r = val if higher_better else -val
            marker = " (proposed)" if algo == proposed else ""
            lines.append(f"  - `{algo}`{marker}: IQM={val_r:.4f}  95% CI=[{lo_r:.4f}, {hi_r:.4f}]")

        # Overlap verdict vs each baseline, computed in the *flipped* (higher-is-better) sign
        # so "proposed > baseline" always means "proposed wins on this KPI".
        p_lo, p_hi = interval_est[proposed][0][0], interval_est[proposed][1][0]
        for algo in baselines:
            if algo not in interval_est:
                continue
            b_lo, b_hi = interval_est[algo][0][0], interval_est[algo][1][0]
            if not (p_hi < b_lo or b_hi < p_lo):
                verdict = "COMPARABLE (CIs overlap)"
            elif p_lo > b_hi:
                verdict = "proposed BETTER (CIs disjoint)"
            else:
                verdict = "proposed WORSE (CIs disjoint)"
            lines.append(f"    -> vs `{algo}`: {verdict}")

        # P(improvement): pairwise seed-vs-seed comparison, orthogonal to the IQM-CI
        # overlap verdict above (Mann-Whitney U over seed scores, not aggregate-level).
        proposed_scores = score_dict[proposed]
        for algo in baselines:
            if algo not in score_dict:
                continue
            pair = {f"{proposed},{algo}": (proposed_scores, score_dict[algo])}
            poi_point, poi_ci = rly.get_interval_estimates(
                pair, lambda x, y: np.array([rly_metrics.probability_of_improvement(x, y)]),
                reps=min(reps, 2000),  # mannwhitneyu-per-resample is the bottleneck, not sample size
            )
            key = f"{proposed},{algo}"
            p_val = poi_point[key][0]
            p_lo_r, p_hi_r = poi_ci[key][0][0], poi_ci[key][1][0]
            lines.append(
                f"    -> P(`{proposed}` > `{algo}`) = {p_val:.3f}  95% CI=[{p_lo_r:.3f}, {p_hi_r:.3f}]"
            )

        # Performance profile: fraction of seeds at/above tau, at each baseline's median.
        # Cheap complement to IQM point estimates — shows spread, not just central tendency.
        for algo in baselines:
            if algo not in score_dict:
                continue
            tau = float(np.median(score_dict[algo]))
            profile, _ = rly.create_performance_profile(
                {proposed: score_dict[proposed], algo: score_dict[algo]}, [tau]
            )
            tau_display = tau if higher_better else -tau
            lines.append(
                f"    -> performance profile @ tau=median({algo})={tau_display:.4f}: "
                f"P(`{proposed}` at least as good as tau)={profile[proposed][0]:.3f}  "
                f"P(`{algo}` at least as good as tau)={profile[algo][0]:.3f}"
            )
    return lines


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", type=str, default="results/eval")
    p.add_argument("--tag", type=str, default="", help="only report runs with this --tag suffix (default: main wave, tag='')")
    p.add_argument("--out", type=str, default="results/RLIABLE.md")
    p.add_argument("--reps", type=int, default=50_000)
    p.add_argument("--stochastic", action="store_true",
                    help="read *_eval_stoch.csv instead of *_eval.csv -- P3 primary readout (goal1.md)")
    p.add_argument("--readout", choices=["greedy", "stochastic", "primary"], default=None,
                   help="'primary' is the gating readout per family: sampled for PPO (P3), "
                        "argmax for DQN (determination 2026-08-16, measured). Overrides "
                        "--stochastic when given.")
    args = p.parse_args()

    readout = args.readout or ("stochastic" if args.stochastic else "greedy")
    suffix = {"greedy": "_eval", "stochastic": "_eval_stoch", "primary": "primary"}[readout]
    readout_label = {
        "greedy": "greedy (report-only, not gate)",
        "stochastic": "stochastic (P3 primary; for DQN this is the discarded epsilon=1.0 column)",
        "primary": "primary per family — sampled for PPO (P3), argmax for DQN (2026-08-16)",
    }[readout]
    df = load_eval_dir(Path(args.eval_dir), suffix=suffix)
    lines = ["# rliable Report — IQM + Stratified Bootstrap 95% CI\n",
             f"Readout: `{readout_label}`.\n",
             f"Tag filter: `{args.tag or '(main wave)'}`. DQN (200K steps) and PPO "
             f"(1M steps) families are never pooled. A CI overlap means "
             f"'comparable', not 'proposed wins' — report accordingly.\n"]

    for proposed, (family, baselines) in MATCHED_BASELINES.items():
        lines.append(f"\n## {proposed} vs {', '.join(baselines)} ({family.upper()} family)\n")
        try:
            lines.extend(compare(df, proposed, baselines, args.tag, args.reps))
        except Exception as e:
            lines.append(f"ERROR building comparison: {e}")

    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
