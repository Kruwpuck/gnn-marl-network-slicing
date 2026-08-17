"""
Collapse-rate / stability report from existing held-out eval data (pre-registration
Fase 0, docs/rev2-implementation-plan.md #3). Zero GPU — reads results/eval/*_eval.csv.

A seed "collapses" if its mean embb_p5_mbps (cell-edge throughput) falls below
--collapse-threshold. Unit of collapse is the seed, not the episode.

Usage:
  python scripts/stability_report.py
  python scripts/stability_report.py --tag _floornone --out results/STABILITY_floornone.md
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.rliable_report import load_eval_dir

KPI = "embb_p5_mbps"


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p_hat = k / n
    center = (p_hat + z * z / (2 * n)) / (1 + z * z / n)
    half = z / (1 + z * z / n) * np.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))
    return (center - half, center + half)


def cvar(x: np.ndarray, alpha: float = 0.2) -> float:
    """Mean of the worst alpha-fraction. Lower is worse for every KPI used here."""
    k = max(1, int(alpha * len(x)))
    return float(np.mean(np.sort(x)[:k]))


def stratified_cvar_ci(sub, kpi: str, alpha: float, n_boot: int = 2000,
                       rng_seed: int = 0) -> tuple[float, float]:
    """95% CI for episode-level CVaR: resample seeds, then episodes within the drawn seeds.

    Bootstrapping episodes directly would treat 150 episodes of one seed as 150 independent
    observations. They are not — the between-seed spread is what carries the uncertainty
    here (the same argument Gate A1 makes for SE_seed over SE_episode), and the naive
    version would report an interval far tighter than the data supports.
    """
    rng = np.random.default_rng(rng_seed)
    seeds = sorted(sub.seed_parsed.unique())
    per_seed = {s: sub[sub.seed_parsed == s][kpi].to_numpy() for s in seeds}
    stats = []
    for _ in range(n_boot):
        drawn = rng.choice(seeds, size=len(seeds), replace=True)
        pooled = np.concatenate([rng.choice(per_seed[s], size=per_seed[s].size, replace=True)
                                 for s in drawn])
        stats.append(cvar(pooled, alpha))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", type=str, default="results/eval")
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--out", type=str, default="results/STABILITY.md")
    p.add_argument("--collapse-threshold", type=float, default=0.01)  # Mbps
    p.add_argument("--alpha", type=float, default=0.2, help="CVaR tail fraction")
    p.add_argument("--kpis", type=str,
                   default="embb_p5_mbps,timely_throughput_mbps,sla_satisfaction_pct",
                   help="KPIs for the CVaR table (goal1.md §Fallback poin 3)")
    p.add_argument("--stochastic", action="store_true",
                   help="read *_eval_stoch.csv instead of *_eval.csv -- P3 primary readout "
                        "(goal1.md). collapse_rate is a primary metric, so it is read the "
                        "same way every other gating number is")
    p.add_argument("--readout", choices=["greedy", "stochastic", "primary"], default=None,
                   help="'primary' is the gating readout per family: sampled for PPO (P3), "
                        "argmax for DQN (determination 2026-08-16, measured). Overrides "
                        "--stochastic when given.")
    args = p.parse_args()

    readout = args.readout or ("stochastic" if args.stochastic else "greedy")
    suffix = {"greedy": "_eval", "stochastic": "_eval_stoch", "primary": "primary"}[readout]
    df = load_eval_dir(Path(args.eval_dir), suffix=suffix)
    df = df[df.tag == args.tag]
    if df.empty:
        raise SystemExit(f"no rows with tag={args.tag!r} under {args.eval_dir}")

    lines = [
        "# Stability Report — collapse rate over embb_p5_mbps\n",
        f"Readout: `{ {'greedy': 'greedy (report-only, not gate)', 'stochastic': 'stochastic (P3 primary; for DQN this is the discarded epsilon=1.0 column)', 'primary': 'primary per family — sampled for PPO (P3), argmax for DQN (2026-08-16)'}[readout] }`.\n",
        f"Tag filter: `{args.tag or '(main wave)'}`. Collapse threshold: "
        f"{args.collapse_threshold} Mbps. Unit of collapse is the seed "
        f"(mean over its held-out episodes), not the episode.\n",
        "| algo | seeds collapsed | rate | 95% Wilson CI | worst seed mean | CVaR@20% |",
        "|---|---|---|---|---|---|",
    ]

    for algo_key in sorted(df.algo_key.unique()):
        sub = df[df.algo_key == algo_key]
        seeds = sorted(sub.seed_parsed.unique())
        per_seed_mean = np.array([sub[sub.seed_parsed == s][KPI].mean() for s in seeds])
        collapsed = per_seed_mean < args.collapse_threshold
        k, n = int(collapsed.sum()), len(seeds)
        lo, hi = wilson_ci(k, n)
        worst = float(per_seed_mean.min()) if n else float("nan")
        cv = cvar(sub[KPI].to_numpy())
        lines.append(
            f"| `{algo_key}` | {k}/{n} | {k / n:.2f} | [{lo:.2f}, {hi:.2f}] "
            f"| {worst:.6f} | {cv:.6f} |"
        )

    kpis = [k for k in args.kpis.split(",") if k]
    n_seeds = df.groupby("algo_key").seed_parsed.nunique().max()
    seed_cvar_k = max(1, int(args.alpha * n_seeds))
    lines += [
        f"\n## CVaR@{args.alpha:.0%} — tail risk beside the collapse rate\n",
        "Collapse rate answers *how often* a seed lands in the bad mode; CVaR answers *how "
        "bad the tail is* when it does. Two units are reported because they are not the "
        "same statistic:\n",
        "- **episode CVaR** — mean of the worst episodes pooled across seeds, i.e. the "
        "within-run tail. 95% CI by stratified bootstrap (seeds resampled first, then "
        "episodes inside them).",
        f"- **seed CVaR** — mean of the worst seeds. At n={n_seeds} seeds and "
        f"alpha={args.alpha:g} this takes the worst {seed_cvar_k} seed"
        f"{'s' if seed_cvar_k > 1 else ''}, so it **degenerates to the worst-seed mean** "
        "and carries no more information than the column above. Printed anyway rather than "
        "quietly dropped: the degeneracy is a consequence of C4 failing at 5 seeds.\n",
        "| algo | KPI | episode CVaR | 95% CI | seed CVaR | mean |",
        "|---|---|---|---|---|---|",
    ]
    for algo_key in sorted(df.algo_key.unique()):
        sub = df[df.algo_key == algo_key]
        seeds = sorted(sub.seed_parsed.unique())
        for kpi in kpis:
            ep_cvar = cvar(sub[kpi].to_numpy(), args.alpha)
            lo_c, hi_c = stratified_cvar_ci(sub, kpi, args.alpha)
            per_seed_mean = np.array([sub[sub.seed_parsed == s][kpi].mean() for s in seeds])
            lines.append(
                f"| `{algo_key}` | `{kpi}` | {ep_cvar:.6f} | [{lo_c:.6f}, {hi_c:.6f}] "
                f"| {cvar(per_seed_mean, args.alpha):.6f} | {sub[kpi].mean():.6f} |"
            )

    # Whether the tail statistic separates the algorithms at all is itself a result, so it
    # is computed here rather than left for a reader to eyeball out of 24 numbers.
    cell_edge = KPI
    ep_cvars = {a: cvar(df[df.algo_key == a][cell_edge].to_numpy(), args.alpha)
                for a in sorted(df.algo_key.unique())}
    worst_algo, worst_val = min(ep_cvars.items(), key=lambda kv: kv[1])
    best_algo, best_val = max(ep_cvars.items(), key=lambda kv: kv[1])
    all_below = all(v < args.collapse_threshold for v in ep_cvars.values())
    lines.append(
        f"\nOn `{cell_edge}` the episode-level tail spans {worst_val:.6f} (`{worst_algo}`) to "
        f"{best_val:.6f} (`{best_algo}`)"
        + (f", and **every** algorithm's tail sits below the {args.collapse_threshold} Mbps "
           "collapse threshold. In the worst 20% of episodes cell-edge service is absent for "
           "all of them; what differs between architectures is how often a whole seed lands "
           "in that mode, which is what the collapse rate above measures. CVaR is reported "
           "as the complement it is, not as a second version of the same finding."
           if all_below else
           ". Not every algorithm's tail falls below the collapse threshold, so the two "
           "statistics are separating different things here — read them together.")
    )

    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
