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
    k = max(1, int(alpha * len(x)))
    return float(np.mean(np.sort(x)[:k]))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", type=str, default="results/eval")
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--out", type=str, default="results/STABILITY.md")
    p.add_argument("--collapse-threshold", type=float, default=0.01)  # Mbps
    p.add_argument("--stochastic", action="store_true",
                   help="read *_eval_stoch.csv instead of *_eval.csv -- P3 primary readout "
                        "(goal1.md). collapse_rate is a primary metric, so it is read the "
                        "same way every other gating number is")
    args = p.parse_args()

    suffix = "_eval_stoch" if args.stochastic else "_eval"
    df = load_eval_dir(Path(args.eval_dir), suffix=suffix)
    df = df[df.tag == args.tag]
    if df.empty:
        raise SystemExit(f"no rows with tag={args.tag!r} under {args.eval_dir}")

    lines = [
        "# Stability Report — collapse rate over embb_p5_mbps\n",
        f"Readout: `{'stochastic (P3 primary)' if args.stochastic else 'greedy (report-only, not gate)'}`.\n",
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

    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
