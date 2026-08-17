"""Gate B (discrimination) recomputed from eval data, reproducibly.

Gate B was previously computed by hand; the only record of how is a ledger line. That was
survivable until the DQN "stochastic" readout turned out to be uniform random actions
(goal1.md 2026-08-16), because the gate numbers span all 8 algorithms and four of them were
being read wrong. A hand-computed gate cannot be re-derived under a corrected readout, so
it lives in a script now.

Definitions are copied from runs/2026-08-05-run01/ledger.md 2026-08-14T19:00, not invented:
  B1  timely_throughput_mbps  (max-min)/min as a percentage   >= 5%
  B2  sla_satisfaction_pct    max-min in pp                   >= 5 pp
  B3  urllc_delay_p99         max-min in ms                   >= 2 ms
  B4  KPIs saturated (one value shared by >= 5 of 8 algos)    <= 1 of 5

Only the 8 pre-registered algorithms count. Baselines added later (mlp-knn-ppo) are excluded
by construction: Gate B is a pre-registration, and widening its algorithm set after seeing
results would change the range it measures.

Usage:
  python scripts/gate_b_report.py --tag _v4 --readout primary
  python scripts/gate_b_report.py --tag _v4 --readout stochastic   # the contaminated one
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.rliable_report import load_eval_dir

PREREGISTERED = ["gnn-madqn_gat", "gnn-madqn_sage", "gnn-mappo_gat", "gnn-mappo_sage",
                 "idqn", "central-dqn", "ippo", "central-ppo"]
PRIMARY_KPIS = ["timely_throughput_mbps", "sla_satisfaction_pct", "urllc_delay_p99",
                "embb_p5_mbps", "jains_fairness"]
SATURATION_QUORUM = 5   # "identical on 5 of 8 algorithms" (ledger 2026-08-14)
ROUND_DP = 4

# Gate B is frozen at the 5 seeds the wave pre-registered (goal1.md, "Eksekusi perluasan
# seed keluarga PPO 2026-08-17"). The PPO family now has 20 seeds and the DQN family still
# has 5; B1-B4 are ranges of means *across* the 8 algorithms, so unequal n would mix
# estimates of different precision and move a pre-registered range for a statistical reason
# rather than because the task's discriminating power changed. Enforced here rather than
# remembered: without it, re-running this script after the extension silently produces
# mixed-n gate numbers.
GATE_SEEDS = [42, 43, 44, 45, 46]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", type=str, default="results/eval")
    p.add_argument("--tag", type=str, default="_v4")
    p.add_argument("--readout", choices=["greedy", "stochastic", "primary"], default="primary")
    p.add_argument("--seeds", type=str, default=",".join(map(str, GATE_SEEDS)),
                   help="seeds the gate is computed over. Frozen at the pre-registered five; "
                        "overriding this changes what a pre-registered range measures, so it "
                        "is a decision to record, not a convenience flag.")
    p.add_argument("--out", type=str, default="results/GATE_B_v4.md")
    args = p.parse_args()

    suffix = {"greedy": "_eval", "stochastic": "_eval_stoch", "primary": "primary"}[args.readout]
    seeds = [int(s) for s in args.seeds.split(",")]
    df = load_eval_dir(Path(args.eval_dir), suffix=suffix)
    df = df[(df.tag == args.tag) & (df.algo_key.isin(PREREGISTERED))]
    dropped = int((~df.seed_parsed.isin(seeds)).sum())
    df = df[df.seed_parsed.isin(seeds)]
    if df.empty:
        raise SystemExit(f"no rows for tag={args.tag!r} among the pre-registered algorithms")

    per_algo_seeds = {a: sorted(df[df.algo_key == a].seed_parsed.unique()) for a in df.algo_key.unique()}
    if len({len(v) for v in per_algo_seeds.values()}) != 1:
        # Never silently: an unequal-n Gate B is not the gate that was pre-registered.
        raise SystemExit(f"unequal seed counts across algorithms, refusing to compute a "
                         f"pre-registered range: { {a: len(v) for a, v in per_algo_seeds.items()} }")

    means = {kpi: {a: float(df[df.algo_key == a][kpi].mean()) for a in sorted(df.algo_key.unique())}
             for kpi in PRIMARY_KPIS}
    n_algos = len(means[PRIMARY_KPIS[0]])

    lines = [
        "# Gate B — discrimination, recomputed\n",
        f"Readout: `{args.readout}`. Tag `{args.tag}`. {n_algos} pre-registered algorithms, "
        f"seeds `{args.seeds}`; later additions such as `mlp-knn-ppo` are excluded because "
        "Gate B is a pre-registration and widening its algorithm set after seeing results "
        "would change the range it measures.\n",
        (f"**Seed freeze.** {dropped} eval rows from seeds outside the pre-registered set were "
         "excluded. The PPO family was extended to 20 seeds on 2026-08-17 while the DQN family "
         "stayed at 5; B1-B4 are ranges *across* the 8 algorithms, so unequal n would shift a "
         "pre-registered range for a statistical reason rather than a change in the task. The "
         "extra seeds are used for the C4 characterisation of the PPO family instead "
         "(`results/STABILITY_v4_primary.md`).\n") if dropped else "",
        "| # | KPI | min | max | range | threshold | verdict |",
        "|---|---|---|---|---|---|---|",
    ]

    def row(tag, kpi, value, unit, threshold, passed):
        lo_a = min(means[kpi], key=means[kpi].get)
        hi_a = max(means[kpi], key=means[kpi].get)
        lines.append(
            f"| {tag} | `{kpi}` | {means[kpi][lo_a]:.4f} (`{lo_a}`) | "
            f"{means[kpi][hi_a]:.4f} (`{hi_a}`) | {value:.4f} {unit} | {threshold} | "
            f"**{'LOLOS' if passed else 'GAGAL'}** |")

    thr = np.array(list(means["timely_throughput_mbps"].values()))
    b1 = (thr.max() - thr.min()) / thr.min() * 100.0
    row("B1", "timely_throughput_mbps", b1, "%", ">= 5%", b1 >= 5.0)

    sla = np.array(list(means["sla_satisfaction_pct"].values()))
    b2 = sla.max() - sla.min()
    row("B2", "sla_satisfaction_pct", b2, "pp", ">= 5 pp", b2 >= 5.0)

    p99 = np.array(list(means["urllc_delay_p99"].values()))
    b3 = p99.max() - p99.min()
    row("B3", "urllc_delay_p99", b3, "ms", ">= 2 ms", b3 >= 2.0)

    saturated = []
    for kpi in PRIMARY_KPIS:
        vals = np.round(np.array(list(means[kpi].values())), ROUND_DP)
        _, counts = np.unique(vals, return_counts=True)
        if counts.max() >= SATURATION_QUORUM:
            saturated.append(kpi)
    lines.append(
        f"| B4 | saturated KPIs | - | - | {len(saturated)} of {len(PRIMARY_KPIS)}"
        f"{' (' + ', '.join(f'`{k}`' for k in saturated) + ')' if saturated else ''} "
        f"| <= 1 of 5 | **{'LOLOS' if len(saturated) <= 1 else 'GAGAL'}** |")

    # C3 asks whether any Gate B verdict depends on pooling the two budget families. The
    # gate itself is defined across all 8 algorithms and is not re-run here; this split is
    # the answer to that question and nothing else. It lives in the generated report so it
    # cannot go stale when the readout changes, which is what happened to the hand-written
    # copy in results/GATE_C.md.
    lines.append("\n## C3 supplement — the same ranges within each budget family\n")
    lines.append("Not a re-run of the gate. The pre-registered verdict above stands as "
                 "computed across all 8 algorithms; this split only answers whether any "
                 "verdict depends on mixing the 200K-step and 1M-step families.\n")
    lines.append("| set | B1 relative range | B2 range | B3 range |")
    lines.append("|---|---|---|---|")
    families = {"all 8 (pre-registered, **this is the gate**)": PREREGISTERED,
                "DQN family (4, 200K steps)": [a for a in PREREGISTERED if "dqn" in a],
                "PPO family (4, 1M steps)": [a for a in PREREGISTERED if "dqn" not in a]}
    for label, members in families.items():
        sel = [a for a in members if a in means["timely_throughput_mbps"]]
        t = np.array([means["timely_throughput_mbps"][a] for a in sel])
        s = np.array([means["sla_satisfaction_pct"][a] for a in sel])
        d = np.array([means["urllc_delay_p99"][a] for a in sel])
        lines.append(f"| {label} | {(t.max() - t.min()) / t.min() * 100:.2f}% | "
                     f"{s.max() - s.min():.2f} pp | {d.max() - d.min():.2f} ms |")

    lines.append("\n## Per-algorithm means\n")
    lines.append("| algo | " + " | ".join(f"`{k}`" for k in PRIMARY_KPIS) + " |")
    lines.append("|" + "---|" * (len(PRIMARY_KPIS) + 1))
    for a in sorted(means[PRIMARY_KPIS[0]]):
        lines.append(f"| `{a}` | " + " | ".join(f"{means[k][a]:.4f}" for k in PRIMARY_KPIS) + " |")

    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"B1={b1:.2f}%  B2={b2:.2f}pp  B3={b3:.2f}ms  B4={len(saturated)}/5")


if __name__ == "__main__":
    main()
