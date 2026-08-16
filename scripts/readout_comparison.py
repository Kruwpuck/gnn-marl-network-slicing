"""
Greedy vs stochastic readout, side by side — the evidence behind protocol P3.

`handoff/goal1.md` P3 makes the stochastic readout primary and requires the
greedy one to be reported alongside, never dropped. That is not a formality:
whether a policy's argmax collapses is a finding about the policy, and if some
architectures collapse and others do not, the gap between the two readouts is
data, not noise to be cleaned away.

Per algorithm, over the same held-out episodes (results/eval/*_eval.csv vs
*_eval_stoch.csv):

  greedy / stoch      mean KPI under each readout, and their difference
  sd_ep               per-episode sd of sla_violation_pct under each readout.
                      P3's second qualitative sign of a dead readout is this
                      collapsing (measured 0.11 pp against a normal 11-14 pp) —
                      a reading that has stopped responding to state.
  p_max/entropy/agree from scripts/policy_confidence.py, PPO only (DQN has
                      Q-values, not a distribution). `agree` < 0.5 is P3's first
                      sign: the argmax is not what the policy actually does.

Usage:
  python scripts/policy_confidence.py --runs "results/logs/*ppo*_v4_seed*.pt" \
      --episodes 5 --out results/policy_confidence_v4.csv
  python scripts/readout_comparison.py --tag _v4 --out results/READOUT_COMPARISON.md
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.rliable_report import load_eval_dir, parse_run_name

KPIS = ["timely_throughput_mbps", "sla_satisfaction_pct", "urllc_delay_p99", "embb_p5_mbps"]


def per_algo(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    sub = df[df.tag == tag]
    if sub.empty:
        raise SystemExit(f"no rows with tag '{tag}' — check --tag and the eval dir")
    agg = sub.groupby("algo_key")[KPIS].mean()
    agg["sd_ep"] = sub.groupby("algo_key")["sla_violation_pct"].std()
    return agg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", type=str, default="results/eval")
    p.add_argument("--tag", type=str, default="_v4")
    p.add_argument("--confidence", type=str, default="results/policy_confidence_v4.csv")
    p.add_argument("--out", type=str, default="results/READOUT_COMPARISON.md")
    args = p.parse_args()

    g = per_algo(load_eval_dir(Path(args.eval_dir), suffix="_eval"), args.tag)
    s = per_algo(load_eval_dir(Path(args.eval_dir), suffix="_eval_stoch"), args.tag)

    conf = None
    conf_path = Path(args.confidence)
    if conf_path.exists():
        cdf = pd.read_csv(conf_path)
        # key off the run name, not the `algo` column: policy_confidence.py copies
        # ckpt["algo"], which has no backbone, so gnn-mappo_gat and gnn-mappo_sage would
        # both join as "gnn-mappo" -- i.e. not at all, and silently read as "no data".
        cdf["algo_key"] = [parse_run_name(r)[0] for r in cdf["run"]]
        conf = cdf.groupby("algo_key")[["p_max", "entropy", "agree", "p_uniform", "ent_max"]].mean()

    lines = [
        "# Readout comparison — greedy vs stochastic (protocol P3)\n",
        f"Tag `{args.tag}`, held-out evaluation, mean over all seeds x episodes. "
        "The stochastic column is the gating readout (`handoff/goal1.md` P3, frozen "
        "2026-08-08, before any v4 result existed); the greedy column is reported, "
        "never used to gate, and never omitted.\n",
        "\n## KPI under each readout\n",
        "| algo | KPI | greedy | stochastic | stoch − greedy |",
        "|---|---|---|---|---|",
    ]
    for algo in sorted(g.index):
        if algo not in s.index:
            continue
        for kpi in KPIS:
            gv, sv = g.loc[algo, kpi], s.loc[algo, kpi]
            lines.append(f"| `{algo}` | {kpi} | {gv:.4f} | {sv:.4f} | {sv - gv:+.4f} |")

    lines += [
        "\n## Signs of a degenerate greedy readout\n",
        "`sd_ep` = per-episode sd of `sla_violation_pct`. A readout that has stopped "
        "responding to state shows a collapsed sd (P3 measured 0.11 pp against a normal "
        "11-14 pp). `agree` = fraction of steps where the argmax equals the action the "
        "policy actually sampled; `agree` < 0.5 means the argmax is not the policy's "
        "behaviour. Entropy is in nats against the `ln(n_actions)` ceiling.\n",
        "\n| algo | sd_ep greedy | sd_ep stoch | p_max | 1/n | entropy | ln n | agree |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for algo in sorted(g.index):
        if algo not in s.index:
            continue
        c = conf.loc[algo] if conf is not None and algo in conf.index else None
        cells = ([f"{c['p_max']:.3f}", f"{c['p_uniform']:.3f}", f"{c['entropy']:.3f}",
                  f"{c['ent_max']:.3f}", f"{c['agree']:.3f}"] if c is not None
                 else ["—", "—", "—", "—", "— (DQN: no action distribution)"])
        lines.append(f"| `{algo}` | {g.loc[algo, 'sd_ep']:.2f} | {s.loc[algo, 'sd_ep']:.2f} | "
                     + " | ".join(cells) + " |")

    if conf is None:
        lines.append(f"\n> `{args.confidence}` not found — confidence columns left empty. "
                     "Generate it with `scripts/policy_confidence.py` (PPO checkpoints only).")

    if conf is not None:
        gap = (s["timely_throughput_mbps"] - g["timely_throughput_mbps"]).reindex(conf.index).dropna()
        ent = conf["entropy"].reindex(gap.index)
        lines.append(
            f"\nAmong the algorithms with an action distribution, entropy spans only "
            f"{ent.max() - ent.min():.3f} nats ({ent.min():.3f}-{ent.max():.3f}, ceiling "
            f"{conf['ent_max'].max():.3f}) while their throughput readout gap spans "
            f"{gap.max() - gap.min():.2f} Mbps ({gap.min():+.2f} to {gap.max():+.2f}). The argmax is "
            "statistically degenerate to the same degree everywhere; whether that degeneracy wrecks "
            "the measured KPI is architecture-dependent. Entropy therefore does not predict which "
            "model the greedy readout will misrepresent — which is precisely why P3 gates on the "
            "stochastic readout rather than screening greedy numbers for plausibility.\n")

    thr_gap = float((s["timely_throughput_mbps"] - g["timely_throughput_mbps"]).abs().max())
    lines += [
        f"\nLargest single-KPI readout gap in this wave: **{thr_gap:.2f} Mbps** on "
        "`timely_throughput_mbps`. A gap that size is not a rounding difference between two "
        "ways of printing the same policy — it is the difference between measuring the policy "
        "and measuring its mode. Which architectures collapse and which do not is reported "
        "above as a finding, not filtered out.\n",
    ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
