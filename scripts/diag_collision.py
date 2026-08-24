"""
Fase 0 diagnostic D5 -- the collision-storm hypothesis (docs/revisi/PLAN-01-DIAGNOSTICS.md).

Hypothesis under test: during training the agents learn to share spectrum stochastically.
Forced to a distributed argmax with no synchronising channel, they all pick their
highest-probability action at once, interference spikes, SINR falls, the episode collapses.

The test is the contrast, not either number alone: `gnn-mappo_gat` loses 51.19 Mbps going
from sampled to greedy while `ippo` gains 0.46. If the hypothesis holds, the collapsing
model should show action agreement rising sharply under argmax and its per-gNB SINR series
falling together rather than one at a time; `ippo`, which has no fine-grained coordination
to break, should not.

This is a hypothesis being measured, not an explanation being written down. PLAN-01
forbids putting the collision-storm story in the paper before D5 confirms it, and records
a limit already known: policy entropy spans only 0.100 nat while the readout gap is 46.28
Mbps, so whatever this is, it is not a simple function of entropy.

Unlike D2/D3 this needs the per-step action vector, which run_episode does not return, so
the rollout loop is local. The policy is still read through the shared
scripts/evaluate_checkpoints.select_actions path -- only the KPI arithmetic is skipped,
and D5 does not need it.

Usage:
  python scripts/diag_collision.py
  python scripts/diag_collision.py --episodes 5
"""
from __future__ import annotations
import argparse
import glob
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.network_slicing_env import NetworkSlicingEnv
from scripts.evaluate_checkpoints import (EVAL_SEED_BASE, load_agent, run_episode,
                                          select_actions)
from scripts.rliable_report import parse_run_name

# An episode counts as collapsed when its timely throughput falls below this fraction of
# its own checkpoint's sampled-arm mean. Declared here, before the numbers are looked at.
#
# Note this is NOT the project's cell-edge collapse (embb_p5_mbps < 0.01 Mbps, unit = seed,
# scripts/stability_report.py). That one is a gate metric and is untouched. D5 is about the
# *greedy readout* collapse -- the 51.19 Mbps drop PLAN-01 D5 cites -- which lives in
# timely_throughput. Using the cell-edge rule here would select almost every *sampled*
# episode and almost no greedy one, i.e. the opposite of the population D5 needs.
#
# 0.50 comes from the task, not from the results: the cited collapse is ~68 -> ~17 Mbps,
# about 25% of baseline, so half separates cleanly without grazing. The other two are the
# mandatory sensitivity check -- if the verdict moves between them, the threshold is
# driving it and that is itself the finding.
COLLAPSE_FRAC = 0.50
SENSITIVITY_FRACS = (0.35, 0.65)


def rollout(env: NetworkSlicingEnv, agent, kind: str, seed: int, greedy: bool) -> dict:
    """One episode, keeping the per-step action vector and per-gNB SINR."""
    obs, info = env.reset(seed=seed)
    actions_t, sinr_t, embb_bps = [], [], []
    done = False
    while not done:
        actions = np.asarray(select_actions(agent, kind, obs, info, env, greedy))
        obs, _, terminated, truncated, info = env.step(actions)
        done = terminated or truncated
        actions_t.append(actions.copy())
        # Read-only: the env exposes SINR nowhere else, and D5 is about how the five series
        # move relative to each other, not about their absolute level.
        sinr_t.append(np.asarray(env._last_sinr_embb, dtype=np.float64).copy())
        embb_bps.extend(np.asarray(info["embb_rates"], dtype=np.float64).tolist())
    return {"actions": np.stack(actions_t),      # (T, N)
            "sinr": np.stack(sinr_t),            # (T, N)
            "embb_p5_mbps": float(np.percentile(embb_bps, 5) / 1e6)}


def mean_pairwise(series: np.ndarray, method: str) -> float:
    """Mean correlation over all gNB pairs of a (T, N) series. Constant columns have no
    defined correlation and are skipped rather than counted as 0 -- a gNB pinned at one
    action carries no evidence either way about agreement."""
    n = series.shape[1]
    vals = []
    for i, j in combinations(range(n), 2):
        a, b = series[:, i], series[:, j]
        if len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
            continue
        r = spearmanr(a, b).statistic if method == "spearman" else np.corrcoef(a, b)[0, 1]
        if not np.isnan(r):
            vals.append(float(r))
    return float(np.mean(vals)) if vals else float("nan")


def episode_stats(ep: dict, kpis: dict) -> dict:
    """Agreement and synchrony for one episode.

    `kpis` comes from scripts/evaluate_checkpoints.run_episode driven over the identical
    seed, so the throughput here is the same quantity every gate number uses. Re-deriving
    it locally would create a second definition of a protected metric, free to drift from
    the first.
    """
    actions, sinr = ep["actions"], ep["sinr"]
    n = actions.shape[1]
    # How much of the fleet picks the same tier at a given step. 1.0 = unanimous.
    mode_share = np.array([np.bincount(row).max() / n for row in actions])
    return {
        "mode_share_mean": float(mode_share.mean()),
        "unanimous_frac": float((mode_share == 1.0).mean()),
        # Reported because it is exactly when action_spearman goes undefined. Every gNB
        # holding one action for the whole episode is the extreme of the hypothesis, not
        # missing data, and a bare nan in the table reads as the opposite.
        "all_actions_constant": float(all(len(np.unique(actions[:, i])) < 2
                                          for i in range(n))),
        "action_spearman_mean": mean_pairwise(actions, "spearman"),
        # Collision storm predicts SINR falling across all gNB together rather than one at
        # a time, i.e. positively correlated series.
        "sinr_corr_mean": mean_pairwise(np.log10(np.maximum(sinr, 1e-12)), "pearson"),
        "timely_throughput_mbps": kpis["timely_throughput_mbps"],
        "embb_p5_mbps": kpis["embb_p5_mbps"],
        # Same episode seen twice, once by each code path. If the two ever disagree the
        # traces and the KPIs describe different episodes and every conditioned number
        # below is meaningless, so the gap is measured rather than assumed away.
        "path_agreement_gap": abs(kpis["embb_p5_mbps"] - ep["embb_p5_mbps"]),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", type=str,
                   default="results/logs/gnn-mappo_gat_v4_seed4*.pt,results/logs/ippo_v4_seed4*.pt",
                   help="comma-separated globs; the default is the collapsing model against "
                        "the one that does not collapse, which is the actual test")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--out", type=str, default="results/DIAG_COLLISION.md")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = sorted({Path(pt) for pattern in args.checkpoints.split(",")
                    for pt in glob.glob(pattern.strip())})
    if not paths:
        raise SystemExit(f"no checkpoints matched {args.checkpoints!r}")

    rows, episodes = [], []
    for pt_path in paths:
        algo, _, seed = parse_run_name(pt_path.stem)
        agent, ckpt_algo, kind = load_agent(pt_path, device)
        env = NetworkSlicingEnv()
        env.cmdp_enabled = False
        row = {"algo": algo, "seed": seed}
        for label, greedy in (("greedy", True), ("sampled", False)):
            per_ep = []
            for ep in range(args.episodes):
                eval_seed = EVAL_SEED_BASE + ep
                # The same episode is walked twice: once by run_episode for the KPIs, on
                # the same code path that produced every gate number, and once by rollout
                # for the per-step traces run_episode does not return. Re-seeding before
                # each makes them the same episode; `path_agreement_gap` checks that they
                # really were.
                torch.manual_seed(eval_seed)
                np.random.seed(eval_seed)
                kpis = run_episode(env, agent, kind, seed=eval_seed, greedy=greedy)
                torch.manual_seed(eval_seed)
                np.random.seed(eval_seed)
                stats = episode_stats(rollout(env, agent, kind, eval_seed, greedy), kpis)
                per_ep.append(stats)
                episodes.append({"algo": algo, "seed": seed, "arm": label,
                                 "episode": ep, **stats})
            for k in per_ep[0]:
                row[f"{label}_{k}"] = float(np.nanmean([e[k] for e in per_ep]))
        env.close()
        rows.append(row)
        print(f"[{pt_path.stem}] mode_share greedy {row['greedy_mode_share_mean']:.3f} vs "
              f"sampled {row['sampled_mode_share_mean']:.3f}  |  sinr_corr greedy "
              f"{row['greedy_sinr_corr_mean']:.3f} vs sampled {row['sampled_sinr_corr_mean']:.3f}")

    df = pd.DataFrame(rows)
    out_path = Path(args.out)
    df.to_csv(out_path.parent / f"{out_path.stem.lower()}.csv", index=False)

    eps = pd.DataFrame(episodes)
    eps.to_csv(out_path.parent / f"{out_path.stem.lower()}_episodes.csv", index=False)
    # Sampled-arm mean per checkpoint is the reference the collapse threshold is relative
    # to, so a checkpoint is only ever compared against itself.
    ref = (eps[eps.arm == "sampled"].groupby(["algo", "seed"])
           .timely_throughput_mbps.mean().rename("sampled_ref"))
    eps = eps.join(ref, on=["algo", "seed"])

    lines = [
        "# D5 -- collision-storm hypothesis\n",
        f"Checkpoints: `{args.checkpoints}`. Episodes per arm: {args.episodes}, seeds from "
        f"`EVAL_SEED_BASE = {EVAL_SEED_BASE}`.\n",
        "**This is a hypothesis under test, not an explanation.** PLAN-01 §Larangan 3 bars "
        "writing the collision-storm account into the paper before D5 confirms it, and "
        "records a limit already known: policy entropy spans only 0.100 nat while the "
        "readout gap is 46.28 Mbps, so whatever this is, it is not a simple function of "
        "entropy. If D5 does not confirm, the *operational validity limit* framing "
        "(PLAN-07 §4) still stands without it.\n",
        "**Both readouts appear here on purpose, and neither is the primary one.** D5 is "
        "about the difference between them: the greedy-vs-sampled contrast is the "
        "measurement. These are not KPI readings and must not be compared against "
        "`results/RLIABLE_*.md`.\n",
        "**What each column means.** `mode_share` is the mean fraction of the 5 gNB picking "
        "the same tier at a step (1.0 = unanimous); the hypothesis predicts it rising "
        "sharply under argmax for the collapsing model. `action_spearman` is the mean "
        "pairwise rank correlation between the five action series. `sinr_corr` is the mean "
        "pairwise Pearson correlation of the per-gNB log-SINR series -- a collision storm "
        "drops every gNB together, so it predicts high positive correlation, whereas "
        "independent per-cell degradation does not. Constant series are skipped rather than "
        "scored 0: a gNB pinned at one action carries no evidence either way.\n",
        "**`action_spearman` / `sinr_corr` read `--` when every gNB held one action for the "
        "whole episode.** Rank correlation is undefined on constant series, and that case "
        "is the *extreme* of the hypothesis rather than missing data -- a bare number there "
        "would be invented. `all const` marks it, and `mode_share` / `unanimous frac` carry "
        "the full answer for those rows.\n",
        "`embb_p5_mbps` is carried along only to show which arm actually collapsed.\n",
        "| algo | seed | arm | mode_share | unanimous frac | all const | action_spearman | sinr_corr | embb_p5 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    def cell(v: float) -> str:
        return "--" if np.isnan(v) else f"{v:.4f}"

    for r in rows:
        for arm in ("greedy", "sampled"):
            lines.append(
                f"| `{r['algo']}` | {r['seed']} | {arm} | {r[f'{arm}_mode_share_mean']:.4f} "
                f"| {r[f'{arm}_unanimous_frac']:.4f} | {r[f'{arm}_all_actions_constant']:.2f} "
                f"| {cell(r[f'{arm}_action_spearman_mean'])} "
                f"| {cell(r[f'{arm}_sinr_corr_mean'])} | {r[f'{arm}_embb_p5_mbps']:.6f} |"
            )

    lines.append("\n## Greedy minus sampled, per algorithm\n")
    lines.append(
        "`locked` counts the seeds whose greedy `mode_share` reached 0.998 or above -- the "
        "hypothesis's own prediction, stated as a count rather than hidden inside a mean. "
        "The Delta columns average only the seeds where the statistic is defined; `n=` says "
        "how many that was, so a mean taken over a subset is never mistaken for a mean over "
        "all seeds.\n")
    lines.append("| algo | seeds | locked (greedy) | Delta mode_share | Delta action_spearman | "
                 "Delta sinr_corr | Delta embb_p5 |")
    lines.append("|---|---|---|---|---|---|---|")
    for algo in sorted({r["algo"] for r in rows}):
        sub = [r for r in rows if r["algo"] == algo]
        locked = sum(r["greedy_mode_share_mean"] >= 0.998 for r in sub)
        cells = []
        for k in ("mode_share_mean", "action_spearman_mean", "sinr_corr_mean", "embb_p5_mbps"):
            d = np.array([r[f"greedy_{k}"] - r[f"sampled_{k}"] for r in sub])
            ok = ~np.isnan(d)
            fmt = "+.6f" if k == "embb_p5_mbps" else "+.4f"
            cells.append("--" if not ok.any() else
                         f"{format(float(d[ok].mean()), fmt)} (n={int(ok.sum())})")
        lines.append(f"| `{algo}` | {len(sub)} | {locked}/{len(sub)} | " + " | ".join(cells) + " |")

    # --- Conditioned on collapsed episodes -------------------------------------------
    # Everything above averages over ALL episodes. PLAN-01 D5's own evidence says the
    # episodes are bimodal (greedy sd 34.95/37.82 pp against sampled 11.04/10.45), and
    # averaging over a bimodal population is exactly how an intermittent effect hides.
    gap = float(eps.path_agreement_gap.max())
    lines += [
        "\n## Conditioned on collapsed episodes\n",
        "Everything above averages over **all** episodes. PLAN-01 D5's own evidence says "
        "the episodes are bimodal -- greedy sd 34.95/37.82 pp against sampled 11.04/10.45 "
        "-- and averaging over a bimodal population is exactly how an intermittent effect "
        "hides. The collision-storm hypothesis is specific to a policy that learned fine "
        "stochastic coordination, so the population it should be tested on is the episodes "
        "that actually collapsed.\n",
        f"**Collapse rule, declared before the numbers were read.** A greedy episode is "
        f"collapsed when its `timely_throughput_mbps` falls below **{COLLAPSE_FRAC:.0%}** of "
        f"its own checkpoint's sampled-arm mean. Per checkpoint, so nothing is compared "
        f"against another model.\n",
        "**This is not the project's cell-edge collapse.** That one is "
        "`embb_p5_mbps < 0.01` Mbps with the **seed** as the unit "
        "(`scripts/stability_report.py`), it is a gate metric, and it is untouched here. "
        "D5 is about the *greedy readout* collapse -- the 51.19 Mbps drop D5 cites -- which "
        "lives in `timely_throughput`. Applying the cell-edge rule here would select almost "
        "every *sampled* episode and almost no greedy one, the opposite of the population "
        "needed. The unit also changes from seed to episode; that is a weaker unit than the "
        "project fixes elsewhere and is named rather than swapped in quietly.\n",
        f"**Path agreement.** Each episode is walked twice -- `run_episode` for the KPIs, "
        f"`rollout` for the per-step traces -- under identical seeding. Largest "
        f"`embb_p5_mbps` disagreement between the two paths across all "
        f"{len(eps)} episodes: **{gap:.3e}**. A non-trivial gap would mean the traces and "
        f"the KPIs describe different episodes, and every number in this section would be "
        f"meaningless.\n",
        "**`sinr_corr` carries its own `n=`, and it is not the episode count.** When "
        "throughput pins near zero the per-gNB SINR series go constant, and a correlation "
        "on a constant series is undefined -- so the collapsed group, the very population "
        "this section exists to examine, is where the measure most often has nothing to "
        "say. A median printed without that count would rest on a fraction of the episodes "
        "while looking like it rested on all of them. `mode_share` is defined everywhere "
        "and needs no such caveat.\n",
        "| algo | greedy episodes | collapsed | sinr_corr collapsed | sinr_corr not-collapsed | "
        "mode_share collapsed | mode_share not-collapsed |",
        "|---|---|---|---|---|---|---|",
    ]
    greedy_eps = eps[eps.arm == "greedy"]
    for algo, g in greedy_eps.groupby("algo"):
        hit = g.timely_throughput_mbps < COLLAPSE_FRAC * g.sampled_ref
        inn, out = g[hit], g[~hit]

        def med(frame, col):
            v = frame[col].dropna()
            if v.empty:
                return f"-- (n=0/{len(frame)})"
            return f"{v.median():.4f} (n={len(v)}/{len(frame)})"
        lines.append(
            f"| `{algo}` | {len(g)} | {int(hit.sum())} | {med(inn, 'sinr_corr_mean')} | "
            f"{med(out, 'sinr_corr_mean')} | {med(inn, 'mode_share_mean')} | "
            f"{med(out, 'mode_share_mean')} |"
        )

    lines += [
        f"\n### Threshold sensitivity\n",
        "Mandatory, not optional: if the picture moves across these thresholds then the "
        "threshold is driving the result, and that is itself the finding rather than a "
        "reason to pick the most convenient one.\n",
        "| threshold | algo | collapsed | sinr_corr collapsed | mode_share collapsed |",
        "|---|---|---|---|---|",
    ]
    for frac in sorted((COLLAPSE_FRAC, *SENSITIVITY_FRACS)):
        for algo, g in greedy_eps.groupby("algo"):
            hit = g.timely_throughput_mbps < frac * g.sampled_ref
            inn = g[hit]
            sc = inn.sinr_corr_mean.dropna()
            ms = inn.mode_share_mean.dropna()
            lines.append(
                f"| {frac:.0%} | `{algo}` | {int(hit.sum())}/{len(g)} | "
                f"{'--' if sc.empty else f'{sc.median():.4f}'} (n={len(sc)}) | "
                f"{'--' if ms.empty else f'{ms.median():.4f}'} (n={len(ms)}) |"
            )

    lines.append(
        f"\nEvery episode is in `{out_path.stem.lower()}_episodes.csv` ({len(eps)} rows) "
        "with its own throughput and synchrony, so the conditioning above can be recomputed "
        "at any other threshold straight from that file.\n")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
