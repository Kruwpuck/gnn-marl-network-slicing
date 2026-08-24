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
from scripts.evaluate_checkpoints import EVAL_SEED_BASE, load_agent, select_actions
from scripts.rliable_report import parse_run_name


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


def episode_stats(ep: dict) -> dict:
    """Agreement and synchrony for one episode."""
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
        "embb_p5_mbps": ep["embb_p5_mbps"],
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

    rows = []
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
                torch.manual_seed(eval_seed)
                np.random.seed(eval_seed)
                per_ep.append(episode_stats(rollout(env, agent, kind, eval_seed, greedy)))
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

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
