"""Freeze f_min for the per-gNB resilient constraint (docs/revisi/PLAN-02 section 7).

Too low and the constraint never binds -- the v3 failure, where lambda went vacuous. Too
high and it is infeasible -- the calibration-round-3 failure. The protocol follows the
Gate A pattern that already worked, and splits along who can run what:

  --sweep   steps 1-2, CPU only. Measure the per-gNB eMBB rate distribution reachable at
            the frozen operating point and propose a candidate f_min. Two populations:
            static allocations by default, and trained v4 checkpoints with --checkpoints.

Measured 2026-08-25, and it decides which population to use: NO static fraction satisfies
delta=0.085. The lowest violation any static allocation reaches is 0.0964, at frac=0.9.
That is structural rather than noise -- delta was calibrated against a TRAINED policy (see
the cmdp block in configs/experiment_config.yaml: static floor 12.22%, trained floor
~5.5%), which is the same lesson delta's own calibration round 4 learned. So the static
sweep is kept for the record and reports "no candidate" honestly, and the candidate is
taken from trained checkpoints, whose violation actually sits under delta.
  --check   steps 3-4, reads the metrics CSV of an ippo reference trained at the candidate
            and prints the two gates. The training itself runs on GPU, by hand
            (docs/HANDOVER.md section 11); this script never trains.

NOT the same quantity as the PRB floor. env._compute_floor's f_min is an allocation
FRACTION (floor.base/k/lo/hi); resilient.f_min_mbps is a RATE floor in Mbps. Same word,
different units, different mechanism.

The candidate rule is declared here rather than chosen after seeing the numbers:

  Among static URLLC fractions whose mean violation satisfies the frozen delta, take the
  one with the highest mean eMBB rate -- the best feasible operating point a static policy
  reaches -- and propose the 25th percentile of its per-gNB rate distribution, computed
  over non-collapsed seeds only.

"Non-collapsed" is the project's existing cell-edge rule, embb_p5_mbps < 0.01 Mbps with the
seed as the unit (scripts/stability_report.py), reused rather than redefined. A collapsed
seed's rates sit at the floor and would drag the percentile toward zero, which is exactly
the "too low to bind" failure the protocol is trying to avoid.

Usage:
  python scripts/calibrate_fmin.py --sweep
  python scripts/calibrate_fmin.py --check results/logs/ippo_fmincal_seed42.csv
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.network_slicing_env import NetworkSlicingEnv
from scripts.evaluate_checkpoints import EVAL_SEED_BASE, load_agent, select_actions
from scripts.rliable_report import parse_run_name

FRACTIONS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
COLLAPSE_P5_MBPS = 0.01     # scripts/stability_report.py, unit = seed. Not redefined here.
CANDIDATE_PERCENTILE = 25   # PLAN-02 section 7 step 2

# PLAN-02 section 7 step 3 names the reference: a NON-GNN baseline, ippo, consistent with
# Gate A. This is enforced rather than left to the ranking rule, because "highest mean eMBB
# among feasible policies" run over all checkpoints picks gnn-mappo -- which would set
# f_min from the proposed method's own reachable distribution and bake its advantage into
# the constraint. That is precisely what Larangan 1 forbids: f_min must never be chosen by
# which algorithm it favours.
REFERENCE_ALGOS = ("ippo",)

# No candidate may rest on a single run. Declared here, before the numbers: a percentile
# over one non-collapsed seed is an n=1 estimate wearing the clothes of a distribution.
MIN_NONCOLLAPSED = 3


def sweep_one(env: NetworkSlicingEnv, frac: float, seed: int, steps: int) -> dict:
    """One static-allocation episode. The action tier closest to `frac` is used, so the
    sweep exercises the same discrete action space the policies do."""
    tier = int(round(frac * (env.n_tiers - 1)))
    env.reset(seed=seed)
    rates, viol = [], []
    for _ in range(steps):
        _, _, _, truncated, info = env.step([tier] * env.n_gnb)
        rates.append(np.asarray(info["embb_thr_bps"], dtype=np.float64))
        viol.append(float(np.mean(info["urllc_violation_rate"])))
        if truncated:
            env.reset(seed=seed)
    r = np.stack(rates)
    return {"frac": frac, "seed": seed,
            "embb_mean_mbps": float(r.mean() / 1e6),
            "embb_p5_mbps": float(np.percentile(r, 5) / 1e6),
            "violation": float(np.mean(viol)),
            "rates_bps": r}


def sweep_checkpoint(env: NetworkSlicingEnv, pt_path: Path, steps: int, seed: int) -> dict:
    """Same measurement as sweep_one, driven by a trained policy instead of a fixed tier.

    Held-out seeds (EVAL_SEED_BASE), and the per-family primary readout -- sampled for PPO
    (P3), argmax for DQN (2026-08-16) -- because a policy read the wrong way produces a
    rate distribution that is not the one it would deliver.
    """
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent, algo, kind = load_agent(pt_path, device)
    greedy = "dqn" in algo
    obs, info = env.reset(seed=seed)
    rates, viol = [], []
    for _ in range(steps):
        actions = select_actions(agent, kind, obs, info, env, greedy)
        obs, _, _, truncated, info = env.step(actions)
        rates.append(np.asarray(info["embb_thr_bps"], dtype=np.float64))
        viol.append(float(np.mean(info["urllc_violation_rate"])))
        if truncated:
            obs, info = env.reset(seed=seed)
    r = np.stack(rates)
    return {"frac": algo, "seed": parse_run_name(pt_path.stem)[2],
            "embb_mean_mbps": float(r.mean() / 1e6),
            "embb_p5_mbps": float(np.percentile(r, 5) / 1e6),
            "violation": float(np.mean(viol)),
            "readout": "argmax" if greedy else "sampled",
            "rates_bps": r}


def run_sweep(seeds: list[int], steps: int, out: Path, checkpoints: str | None = None) -> None:
    env = NetworkSlicingEnv()
    if env.resilient_mode != "none":
        raise SystemExit("run the sweep with resilient.mode=none -- calibrating f_min "
                         "against a policy already shaped by f_min is circular")
    delta = env.delta

    rows, pooled = [], {}
    if checkpoints:
        import glob as _glob
        paths = sorted(Path(p) for p in _glob.glob(checkpoints))
        if not paths:
            raise SystemExit(f"no checkpoints matched {checkpoints!r}")
        by_algo: dict[str, list] = {}
        for i, pt in enumerate(paths):
            s = sweep_checkpoint(env, pt, steps, EVAL_SEED_BASE + i)
            rows.append({k: v for k, v in s.items() if k != "rates_bps"})
            if s["embb_p5_mbps"] >= COLLAPSE_P5_MBPS:
                by_algo.setdefault(s["frac"], []).append(s["rates_bps"])
        pooled = {a: np.concatenate([k.ravel() for k in v]) for a, v in by_algo.items()}
        population = f"trained checkpoints `{checkpoints}`"
    else:
        for frac in FRACTIONS:
            keep = []
            for seed in seeds:
                s = sweep_one(env, frac, seed, steps)
                rows.append({k: v for k, v in s.items() if k != "rates_bps"})
                if s["embb_p5_mbps"] >= COLLAPSE_P5_MBPS:
                    keep.append(s["rates_bps"])
            pooled[frac] = np.concatenate([k.ravel() for k in keep]) if keep else np.array([])
        population = f"static allocations, seeds {seeds}"
    floor_mode = env.floor_mode
    env.close()

    df = pd.DataFrame(rows)
    agg = df.groupby("frac").agg(embb_mean_mbps=("embb_mean_mbps", "mean"),
                                 embb_p5_mbps=("embb_p5_mbps", "mean"),
                                 violation=("violation", "mean")).reset_index()
    agg["n_noncollapsed"] = [int(sum(1 for s in rows if s["frac"] == f
                                     and s["embb_p5_mbps"] >= COLLAPSE_P5_MBPS))
                             for f in agg.frac]
    agg["feasible"] = agg.violation <= delta

    agg["is_reference"] = [True if not checkpoints else (a in REFERENCE_ALGOS)
                           for a in agg.frac]
    feasible = agg[agg.feasible & (agg.n_noncollapsed >= MIN_NONCOLLAPSED) & agg.is_reference]
    lines = [
        "# f_min calibration -- steps 1-2\n",
        f"Population: {population}. {steps} steps each, frozen operating point "
        f"(`delta={delta}`, `floor.mode={floor_mode}`). The constraint is switched off for "
        "this sweep: calibrating `f_min` against a policy already shaped by `f_min` would "
        "be circular.\n",
        "**Not the PRB floor.** `env._compute_floor`'s `f_min` is an allocation *fraction*; "
        "`resilient.f_min_mbps` is a *rate* floor in Mbps. Same word, different units.\n",
        "**Rule, declared before the numbers were read.** Among candidate policies "
        f"whose mean violation satisfies `delta = {delta}`, take the one with the highest "
        f"mean eMBB rate, and propose the {CANDIDATE_PERCENTILE}th percentile of its "
        "per-gNB rate distribution over **non-collapsed seeds only**. Collapse is the "
        f"project's existing cell-edge rule, `embb_p5_mbps < {COLLAPSE_P5_MBPS}` with the "
        "seed as the unit (`scripts/stability_report.py`) -- reused, not redefined. A "
        "collapsed seed sits at the floor and would drag the percentile toward zero, which "
        "is the *too low to bind* failure this protocol exists to avoid.\n",
        "**Only the reference family is eligible.** PLAN-02 section 7 step 3 names a "
        f"non-GNN baseline -- `{'`, `'.join(REFERENCE_ALGOS)}`, consistent with Gate A -- "
        "and that is enforced here rather than left to the ranking rule. Ranking every "
        "checkpoint by mean eMBB picks `gnn-mappo`, which would set `f_min` from the "
        "proposed method's own reachable distribution and bake its advantage into the "
        "constraint. Larangan 1 forbids exactly that. Non-reference rows are shown for "
        f"context and marked. A candidate also needs at least {MIN_NONCOLLAPSED} "
        "non-collapsed runs, so no percentile can rest on a single seed.\n",
        "| policy | eMBB mean (Mbps) | eMBB p5 (Mbps) | violation | <= delta | non-collapsed | eligible |",
        "|---|---|---|---|---|---|---|",
    ]
    n_per = df.groupby("frac").size().to_dict()
    for _, r in agg.iterrows():
        label = f"{r.frac:.1f}" if isinstance(r.frac, float) else f"`{r.frac}`"
        lines.append(f"| {label} | {r.embb_mean_mbps:.3f} | {r.embb_p5_mbps:.4f} | "
                     f"{r.violation:.4f} | {'yes' if r.feasible else 'no'} | "
                     f"{int(r.n_noncollapsed)}/{n_per[r.frac]} | "
                     f"{'reference' if r.is_reference else 'context only'} |")

    if feasible.empty:
        lines.append(
            f"\n## No candidate\n\nNothing eligible both satisfies `delta = {delta}` and "
            f"leaves at least {MIN_NONCOLLAPSED} non-collapsed runs. That is a finding "
            "about the operating point, not a number to soften: report it and do not pick "
            "an `f_min` anyway.\n\n"
            "For the **static** population this is the expected outcome -- `delta` was "
            "calibrated against a trained policy, so re-run with `--checkpoints`.\n\n"
            "For the **trained** population it is a real blocker and belongs in front of a "
            "human. The reference family cannot currently satisfy the constraint and avoid "
            "cell-edge collapse at the same time, which is the same condition that leaves "
            "the project at NOT DONE (Gate B3 fails, C4 fails for the DQN family). "
            "Calibrating a rate floor against a reference that itself collapses is not "
            "possible; either the operating point moves, or the reference family for this "
            "calibration is re-declared -- and both are decisions for a human, recorded in "
            "the ledger before anything is frozen.\n")
    else:
        best = feasible.loc[feasible.embb_mean_mbps.idxmax()]
        rates = pooled[best.frac]
        cand = float(np.percentile(rates, CANDIDATE_PERCENTILE) / 1e6)
        lines += [
            "\n## Candidate\n",
            f"Best feasible policy: **{best.frac}** (eMBB mean "
            f"{best.embb_mean_mbps:.3f} Mbps, violation {best.violation:.4f} <= {delta}), "
            f"pooled over {int(best.n_noncollapsed)} non-collapsed runs "
            f"({len(rates)} gNB-slot samples).\n",
            f"**Candidate `f_min_mbps` = {cand:.4f}**\n",
            "Not frozen by this script. Freeze it only after the two gates below pass, and "
            "record it in the ledger before the wave (PLAN-02 section 7; Larangan 1: never "
            "chosen by which algorithm it favours).\n",
            "## Steps 3-4 -- run by hand, on GPU\n",
            "```",
            f"# set resilient.mode=fixed and resilient.f_min_mbps={cand:.4f} in the config",
            "python scripts/run_wave.py --algos ippo --seeds 42 --resilient fixed",
            "python scripts/calibrate_fmin.py --check results/logs/<that run>.csv",
            "```\n",
            "`ippo` is the reference, consistent with Gate A. The check reads `mu_mean` and "
            "`resilient_shortfall_mean` from the metrics CSV and prints both gates.\n",
        ]

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    df.to_csv(out.parent / f"{out.stem.lower()}.csv", index=False)
    print(f"wrote {out}")


def run_check(csv_path: Path, tail_frac: float) -> None:
    """Steps 3-4: the two gates of PLAN-02 section 7, read off a trained run's metrics."""
    df = pd.read_csv(csv_path)
    for col in ("mu_mean", "resilient_shortfall_mean", "embb_p5_mbps"):
        if col not in df.columns:
            raise SystemExit(f"{csv_path} has no `{col}` column -- it predates the "
                             "resilient constraint, or the run had resilient.mode=none")
    n = len(df)
    tail = df.iloc[int(n * (1 - tail_frac)):]
    head = df.iloc[:max(int(n * tail_frac), 1)]

    # FEASIBLE: the shortfall must settle, not grow monotonically. Compared head against
    # tail rather than by a slope fit, because a slope is dominated by the transient.
    short_head = float(head.resilient_shortfall_mean.mean())
    short_tail = float(tail.resilient_shortfall_mean.mean())
    feasible = short_tail <= short_head

    # BINDING: a steady-state mu above zero, AND embb_p5 moved by at least one window std
    # between the mu=0 rows and the mu>0 rows. A positive mu that moves nothing is a
    # constraint that is priced but not felt.
    mu_tail = float(tail.mu_mean.mean())
    off, on = df[df.mu_mean <= 0.0], df[df.mu_mean > 0.0]
    if len(off) and len(on):
        window_std = float(df.embb_p5_mbps.std())
        moved = abs(float(on.embb_p5_mbps.mean() - off.embb_p5_mbps.mean()))
        binding = mu_tail > 0.0 and window_std > 0 and moved >= window_std
    else:
        window_std = moved = float("nan")
        binding = False

    print(f"rows {n}, tail = last {tail_frac:.0%}")
    print(f"  FEASIBLE : shortfall head {short_head:.6f} -> tail {short_tail:.6f}  "
          f"[{'PASS' if feasible else 'FAIL'}]")
    print(f"  BINDING  : mu tail {mu_tail:.6f}, embb_p5 moved {moved:.6f} against window "
          f"std {window_std:.6f}  [{'PASS' if binding else 'FAIL'}]")
    if not len(off) or not len(on):
        print("             (no rows on one side of mu=0, so the movement half is "
              "undefined -- reported as FAIL rather than skipped)")
    print(f"VERDICT: {'freeze f_min' if feasible and binding else 'adjust f_min and repeat'}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", action="store_true", help="steps 1-2, CPU")
    p.add_argument("--check", type=str, default=None,
                   help="steps 3-4, metrics CSV of an ippo reference run")
    p.add_argument("--seeds", type=str, default="42,43,44,45,46")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--tail-frac", type=float, default=0.25)
    p.add_argument("--checkpoints", type=str, default=None,
                   help="glob of trained checkpoints; without it the sweep uses static "
                        "allocations, which cannot reach delta at this operating point")
    p.add_argument("--out", type=str, default="results/CALIBRATE_FMIN.md")
    args = p.parse_args()

    if args.check:
        run_check(Path(args.check), args.tail_frac)
    elif args.sweep:
        run_sweep([int(s) for s in args.seeds.split(",")], args.steps, Path(args.out),
                  args.checkpoints)
    else:
        p.error("pass --sweep or --check")


if __name__ == "__main__":
    main()
