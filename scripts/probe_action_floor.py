"""
Measure the LOWEST URLLC violation each action space can physically reach, with no
learning involved. This is what decides whether cmdp.delta is feasible at all.

Why this exists: Gate A round 3 (2026-08-06) passed A2 vacuously -- lambda rose 18x
while the policy did not move. The cause was not a lazy dual; it was an infeasible
constraint. delta=0.05 sits BELOW the best violation the reference baseline's action
space can produce, so `violation - delta` stays positive forever and the Lagrangian
integrates without a fixed point. That is correct dual behaviour on an infeasible
constraint, not a tuning failure.

scripts/calibrate_load.py, diag_traffic_sweep.py and diag_breakdown.py all sweep
UNIFORM allocations (one tier broadcast to every gNB), which is exactly the action
space of `central-ppo`/`central-dqn`. The per-gNB algorithms (ippo/idqn/gnn-*) give
each gNB its own tier, so their floor is lower -- cell-edge gNBs can be fed more PRB
than interior ones. This script measures both:

  * uniform floor  -- full tier grid, all gNB on the same tier (central-* space)
  * per-gNB floor  -- coordinate descent over per-gNB tiers (ippo/idqn/gnn-* space)

delta must sit ABOVE the floor of the reference baseline's action space by more than
the dual-update window std (1.62pp measured on central-ppo_calib3_seed42), or the
constraint is infeasible and the dual cannot reach a fixed point.

Violation is reported as the quantity the CMDP actually constrains: mean of the
per-gNB violation rate over gNBs, averaged over steps (env.step ->
`mean_violation_rate`, compared against delta in the dual update). The pooled
arrival-weighted rate is printed alongside because the two differ when load or
violation is uneven across gNBs (see diag_breakdown.py) -- and a per-gNB allocation
deliberately makes them uneven.

Same measurement conditions as calibrate_load.py: cmdp disabled (no lambda feedback),
floor_mode="none" (raw traffic/deadline feasibility, not the floor mitigation), and
identical episode seeds for every candidate so allocations are compared on common
random numbers.

Usage:
  python scripts/probe_action_floor.py
  python scripts/probe_action_floor.py --episodes 20 --rounds 3   # tighter, slower
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from envs.network_slicing_env import NetworkSlicingEnv


def eval_tiers(env: NetworkSlicingEnv, tiers: list[int], episodes: int, seed0: int = 0) -> dict:
    """Violation of a fixed per-gNB tier vector. Common random numbers across calls."""
    arrived = late = overflow = 0
    step_rates: list[float] = []
    embb_mbps: list[float] = []
    for ep in range(episodes):
        env.reset(seed=seed0 + ep)
        done = False
        while not done:
            _, _, terminated, truncated, info = env.step(list(tiers))
            done = terminated or truncated
            step_rates.append(float(np.mean(info["urllc_violation_rate"])))
            arrived += int(np.sum(info["urllc_arrived"]))
            late += int(np.sum(info["urllc_dropped_late"]))
            overflow += int(np.sum(info["urllc_dropped_overflow"]))
            embb_mbps.extend((np.asarray(info["embb_rates"]) / 1e6).tolist())
    arrived = max(arrived, 1)
    return {"cmdp_viol_pct": 100.0 * float(np.mean(step_rates)),
            "pooled_viol_pct": 100.0 * (late + overflow) / arrived,
            "late_pct": 100.0 * late / arrived,
            "overflow_pct": 100.0 * overflow / arrived,
            "embb_mbps": float(np.mean(embb_mbps)) if embb_mbps else 0.0}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=40, help="episodes per candidate during the sweep")
    p.add_argument("--final-episodes", type=int, default=60, help="re-measure the winners tighter")
    p.add_argument("--rounds", type=int, default=2, help="coordinate-descent sweeps over the gNBs")
    p.add_argument("--final-seed0", type=int, default=10_000,
                   help="seeds for the final re-measure, DISJOINT from the sweep seeds. The first "
                        "run of this probe (10 episodes) reported a per-gNB floor ABOVE the uniform "
                        "floor, which is impossible -- the per-gNB action set contains every uniform "
                        "vector -- so it was descent overfitting to its own episode sample. Scoring "
                        "the winner on held-out seeds is what makes the number reportable "
                        "(goal1.md C5 uses seed >= 10000 for the same reason)")
    p.add_argument("--config", type=str, default=None)
    args = p.parse_args()

    env = NetworkSlicingEnv(config_path=args.config)
    env.cmdp_enabled = False
    env.floor_mode = "none"
    n_gnb, n_tiers = env.n_gnb, env.n_tiers

    print(f"n_gnb={n_gnb} n_tiers={n_tiers} delta={env.delta:.4f} episodes={args.episodes} "
          f"(cmdp off, floor_mode=none)")

    # 1. uniform sweep = central-* action space, full grid (calibrate_load stops at frac 0.8)
    print(f"\n{'tier':>5}{'frac':>7}{'cmdp%':>9}{'pooled%':>9}{'late%':>9}{'ovfl%':>9}{'eMBB':>9}")
    uni = []
    for t in range(n_tiers):
        r = eval_tiers(env, [t] * n_gnb, args.episodes)
        uni.append((t, r))
        print(f"{t:>5}{t/(n_tiers-1):>7.2f}{r['cmdp_viol_pct']:>9.2f}{r['pooled_viol_pct']:>9.2f}"
              f"{r['late_pct']:>9.2f}{r['overflow_pct']:>9.2f}{r['embb_mbps']:>9.2f}")
    t_uni, r_uni = min(uni, key=lambda x: x[1]["cmdp_viol_pct"])

    # 2. coordinate descent from the uniform optimum = per-gNB action space
    tiers = [t_uni] * n_gnb
    best = r_uni["cmdp_viol_pct"]
    for rnd in range(args.rounds):
        moved = False
        for g in range(n_gnb):
            for t in range(n_tiers):
                if t == tiers[g]:
                    continue
                cand = list(tiers)
                cand[g] = t
                v = eval_tiers(env, cand, args.episodes)["cmdp_viol_pct"]
                if v < best - 1e-9:
                    best, tiers, moved = v, cand, True
        print(f"round {rnd+1}: tiers={tiers} cmdp_viol={best:.2f}%")
        if not moved:
            break

    # 3. re-measure both winners on DISJOINT seeds -- the descent minimised over its own
    #    episode sample, so its reported minimum is optimistically biased
    fin_uni = eval_tiers(env, [t_uni] * n_gnb, args.final_episodes, seed0=args.final_seed0)
    fin_per = eval_tiers(env, tiers, args.final_episodes, seed0=args.final_seed0)
    env.close()

    print(f"\nFLOOR (episodes={args.final_episodes}, held-out seeds {args.final_seed0}+, "
          f"cmdp off, floor_mode=none)")
    print(f"  uniform  tier={t_uni} ({t_uni/(n_tiers-1):.2f})   cmdp={fin_uni['cmdp_viol_pct']:.2f}% "
          f"pooled={fin_uni['pooled_viol_pct']:.2f}%  eMBB={fin_uni['embb_mbps']:.2f} Mbps"
          f"   <- central-ppo / central-dqn")
    print(f"  per-gNB  tiers={tiers}   cmdp={fin_per['cmdp_viol_pct']:.2f}% "
          f"pooled={fin_per['pooled_viol_pct']:.2f}%  eMBB={fin_per['embb_mbps']:.2f} Mbps"
          f"   <- ippo / idqn / gnn-*")
    print("\n  Coordinate descent gives an UPPER bound on the true per-gNB floor (single-gNB "
          "moves only), and the per-gNB floor is <= the uniform floor by construction. If the "
          "per-gNB number comes out HIGHER, the descent overfit its seeds -- raise --episodes.")
    print("  delta must exceed the reference baseline's floor by more than the dual-update "
          "window std (1.62pp), and the floor that matters is the one reachable WITHOUT "
          "starving eMBB: check the eMBB column, frac=1.00 lowers violation only by handing "
          "every PRB to URLLC, which r_obj penalises hard.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
