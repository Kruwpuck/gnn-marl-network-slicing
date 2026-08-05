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
    """Violation of a fixed per-gNB tier vector. Common random numbers across calls.

    Reports the standard error over EPISODES, not over steps: episodes are the
    independent unit (each reset redraws the topology and the UE positions) and the
    between-episode spread is what actually limits how precisely delta can be placed.
    """
    arrived = late = overflow = 0
    ep_viol: list[float] = []
    embb_mbps: list[float] = []
    for ep in range(episodes):
        env.reset(seed=seed0 + ep)
        done = False
        step_rates: list[float] = []
        while not done:
            _, _, terminated, truncated, info = env.step(list(tiers))
            done = terminated or truncated
            step_rates.append(float(np.mean(info["urllc_violation_rate"])))
            arrived += int(np.sum(info["urllc_arrived"]))
            late += int(np.sum(info["urllc_dropped_late"]))
            overflow += int(np.sum(info["urllc_dropped_overflow"]))
            embb_mbps.extend((np.asarray(info["embb_rates"]) / 1e6).tolist())
        ep_viol.append(100.0 * float(np.mean(step_rates)))
    arrived = max(arrived, 1)
    v = np.asarray(ep_viol)
    return {"cmdp_viol_pct": float(v.mean()),
            "se_pp": float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan"),
            "ep_std_pp": float(v.std(ddof=1)) if len(v) > 1 else float("nan"),
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
    p.add_argument("--floor-mode", type=str, default=None, choices=["none", "static", "dynamic"],
                   help="default: whatever the config says, i.e. the regime the wave actually runs "
                        "in. floor=none measures the RAW traffic/deadline feasibility region "
                        "(what calibrate_load.py reports) -- useful as a reference, but a floor "
                        "measured there is NOT the floor Gate A2a needs: the PRB floor projection "
                        "lets a trained policy reach violations below it")
    p.add_argument("--seed0", type=int, default=0, help="seeds for the sweep / descent")
    p.add_argument("--lambda-arrival", type=float, default=None,
                   help="override traffic.urllc.lambda_arrival for this probe only (nothing is "
                        "written to the config). Used to find a load where the action actually "
                        "moves the violation: at 25000 with floor=dynamic the whole reachable "
                        "range is 3.66-5.26%%, so no policy can steer the constraint")
    p.add_argument("--tiers", type=str, default=None,
                   help="comma-separated tier subset, e.g. 1,3,5,7,9 -- for cheap load sweeps")
    p.add_argument("--urllc-max-bits", type=float, default=None,
                   help="override buffer.urllc_max_bits for this probe only. Raising the load "
                        "WITHOUT raising this makes drops overflow-dominated and fails Gate A3: "
                        "the config rule is buffer = 2x delay-bandwidth product = 2 * lambda * "
                        "packet_size_bits * max_delay_s. Note the coupling -- urllc_max_bits is "
                        "also q_ref in env._compute_floor(), so scaling it with the load also "
                        "weakens the dynamic floor, which is what gives the action leverage back")
    p.add_argument("--dbp-buffer", type=float, default=None, metavar="K",
                   help="shorthand: set urllc_max_bits = K * delay-bandwidth product at the "
                        "probed load. --dbp-buffer 2 reproduces the rule the config already uses")
    p.add_argument("--descent", action="store_true",
                   help="run coordinate descent for the per-gNB floor. Off by default: on held-out "
                        "seeds it bought 0.01pp over the uniform floor (5.33 vs 5.34), so it costs "
                        "~10x the episodes for nothing. The per-gNB action space pays off in eMBB "
                        "at equal violation (2.77 vs 0.00 Mbps), not in the floor itself")
    args = p.parse_args()

    env = NetworkSlicingEnv(config_path=args.config)
    env.cmdp_enabled = False
    if args.floor_mode is not None:
        env.floor_mode = args.floor_mode
    if args.lambda_arrival is not None:
        # reset() rebuilds the PoissonTraffic generators from this attribute, so setting it
        # here is enough -- every eval_tiers call resets first
        env.urllc_lambda = args.lambda_arrival
    if args.dbp_buffer is not None:
        dbp = env.urllc_lambda * env.urllc_pkt_bits * (env.urllc_max_delay_ms / 1000.0)
        env.urllc_max_bits = args.dbp_buffer * dbp
    if args.urllc_max_bits is not None:
        env.urllc_max_bits = args.urllc_max_bits

    n_gnb, n_tiers = env.n_gnb, env.n_tiers
    tiers_to_sweep = ([int(t) for t in args.tiers.split(",")] if args.tiers
                      else list(range(n_tiers)))
    print(f"n_gnb={n_gnb} n_tiers={n_tiers} delta={env.delta:.4f} "
          f"lambda_arrival={env.urllc_lambda:.0f} urllc_max_bits={env.urllc_max_bits:.0f} "
          f"episodes={args.episodes} seeds {args.seed0}+ "
          f"(cmdp off, floor_mode={env.floor_mode})")

    # 1. uniform sweep = central-* action space, full grid (calibrate_load stops at frac 0.8)
    print(f"\n{'tier':>5}{'frac':>7}{'cmdp%':>9}{'+-SE':>7}{'epStd':>8}{'pooled%':>9}"
          f"{'late%':>8}{'ovfl%':>8}{'eMBB':>8}")
    uni = []
    for t in tiers_to_sweep:
        r = eval_tiers(env, [t] * n_gnb, args.episodes, seed0=args.seed0)
        uni.append((t, r))
        print(f"{t:>5}{t/(n_tiers-1):>7.2f}{r['cmdp_viol_pct']:>9.2f}{r['se_pp']:>7.2f}"
              f"{r['ep_std_pp']:>8.2f}{r['pooled_viol_pct']:>9.2f}"
              f"{r['late_pct']:>8.2f}{r['overflow_pct']:>8.2f}{r['embb_mbps']:>8.2f}")
    t_uni, r_uni = min(uni, key=lambda x: x[1]["cmdp_viol_pct"])

    if not args.descent:
        fin = eval_tiers(env, [t_uni] * n_gnb, args.final_episodes, seed0=args.final_seed0)
        env.close()
        floor_mode = env.floor_mode  # read before close(); args.floor_mode is None when inherited
        print(f"\nFLOOR uniform tier={t_uni} ({t_uni/(n_tiers-1):.2f}), held-out seeds "
              f"{args.final_seed0}+, {args.final_episodes} episodes, floor_mode={floor_mode}: "
              f"{fin['cmdp_viol_pct']:.2f}% +-{fin['se_pp']:.2f}pp  eMBB={fin['embb_mbps']:.2f} Mbps")
        print("  The per-gNB floor is <= this by construction (its action set contains every "
              "uniform vector); --descent measures how much lower, and on held-out seeds that "
              "was 0.01pp.")
        return 0

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
                v = eval_tiers(env, cand, args.episodes, seed0=args.seed0)["cmdp_viol_pct"]
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
