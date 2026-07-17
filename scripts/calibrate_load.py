"""
Gate before the big training wave: run static (untrained) URLLC-fraction policies
against the packet-level env and check the traffic/buffer/deadline parameters put
the system in a genuinely contested regime — not trivially easy (violation ~0%)
and not infeasible (violation ~100% everywhere).

Usage:
  python scripts/calibrate_load.py [--episodes 20] [--config configs/experiment_config.yaml]

Exit code 0 = pass (safe to proceed to the training wave), 1 = fail (retune traffic).
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.network_slicing_env import NetworkSlicingEnv

STATIC_FRACS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8]


def run_static_policy(config_path: str | None, urllc_frac: float, episodes: int, seed0: int = 0) -> dict:
    env = NetworkSlicingEnv(config_path=config_path)
    env.cmdp_enabled = False  # pure measurement run, no dual updates / lambda noise
    env.floor_mode = "none"   # measure the RAW traffic/deadline feasibility region,
                               # independent of the floor mitigation (that's a separate lever)
    tier = int(round(urllc_frac * (env.n_tiers - 1)))

    arrived = dropped_late = dropped_overflow = 0
    delays: list[float] = []
    embb_mbps: list[float] = []
    delivered_bits_total = 0.0
    n_steps = 0

    for ep in range(episodes):
        env.reset(seed=seed0 + ep)
        done = False
        while not done:
            _, _, terminated, truncated, info = env.step([tier] * env.n_gnb)
            done = terminated or truncated
            arrived += int(np.sum(info["urllc_arrived"]))
            dropped_late += int(np.sum(info["urllc_dropped_late"]))
            dropped_overflow += int(np.sum(info["urllc_dropped_overflow"]))
            for gnb_delays in info["urllc_delivered_delays"]:
                delays.extend(gnb_delays)
            delivered_bits_total += float(np.sum(info["urllc_delivered_bits"]))
            embb_mbps.extend((np.asarray(info["embb_rates"]) / 1e6).tolist())
            n_steps += 1
    env.close()

    arrived = max(arrived, 1)
    delay_arr = np.asarray(delays) if delays else np.array([0.0])
    return {
        "urllc_frac": urllc_frac,
        "violation_pct": 100.0 * (dropped_late + dropped_overflow) / arrived,
        "drop_late_pct": 100.0 * dropped_late / arrived,
        "drop_overflow_pct": 100.0 * dropped_overflow / arrived,
        "delay_p95_ms": float(np.percentile(delay_arr, 95)),
        "delay_p99_ms": float(np.percentile(delay_arr, 99)),
        "embb_mbps_mean": float(np.mean(embb_mbps)) if embb_mbps else 0.0,
        "timely_throughput_mbps": delivered_bits_total / max(n_steps * env.slot_s, env.slot_s) / 1e6,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--delta", type=float, default=None,
                   help="override cmdp.delta from config for the feasibility check")
    args = p.parse_args()

    env_tmp = NetworkSlicingEnv(config_path=args.config)
    delta = args.delta if args.delta is not None else env_tmp.delta
    deadline_ms = env_tmp.urllc_max_delay_ms
    env_tmp.close()

    rows = [run_static_policy(args.config, f, args.episodes) for f in STATIC_FRACS]

    header = f"{'frac':>6}{'viol%':>10}{'late%':>10}{'ovfl%':>10}{'p95ms':>10}{'p99ms':>10}{'eMBB Mbps':>12}{'timely Mbps':>13}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['urllc_frac']:>6.2f}{r['violation_pct']:>10.2f}{r['drop_late_pct']:>10.2f}"
              f"{r['drop_overflow_pct']:>10.2f}{r['delay_p95_ms']:>10.3f}{r['delay_p99_ms']:>10.3f}"
              f"{r['embb_mbps_mean']:>12.2f}{r['timely_throughput_mbps']:>13.3f}")

    ok = True
    reasons = []

    contested = [r for r in rows if 5.0 <= r["violation_pct"] <= 30.0]
    if not contested:
        ok = False
        reasons.append("No static policy lands in the 5-30% violation regime "
                        "(either everything is trivially easy or everything is infeasible).")

    r0 = rows[0]  # urllc_frac = 0.0
    if r0["violation_pct"] < 90.0:
        ok = False
        reasons.append(f"0% URLLC allocation should starve almost totally (expected >=90% "
                        f"violation), got {r0['violation_pct']:.2f}% — starvation mechanism "
                        f"too weak to matter.")

    r_high = [r for r in rows if r["urllc_frac"] >= 0.5]
    if not any(r["violation_pct"] < 10.0 for r in r_high):
        ok = False
        reasons.append("No high-URLLC-allocation policy (frac>=0.5) achieves <10% violation — "
                        "even generous allocation can't satisfy the SLA; traffic/deadline too harsh.")
    if r_high and all(r["embb_mbps_mean"] >= rows[0]["embb_mbps_mean"] * 0.95 for r in r_high):
        reasons.append("WARNING: high URLLC allocation doesn't visibly cost eMBB throughput — "
                        "trade-off may be too weak to make CMDP/floor meaningfully interesting.")

    worst_p99 = max(r["delay_p99_ms"] for r in rows)
    if worst_p99 > deadline_ms + 1e-6:
        ok = False
        reasons.append(f"delivered-packet P99 delay ({worst_p99:.3f} ms) exceeds the deadline "
                        f"({deadline_ms} ms) — deadline-drop invariant is broken, check the env.")

    feasible = any(r["violation_pct"] / 100.0 <= delta for r in rows)
    if not feasible:
        ok = False
        reasons.append(f"No static policy reaches violation_rate <= delta ({delta}) — "
                        f"the CMDP constraint is infeasible as configured. Loosen delta or "
                        f"retune traffic.")

    print()
    if ok:
        print("PASS — traffic/buffer/deadline calibration looks contested and feasible.")
    else:
        print("FAIL — retune traffic before running the training wave:")
    for r in reasons:
        print(f"  - {r}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
