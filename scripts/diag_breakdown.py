"""Per-gNB violation breakdown at candidate (lambda, frac) points, reporting both:
  - pooled/arrival-weighted violation (sum(drops)/sum(arrivals) per gNB)
  - the network-wide unweighted mean-of-per-step-per-gNB-rate (the actual CMDP
    quantity compared against delta in env.step(), i.e. mean(urllc_violation_rate)
    averaged again over steps) — these can differ when load/violation is uneven
    across gNBs or across steps (small-sample per-step ratios).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from envs.network_slicing_env import NetworkSlicingEnv

LAMBDAS = [25000, 40000]
FRACS = [0.1, 0.5, 0.8]
EPISODES = 15


def run(lam, frac):
    env = NetworkSlicingEnv()
    env.floor_mode = "none"
    env.cmdp_enabled = False
    tier = int(round(frac * (env.n_tiers - 1)))
    n = env.n_gnb
    arrived = np.zeros(n)
    dropped = np.zeros(n)
    per_step_mean_viol = []  # the actual CMDP quantity: mean over gNBs, per step
    for ep in range(EPISODES):
        env.reset(seed=ep)
        env._urllc_traffic = [
            type(env._urllc_traffic[0])(lam, env.urllc_pkt_bits, np.random.default_rng(ep * 100 + i))
            for i in range(n)
        ]
        done = False
        while not done:
            _, _, terminated, truncated, info = env.step([tier] * n)
            done = terminated or truncated
            arrived += info["urllc_arrived"]
            dropped += info["urllc_dropped_late"] + info["urllc_dropped_overflow"]
            per_step_mean_viol.append(info["urllc_violation_rate"].mean())
    env.close()
    pooled_per_gnb = 100.0 * dropped / np.maximum(arrived, 1)
    cmdp_quantity = 100.0 * float(np.mean(per_step_mean_viol))
    return pooled_per_gnb, cmdp_quantity


for lam in LAMBDAS:
    print(f"\n=== lambda_arrival={lam} ===")
    for frac in FRACS:
        pooled_per_gnb, cmdp_quantity = run(lam, frac)
        print(f"  frac={frac}: per-gNB pooled%={np.round(pooled_per_gnb,1)}  "
              f"  network pooled mean%={pooled_per_gnb.mean():5.2f}"
              f"  CMDP quantity (mean-of-per-step-mean)%={cmdp_quantity:5.2f}")
