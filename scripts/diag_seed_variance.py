"""Per-seed (not pooled) violation-rate variance at the chosen operating point
(lambda_arrival=25000, frac=0.8) — checks whether the +4.4pp margin above delta=0.10
holds robustly across seeds, not just in a pooled/average sense."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from envs.network_slicing_env import NetworkSlicingEnv

N_SEEDS = 20
FRAC = 0.8


def run_one_seed(seed):
    env = NetworkSlicingEnv()
    env.floor_mode = "none"
    env.cmdp_enabled = False
    tier = int(round(FRAC * (env.n_tiers - 1)))
    env.reset(seed=seed)
    arrived = dropped = 0
    per_step_mean_viol = []
    done = False
    while not done:
        _, _, terminated, truncated, info = env.step([tier] * env.n_gnb)
        done = terminated or truncated
        arrived += int(np.sum(info["urllc_arrived"]))
        dropped += int(np.sum(info["urllc_dropped_late"]) + np.sum(info["urllc_dropped_overflow"]))
        per_step_mean_viol.append(info["urllc_violation_rate"].mean())
    env.close()
    pooled_pct = 100.0 * dropped / max(arrived, 1)
    cmdp_pct = 100.0 * float(np.mean(per_step_mean_viol))
    return pooled_pct, cmdp_pct


pooled_vals, cmdp_vals = [], []
for seed in range(N_SEEDS):
    p, c = run_one_seed(seed)
    pooled_vals.append(p)
    cmdp_vals.append(c)
    print(f"seed={seed:>2}  pooled%={p:6.2f}  cmdp_quantity%={c:6.2f}")

pooled_vals = np.array(pooled_vals)
cmdp_vals = np.array(cmdp_vals)
print(f"\npooled:  mean={pooled_vals.mean():.2f}%  std={pooled_vals.std():.2f}pp  "
      f"min={pooled_vals.min():.2f}%  max={pooled_vals.max():.2f}%")
print(f"cmdp_q:  mean={cmdp_vals.mean():.2f}%  std={cmdp_vals.std():.2f}pp  "
      f"min={cmdp_vals.min():.2f}%  max={cmdp_vals.max():.2f}%")
print(f"\ndelta=10.0%  margin at mean: {10.0 - cmdp_vals.mean():.2f}pp  "
      f"margin at worst seed (max): {10.0 - cmdp_vals.max():.2f}pp")
