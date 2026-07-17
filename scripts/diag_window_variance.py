"""Window-level (not episode-level) violation-rate variance at the chosen operating
point (lambda_arrival=25000, frac=0.8) — this is the quantity that actually feeds
the CMDP dual update (mean over cmdp.dual_update_every=2000 steps, i.e. ~10 episodes
at 200 steps/episode), not a single episode."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from envs.network_slicing_env import NetworkSlicingEnv

FRAC = 0.8
WINDOW_STEPS = 2000       # matches cmdp.dual_update_every
N_WINDOWS = 20            # 20 independent windows -> 20*2000 = 40000 steps total


def main():
    env = NetworkSlicingEnv()
    env.floor_mode = "none"
    env.cmdp_enabled = False
    tier = int(round(FRAC * (env.n_tiers - 1)))

    window_means = []
    step_in_window = 0
    per_step_viol = []
    ep_seed = 0
    env.reset(seed=ep_seed)

    for _ in range(WINDOW_STEPS * N_WINDOWS):
        _, _, terminated, truncated, info = env.step([tier] * env.n_gnb)
        per_step_viol.append(info["urllc_violation_rate"].mean())
        step_in_window += 1
        if terminated or truncated:
            ep_seed += 1
            env.reset(seed=ep_seed)
        if step_in_window == WINDOW_STEPS:
            window_means.append(100.0 * float(np.mean(per_step_viol)))
            per_step_viol = []
            step_in_window = 0
    env.close()

    w = np.array(window_means)
    print(f"n_windows={len(w)}  window_steps={WINDOW_STEPS} (~{WINDOW_STEPS//200} episodes/window)")
    for i, v in enumerate(w):
        print(f"  window={i:>2}  mean_violation%={v:6.2f}")
    print(f"\nwindow-level:  mean={w.mean():.2f}%  std={w.std():.2f}pp  "
          f"min={w.min():.2f}%  max={w.max():.2f}%")
    print(f"delta=10.0%  margin at mean: {10.0 - w.mean():.2f}pp  "
          f"margin at worst window (max): {10.0 - w.max():.2f}pp")


if __name__ == "__main__":
    main()
