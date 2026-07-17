"""Sweep urllc lambda_arrival to find a load where allocation genuinely matters
(contested regime), now that the channel/topology fix makes SINR realistic."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from envs.network_slicing_env import NetworkSlicingEnv

LAMBDAS = [4000, 8000, 15000, 25000, 40000, 60000]
FRACS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8]


def run(lam, frac, episodes=10):
    env = NetworkSlicingEnv()
    env.floor_mode = "none"
    env.cmdp_enabled = False
    env.urllc_lambda = lam
    tier = int(round(frac * (env.n_tiers - 1)))
    arrived = dropped = 0
    for ep in range(episodes):
        env.reset(seed=ep)
        env._urllc_traffic = [
            type(env._urllc_traffic[0])(lam, env.urllc_pkt_bits, np.random.default_rng(ep * 100 + i))
            for i in range(env.n_gnb)
        ]
        done = False
        while not done:
            _, _, terminated, truncated, info = env.step([tier] * env.n_gnb)
            done = terminated or truncated
            arrived += int(np.sum(info["urllc_arrived"]))
            dropped += int(np.sum(info["urllc_dropped_late"]) + np.sum(info["urllc_dropped_overflow"]))
    env.close()
    return 100.0 * dropped / max(arrived, 1)


for lam in LAMBDAS:
    row = [run(lam, f) for f in FRACS]
    mbps = lam * 256 / 1e6
    print(f"lambda={lam:>6} ({mbps:5.2f} Mbps/gNB offered)  " +
          "  ".join(f"f={f}:{v:5.1f}%" for f, v in zip(FRACS, row)))
