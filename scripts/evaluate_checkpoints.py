"""
Held-out evaluation protocol for trained checkpoints.

The training CSVs mix exploration noise and CMDP non-stationarity (lambda still
moving) into the last-10%-of-training window that scripts/analyze_results.py used
in v2 — not a fair cross-algorithm comparison. This script instead loads the final
saved model (results/logs/{run}.pt), runs it *greedy* for N episodes on seeds that
were never used during training, and writes one row per episode to
results/eval/{run}_eval.csv — the input rliable_report.py expects.

Usage:
  python scripts/evaluate_checkpoints.py --runs results/logs/*.pt --episodes 30
  python scripts/evaluate_checkpoints.py --run results/logs/gnn-madqn_gat_seed42.pt
"""
from __future__ import annotations
import argparse
import csv
import glob
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.network_slicing_env import NetworkSlicingEnv
from gnn import BACKBONES
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from agents.mlp_agent import MLPDQNAgent, MLPPPOAgent

EVAL_SEED_BASE = 10_000  # disjoint from training seeds (42-46) and smoke seeds (999)

EVAL_COLUMNS = [
    "run_name", "algo", "seed", "episode",
    "ep_reward",
    "embb_mbps_mean", "spectral_eff_bps_hz", "embb_p5_mbps",
    "urllc_delay_ms_mean", "urllc_delay_p95", "urllc_delay_p99",
    "urllc_drop_late_pct", "urllc_drop_overflow_pct",
    "sla_violation_pct", "sla_satisfaction_pct",
    "timely_throughput_mbps", "jains_fairness", "lam",
]


def jains_fairness(rates: np.ndarray) -> float:
    if rates.sum() == 0:
        return 0.0
    return float((rates.sum() ** 2) / (len(rates) * (rates ** 2).sum()))


def load_agent(pt_path: Path, device: torch.device):
    ckpt = torch.load(pt_path, map_location=device, weights_only=False)
    algo = ckpt["algo"]
    if algo.startswith("gnn-madqn"):
        agent = DQNAgent(BACKBONES[ckpt["backbone"]]()).to(device)
        agent.load_state_dict(ckpt["state_dict"])
        kind = "gnn-dqn"
    elif algo.startswith("gnn-mappo"):
        agent = PPOAgent(BACKBONES[ckpt["backbone"]]()).to(device)
        agent.load_state_dict(ckpt["state_dict"])
        kind = "gnn-ppo"
    elif "dqn" in algo:
        agent = MLPDQNAgent(ckpt["obs_dim"]).to(device)
        agent.load_state_dict(ckpt["state_dict"])
        kind = "mlp-dqn-central" if algo == "central-dqn" else "mlp-dqn"
    else:
        agent = MLPPPOAgent(ckpt["obs_dim"]).to(device)
        agent.load_state_dict(ckpt["state_dict"])
        kind = "mlp-ppo-central" if algo == "central-ppo" else "mlp-ppo"
    agent.eval()
    return agent, algo, kind


def run_episode(env: NetworkSlicingEnv, agent, kind: str, seed: int) -> dict:
    obs, info = env.reset(seed=seed)
    done = False
    ep_reward = 0.0
    embb_bps: list[float] = []
    fairness: list[float] = []
    delays: list[float] = []
    arrived = dropped_late = dropped_overflow = 0
    delivered_bits = 0.0
    lam_last = 0.0
    n_steps = 0

    while not done:
        if kind == "gnn-dqn":
            actions = agent.act(info["graph"], greedy=True)
        elif kind == "gnn-ppo":
            actions, _, _ = agent.act(info["graph"], greedy=True)
        elif kind == "mlp-dqn-central":
            raw = agent.act(obs.flatten(), greedy=True)
            actions = np.full(env.n_gnb, int(raw) if np.isscalar(raw) else int(np.asarray(raw).flat[0]))
        elif kind == "mlp-dqn":
            actions = np.array([int(agent.act(obs[i], greedy=True)) for i in range(env.n_gnb)])
        elif kind == "mlp-ppo-central":
            actions_c, _, _ = agent.act(obs.flatten(), greedy=True)
            actions = np.full(env.n_gnb, int(actions_c[0]))
        else:  # mlp-ppo (independent)
            actions, _, _ = agent.act(obs, greedy=True)

        obs, reward, terminated, truncated, info = env.step(actions)
        done = terminated or truncated
        ep_reward += reward
        n_steps += 1

        embb = np.asarray(info["embb_rates"], dtype=np.float64)
        embb_bps.extend(embb.tolist())
        fairness.append(jains_fairness(embb))
        for gnb_delays in info["urllc_delivered_delays"]:
            delays.extend(gnb_delays)
        arrived += int(np.sum(info["urllc_arrived"]))
        dropped_late += int(np.sum(info["urllc_dropped_late"]))
        dropped_overflow += int(np.sum(info["urllc_dropped_overflow"]))
        delivered_bits += float(np.sum(info["urllc_delivered_bits"]))
        lam_last = float(info["lam"])

    embb_arr = np.asarray(embb_bps) if embb_bps else np.array([0.0])
    delay_arr = np.asarray(delays) if delays else np.array([0.0])
    arrived = max(arrived, 1)
    drop_late_pct = 100.0 * dropped_late / arrived
    drop_overflow_pct = 100.0 * dropped_overflow / arrived
    viol_pct = drop_late_pct + drop_overflow_pct
    window_s = max(n_steps * env.slot_s, env.slot_s)

    return {
        "ep_reward": ep_reward,
        "embb_mbps_mean": float(embb_arr.mean() / 1e6),
        "spectral_eff_bps_hz": float((embb_arr / env.bandwidth_hz).mean()),
        "embb_p5_mbps": float(np.percentile(embb_arr, 5) / 1e6),
        "urllc_delay_ms_mean": float(delay_arr.mean()),
        "urllc_delay_p95": float(np.percentile(delay_arr, 95)),
        "urllc_delay_p99": float(np.percentile(delay_arr, 99)),
        "urllc_drop_late_pct": drop_late_pct,
        "urllc_drop_overflow_pct": drop_overflow_pct,
        "sla_violation_pct": viol_pct,
        "sla_satisfaction_pct": 100.0 - viol_pct,
        "timely_throughput_mbps": float(delivered_bits / window_s / 1e6),
        "jains_fairness": float(np.mean(fairness)) if fairness else 0.0,
        "lam": lam_last,
    }


def evaluate_run(pt_path: Path, episodes: int, config_path: str | None, out_dir: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent, algo, kind = load_agent(pt_path, device)
    run_name = pt_path.stem
    seed = None
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    seed = ckpt.get("seed")

    env = NetworkSlicingEnv(config_path=config_path)
    env.cmdp_enabled = False  # evaluation: report raw violation, don't let lambda drift here

    out_path = out_dir / f"{run_name}_eval.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EVAL_COLUMNS)
        writer.writeheader()
        for ep in range(episodes):
            metrics = run_episode(env, agent, kind, seed=EVAL_SEED_BASE + ep)
            row = {"run_name": run_name, "algo": algo, "seed": seed, "episode": ep, **metrics}
            writer.writerow(row)
    env.close()
    print(f"[{run_name}] {episodes} eval episodes -> {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=str, default=None, help="single .pt checkpoint path")
    p.add_argument("--runs", type=str, default=None, help="glob pattern for .pt checkpoints")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--out-dir", type=str, default="results/eval")
    args = p.parse_args()

    paths: list[Path] = []
    if args.run:
        paths.append(Path(args.run))
    if args.runs:
        paths.extend(Path(p) for p in sorted(glob.glob(args.runs)))
    if not paths:
        paths = [Path(p) for p in sorted(glob.glob("results/logs/*.pt"))]

    for pt_path in paths:
        evaluate_run(pt_path, args.episodes, args.config, Path(args.out_dir))


if __name__ == "__main__":
    main()
