"""
Phase 2b: Network performance evaluation.

Runs trained agents for N evaluation episodes and collects:
  - eMBB: throughput (Mbps), spectral efficiency (bps/Hz)
  - URLLC: mean delay, 99.9th-pct delay, SLA violation rate (%)
  - PRB utilization (%), Jain's Fairness Index

Usage:
  python evaluation/network_perf_eval.py --algo gnn-madqn --backbone gat --episodes 100
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def jains_fairness(rates: np.ndarray) -> float:
    """Jain's Fairness Index: (sum r_i)^2 / (N * sum r_i^2)."""
    if rates.sum() == 0:
        return 0.0
    return float((rates.sum() ** 2) / (len(rates) * (rates ** 2).sum()))


class NetworkPerfEvaluator:
    """Run episodes with a trained agent and collect per-step network metrics."""

    def __init__(self, config_path: str | None = None):
        self.config_path = config_path

    def _make_env(self):
        from envs.network_slicing_env import NetworkSlicingEnv
        return NetworkSlicingEnv(config_path=self.config_path)

    def evaluate_gnn_agent(
        self,
        agent,
        n_episodes: int = 100,
        seed: int = 0,
    ) -> dict:
        env = self._make_env()
        all_embb_mbps, all_se, all_urllc_delay, all_sla_viol, all_prb_util = [], [], [], [], []

        for ep in range(n_episodes):
            obs, info = env.reset(seed=seed + ep)
            done = False
            while not done:
                graph = info["graph"]
                actions = agent.act(graph, greedy=True)
                obs, reward, terminated, truncated, info = env.step(actions)
                done = terminated or truncated

                embb = info.get("embb_rates", np.zeros(env.n_gnb))
                delay = info.get("urllc_delay_ms", np.zeros(env.n_gnb))

                all_embb_mbps.extend((embb / 1e6).tolist())
                all_se.extend((embb / env.bandwidth_hz).tolist())
                all_urllc_delay.extend(delay.tolist())
                all_sla_viol.extend((delay > env.urllc_max_delay_ms).astype(float).tolist())

        env.close()

        delays = np.array(all_urllc_delay)
        embb_arr = np.array(all_embb_mbps)
        return {
            "embb_throughput_mbps_mean": float(embb_arr.mean()),
            "embb_throughput_mbps_std": float(embb_arr.std()),
            "spectral_efficiency_bps_hz": float(np.array(all_se).mean()),
            "urllc_delay_ms_mean": float(delays.mean()),
            "urllc_delay_ms_p999": float(np.percentile(delays, 99.9)),
            "sla_violation_rate_pct": float(np.mean(all_sla_viol) * 100),
            "jains_fairness": float(jains_fairness(embb_arr)),
            "n_episodes": n_episodes,
        }

    def evaluate_mlp_agent(
        self,
        agent,
        n_episodes: int = 100,
        seed: int = 0,
        centralized: bool = False,
    ) -> dict:
        env = self._make_env()
        all_embb_mbps, all_se, all_urllc_delay, all_sla_viol = [], [], [], []

        for ep in range(n_episodes):
            obs, info = env.reset(seed=seed + ep)
            done = False
            while not done:
                if centralized:
                    raw = agent.act(obs.flatten())
                    actions = np.full(env.n_gnb, int(raw[0]) if hasattr(raw, '__len__') else int(raw))
                else:
                    actions = np.array([int(agent.act(obs[i])) for i in range(env.n_gnb)])
                obs, reward, terminated, truncated, info = env.step(actions)
                done = terminated or truncated

                embb = info.get("embb_rates", np.zeros(env.n_gnb))
                delay = info.get("urllc_delay_ms", np.zeros(env.n_gnb))
                all_embb_mbps.extend((embb / 1e6).tolist())
                all_se.extend((embb / env.bandwidth_hz).tolist())
                all_urllc_delay.extend(delay.tolist())
                all_sla_viol.extend((delay > env.urllc_max_delay_ms).astype(float).tolist())

        env.close()
        delays = np.array(all_urllc_delay)
        embb_arr = np.array(all_embb_mbps)
        return {
            "embb_throughput_mbps_mean": float(embb_arr.mean()),
            "embb_throughput_mbps_std": float(embb_arr.std()),
            "spectral_efficiency_bps_hz": float(np.array(all_se).mean()),
            "urllc_delay_ms_mean": float(delays.mean()),
            "urllc_delay_ms_p999": float(np.percentile(delays, 99.9)),
            "sla_violation_rate_pct": float(np.mean(all_sla_viol) * 100),
            "jains_fairness": float(jains_fairness(embb_arr)),
            "n_episodes": n_episodes,
        }

    @staticmethod
    def print_table(results: dict[str, dict]) -> None:
        cols = [
            ("eMBB (Mbps)", "embb_throughput_mbps_mean"),
            ("SE (bps/Hz)", "spectral_efficiency_bps_hz"),
            ("Delay (ms)", "urllc_delay_ms_mean"),
            ("D-p99.9 (ms)", "urllc_delay_ms_p999"),
            ("SLA viol %", "sla_violation_rate_pct"),
            ("Jain's F", "jains_fairness"),
        ]
        header = f"{'Algorithm':<22}" + "".join(f"{c[0]:>14}" for c in cols)
        print(header)
        print("-" * len(header))
        for label, m in results.items():
            row = f"{label:<22}" + "".join(f"{m.get(c[1], float('nan')):>14.4f}" for c in cols)
            print(row)

    @staticmethod
    def to_latex(results: dict[str, dict]) -> str:
        rows = []
        for label, m in results.items():
            rows.append(
                f"{label} & {m['embb_throughput_mbps_mean']:.2f} & "
                f"{m['spectral_efficiency_bps_hz']:.4f} & "
                f"{m['urllc_delay_ms_mean']:.2f} & "
                f"{m['urllc_delay_ms_p999']:.2f} & "
                f"{m['sla_violation_rate_pct']:.2f}\\% & "
                f"{m['jains_fairness']:.4f} \\\\"
            )
        return "\n".join(rows)


if __name__ == "__main__":
    import torch
    from gnn import BACKBONES
    from agents.dqn_agent import DQNAgent
    from agents.mlp_agent import MLPDQNAgent

    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="gnn-madqn")
    parser.add_argument("--backbone", default="gat")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ev = NetworkPerfEvaluator()

    if args.algo.startswith("gnn"):
        backbone = BACKBONES[args.backbone]()
        agent = DQNAgent(backbone)
        results = {"untrained_" + args.algo: ev.evaluate_gnn_agent(agent, args.episodes, args.seed)}
    else:
        agent = MLPDQNAgent(obs_dim=8)
        results = {"untrained_" + args.algo: ev.evaluate_mlp_agent(agent, args.episodes, args.seed)}

    ev.print_table(results)
    print("\nLaTeX row:")
    print(ev.to_latex(results))
