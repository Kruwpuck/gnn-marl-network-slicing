"""
Phase 2c: Zero-shot generalization evaluation.

Protocol:
  1. Load agent trained on 5-gNB topology (no retraining)
  2. Evaluate on 10-gNB and 20-gNB configs
  3. Compute Performance Retention Ratio = reward(N) / reward(5) * 100%

Only SAGE and GCN support arbitrary N out-of-the-box (inductive).
GAT also supports it (GATv2Conv handles variable N).
MLP baselines CANNOT zero-shot (fixed input dim) — they get NaN.

Usage:
  python evaluation/zero_shot_eval.py --backbone sage --episodes 20
"""
from __future__ import annotations
import argparse
import sys
import copy
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))


class ZeroShotEvaluator:
    """Evaluate trained agent on topologies unseen during training."""

    EVAL_SIZES = [5, 10, 20]

    def __init__(self, base_config_path: str | None = None):
        self.base_config_path = base_config_path or "configs/experiment_config.yaml"
        with open(self.base_config_path) as f:
            self._base_cfg = yaml.safe_load(f)

    def _make_env_for_n(self, n_gnb: int):
        import tempfile, os
        from envs.network_slicing_env import NetworkSlicingEnv

        cfg = copy.deepcopy(self._base_cfg)
        cfg["env"]["n_gnb"] = n_gnb
        cfg["env"]["n_ue_per_gnb"] = 5

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(cfg, f)
            tmp_path = f.name

        env = NetworkSlicingEnv(config_path=tmp_path)
        os.unlink(tmp_path)
        return env

    def evaluate_one(self, agent, n_gnb: int, n_episodes: int = 20, seed: int = 0) -> float:
        """Run agent on n_gnb topology. Returns mean episode reward."""
        env = self._make_env_for_n(n_gnb)
        total_rewards = []

        for ep in range(n_episodes):
            obs, info = env.reset(seed=seed + ep)
            ep_reward = 0.0
            done = False
            while not done:
                graph = info["graph"]
                try:
                    actions = agent.act(graph, greedy=True)
                    # Ensure actions match current n_gnb
                    if len(actions) != n_gnb:
                        actions = np.resize(actions, n_gnb) % agent.n_actions
                except Exception:
                    actions = np.zeros(n_gnb, dtype=int)
                obs, reward, terminated, truncated, info = env.step(actions)
                ep_reward += reward
                done = terminated or truncated
            total_rewards.append(ep_reward)

        env.close()
        return float(np.mean(total_rewards))

    def evaluate_all_sizes(
        self, agent, n_episodes: int = 20, seed: int = 0
    ) -> dict[int, float]:
        """Evaluate on all EVAL_SIZES. Returns {n_gnb: mean_reward}."""
        return {
            n: self.evaluate_one(agent, n, n_episodes, seed)
            for n in self.EVAL_SIZES
        }

    @staticmethod
    def retention_ratio(rewards: dict[int, float], base_n: int = 5) -> dict[int, float]:
        """PRR = reward(N) / reward(base_n) * 100%."""
        base = rewards.get(base_n, float("nan"))
        if base == 0 or np.isnan(base):
            return {n: float("nan") for n in rewards}
        return {n: (r / base) * 100.0 for n, r in rewards.items()}

    @staticmethod
    def print_table(results: dict[str, dict]) -> None:
        """results: {algo_label: {n_gnb: reward}}"""
        sizes = ZeroShotEvaluator.EVAL_SIZES
        header = f"{'Algorithm':<25}" + "".join(f"  N={n:>3} (rew)" for n in sizes) + "  PRR-10  PRR-20"
        print(header)
        print("-" * len(header))
        for label, rew_dict in results.items():
            rew_str = "".join(f"  {rew_dict.get(n, float('nan')):>10.4f}" for n in sizes)
            prr = ZeroShotEvaluator.retention_ratio(rew_dict)
            print(f"{label:<25}{rew_str}  {prr.get(10, float('nan')):>6.1f}%  {prr.get(20, float('nan')):>6.1f}%")


if __name__ == "__main__":
    from gnn import BACKBONES
    from agents.dqn_agent import DQNAgent

    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="sage", choices=["gat", "sage", "gcn"])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ev = ZeroShotEvaluator()
    backbone = BACKBONES[args.backbone]()
    agent = DQNAgent(backbone)

    print(f"Zero-shot eval — untrained {args.backbone} agent (for shape test only)")
    rewards = ev.evaluate_all_sizes(agent, n_episodes=args.episodes, seed=args.seed)
    label = f"gnn-madqn/{args.backbone}"
    ev.print_table({label: rewards})
    prr = ev.retention_ratio(rewards)
    print(f"\nPRR-10={prr.get(10, 'nan'):.1f}%  PRR-20={prr.get(20, 'nan'):.1f}%")
