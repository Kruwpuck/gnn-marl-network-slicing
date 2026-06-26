"""
Phase 3: Ablation study runner.

Ablations:
  A1. GNN backbone: GCN vs GraphSAGE vs GAT (fix RL=MAPPO)
  A2. Reward weights: (w1,w2,w3,w4) variants
  A3. GNN depth: 1 / 2 / 3 layers
  A4. Traffic burstiness: MMPP low/medium/high

Usage:
  python ablation/backbone_ablation.py --ablation backbone --steps 200000 --seed 42
  python ablation/backbone_ablation.py --ablation reward-weights --steps 200000 --seed 42
  python ablation/backbone_ablation.py --ablation gnn-depth --steps 200000 --seed 42
"""
from __future__ import annotations
import argparse
import csv
import copy
import sys
import tempfile
import os
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from gnn import BACKBONES, SAGEBackbone, GATBackbone, GCNBackbone
from agents.ppo_agent import PPOAgent
from training.rollout_buffer import RolloutBuffer


REWARD_WEIGHT_VARIANTS = [
    {"w1": 0.5, "w2": 0.2, "w3": 0.2, "w4": 0.1},
    {"w1": 0.4, "w2": 0.3, "w3": 0.2, "w4": 0.1},  # default
    {"w1": 0.3, "w2": 0.3, "w3": 0.3, "w4": 0.1},
]

MMPP_VARIANTS = {
    "low_burst":    {"lambda_burst": 2.0, "lambda_idle": 0.5, "r_burst_to_idle": 0.1, "r_idle_to_burst": 0.5},
    "medium_burst": {"lambda_burst": 4.0, "lambda_idle": 0.5, "r_burst_to_idle": 0.2, "r_idle_to_burst": 0.8},
    "high_burst":   {"lambda_burst": 8.0, "lambda_idle": 0.5, "r_burst_to_idle": 0.4, "r_idle_to_burst": 0.9},
}

ROLLOUT_STEPS = 256


class BackboneAblation:
    """Run systematic ablations and collect convergence data."""

    def __init__(self, base_config: str = "configs/experiment_config.yaml"):
        with open(base_config) as f:
            self._base_cfg = yaml.safe_load(f)
        self.base_config = base_config

    def _make_env(self, cfg_override: dict | None = None):
        from envs.network_slicing_env import NetworkSlicingEnv

        if cfg_override is None:
            return NetworkSlicingEnv(config_path=self.base_config)

        cfg = copy.deepcopy(self._base_cfg)
        for section, vals in cfg_override.items():
            if section in cfg and isinstance(vals, dict):
                cfg[section].update(vals)
            else:
                cfg[section] = vals

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(cfg, f)
            tmp = f.name

        env = NetworkSlicingEnv(config_path=tmp)
        os.unlink(tmp)
        return env

    def _run_ppo(
        self,
        backbone,
        label: str,
        steps: int,
        seed: int,
        log_path: Path,
        cfg_override: dict | None = None,
    ) -> list[float]:
        np.random.seed(seed)
        env = self._make_env(cfg_override)
        n_gnb = env.n_gnb
        agent = PPOAgent(backbone)
        buf = RolloutBuffer(n_steps=ROLLOUT_STEPS, n_agents=n_gnb)

        obs, info = env.reset(seed=seed)
        graph = info["graph"]
        ep_reward = 0.0
        ep_count = 0
        rewards_log = []

        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "episode", "ep_reward", "loss"])

            for step in range(steps):
                actions, log_probs, values = agent.act(graph)
                next_obs, reward, terminated, truncated, next_info = env.step(actions)
                done = terminated or truncated
                ep_reward += reward

                buf.add(graph, actions, log_probs, reward, values, done)

                if done:
                    ep_count += 1
                    obs, info = env.reset(seed=seed + ep_count)
                    graph = info["graph"]
                else:
                    graph = next_info["graph"]

                if buf.is_full():
                    rollout = buf.compute_advantages()
                    loss_info = agent.learn(rollout)
                    buf.clear()
                    writer.writerow([step, ep_count, round(ep_reward, 4),
                                     round(loss_info.get("loss", 0), 6)])
                    rewards_log.append(ep_reward)
                    ep_reward = 0.0

        env.close()
        print(f"[ablation/{label}] done → {log_path}")
        return rewards_log

    def run_backbone_ablation(self, steps: int, seed: int, log_dir: Path) -> dict:
        log_dir.mkdir(parents=True, exist_ok=True)
        results = {}
        for name, Cls in BACKBONES.items():
            log_path = log_dir / f"ablation_backbone_{name}_seed{seed}.csv"
            rewards = self._run_ppo(Cls(), f"backbone/{name}", steps, seed, log_path)
            results[name] = rewards
        return results

    def run_reward_weight_ablation(self, steps: int, seed: int, log_dir: Path) -> dict:
        log_dir.mkdir(parents=True, exist_ok=True)
        results = {}
        for i, weights in enumerate(REWARD_WEIGHT_VARIANTS):
            label = f"w{i+1}_w1={weights['w1']}"
            log_path = log_dir / f"ablation_reward_{i}_seed{seed}.csv"
            backbone = GATBackbone()
            rewards = self._run_ppo(
                backbone, label, steps, seed, log_path,
                cfg_override={"reward": weights}
            )
            results[label] = {"weights": weights, "rewards": rewards}
        return results

    def run_depth_ablation(self, steps: int, seed: int, log_dir: Path) -> dict:
        """Ablate GAT depth by stacking extra conv layers."""
        import torch.nn as nn
        from torch_geometric.nn import GATv2Conv
        from gnn.base_backbone import GNNBackbone
        import torch
        import torch.nn.functional as F

        log_dir.mkdir(parents=True, exist_ok=True)
        results = {}

        class GATDepthN(GNNBackbone):
            def __init__(self, n_layers: int):
                super().__init__()
                self._out_dim = 64
                self.convs = nn.ModuleList()
                in_ch = 8
                for i in range(n_layers):
                    out_ch = 64 if i == n_layers - 1 else 32
                    heads = 1 if i == n_layers - 1 else 4
                    concat = False if i == n_layers - 1 else True
                    self.convs.append(
                        GATv2Conv(in_ch, out_ch if not concat else out_ch, heads=heads,
                                  edge_dim=1, concat=concat)
                    )
                    in_ch = out_ch * (heads if concat else 1)

            @property
            def output_dim(self):
                return self._out_dim

            def forward(self, x, edge_index, edge_attr):
                x, edge_index, edge_attr = self.to_tensors(x, edge_index, edge_attr)
                for i, conv in enumerate(self.convs[:-1]):
                    x = F.elu(conv(x, edge_index, edge_attr=edge_attr))
                return self.convs[-1](x, edge_index, edge_attr=edge_attr)

        for depth in [1, 2, 3]:
            label = f"gat_depth_{depth}"
            log_path = log_dir / f"ablation_depth_{depth}_seed{seed}.csv"
            backbone = GATDepthN(n_layers=depth)
            rewards = self._run_ppo(backbone, label, steps, seed, log_path)
            results[label] = rewards
        return results

    def run_burstiness_ablation(self, steps: int, seed: int, log_dir: Path) -> dict:
        log_dir.mkdir(parents=True, exist_ok=True)
        results = {}
        for variant_name, mmpp_params in MMPP_VARIANTS.items():
            log_path = log_dir / f"ablation_burst_{variant_name}_seed{seed}.csv"
            backbone = GATBackbone()
            rewards = self._run_ppo(
                backbone, variant_name, steps, seed, log_path,
                cfg_override={"traffic": {"embb": mmpp_params,
                                          "urllc": self._base_cfg["traffic"]["urllc"]}}
            )
            results[variant_name] = rewards
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", choices=["backbone", "reward-weights", "gnn-depth", "burstiness"],
                        default="backbone")
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", default="results/logs")
    args = parser.parse_args()

    abl = BackboneAblation()
    log_dir = Path(args.log_dir)

    if args.ablation == "backbone":
        abl.run_backbone_ablation(args.steps, args.seed, log_dir)
    elif args.ablation == "reward-weights":
        abl.run_reward_weight_ablation(args.steps, args.seed, log_dir)
    elif args.ablation == "gnn-depth":
        abl.run_depth_ablation(args.steps, args.seed, log_dir)
    elif args.ablation == "burstiness":
        abl.run_burstiness_ablation(args.steps, args.seed, log_dir)
