"""
Train proposed GNN-MARL agents (GNN-MADQN, GNN-MAPPO).

Usage:
  python training/train_proposed.py --algo gnn-madqn --backbone gat --steps 1000000 --seed 42
  python training/train_proposed.py --algo gnn-mappo --backbone gat --steps 1000000 --seed 42
  python training/train_proposed.py --algo gnn-mappo --backbone sage --steps 1000000 --seed 42
  python training/train_proposed.py --algo gnn-madqn --backbone gcn  --steps 1000000 --seed 42
"""
from __future__ import annotations
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.network_slicing_env import NetworkSlicingEnv
from gnn import BACKBONES
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from training.replay_buffer import ReplayBuffer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from training.rollout_buffer import RolloutBuffer

BATCH_SIZE = 64
REPLAY_START = 1000
ROLLOUT_STEPS = 512


def make_env(seed: int) -> NetworkSlicingEnv:
    env = NetworkSlicingEnv()
    env.reset(seed=seed)
    return env


def _graph_from_info(info: dict) -> dict:
    return info["graph"]


def train_gnn_dqn(backbone_name: str, steps: int, seed: int, log_path: Path) -> None:
    np.random.seed(seed)
    env = make_env(seed)

    backbone = BACKBONES[backbone_name]()
    agent = DQNAgent(backbone).to(DEVICE)
    print(f"[gnn-madqn/{backbone_name}] device={DEVICE}")
    buf = ReplayBuffer(capacity=50_000)

    obs_arr, info = env.reset(seed=seed)
    graph = _graph_from_info(info)
    ep_reward = 0.0
    ep_count = 0

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "episode", "ep_reward", "epsilon", "loss"])
        t0 = time.time()

        for step in range(steps):
            actions = agent.act(graph)
            next_obs, reward, terminated, truncated, next_info = env.step(actions)
            done = terminated or truncated
            next_graph = _graph_from_info(next_info)

            buf.push(graph, actions, reward, next_graph, done)
            ep_reward += reward
            graph = next_graph

            loss = None
            if len(buf) >= REPLAY_START:
                batch = buf.sample(BATCH_SIZE)
                loss = agent.learn(batch)

            if done:
                ep_count += 1
                if step % 1000 == 0:
                    elapsed = time.time() - t0
                    print(f"[gnn-madqn/{backbone_name}] step={step:>7} ep={ep_count:>4} "
                          f"rew={ep_reward:+.3f} eps={agent.epsilon:.3f} "
                          f"loss={loss:.4f if loss else 'N/A'} t={elapsed:.0f}s")
                writer.writerow([step, ep_count, round(ep_reward, 4),
                                 round(agent.epsilon, 4), round(loss, 6) if loss else ""])
                obs_arr, info = env.reset(seed=seed + ep_count)
                graph = _graph_from_info(info)
                ep_reward = 0.0

    env.close()
    model_path = log_path.with_suffix(".pt")
    torch.save({"state_dict": agent.state_dict(), "backbone": backbone_name,
                "algo": "gnn-madqn", "steps": steps, "seed": seed}, model_path)
    print(f"[gnn-madqn/{backbone_name}] done → {log_path} | model → {model_path}")


def train_gnn_ppo(backbone_name: str, steps: int, seed: int, log_path: Path) -> None:
    np.random.seed(seed)
    env = make_env(seed)
    n_gnb = env.n_gnb

    backbone = BACKBONES[backbone_name]()
    agent = PPOAgent(backbone).to(DEVICE)
    print(f"[gnn-mappo/{backbone_name}] device={DEVICE}")
    buf = RolloutBuffer(n_steps=ROLLOUT_STEPS, n_agents=n_gnb)

    obs_arr, info = env.reset(seed=seed)
    graph = _graph_from_info(info)
    ep_reward = 0.0
    ep_count = 0

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "episode", "ep_reward", "loss"])
        t0 = time.time()

        for step in range(steps):
            actions, log_probs, values = agent.act(graph)
            next_obs, reward, terminated, truncated, next_info = env.step(actions)
            done = terminated or truncated
            next_graph = _graph_from_info(next_info)
            ep_reward += reward

            buf.add(graph, actions, log_probs, reward, values, done)

            if done:
                ep_count += 1
                obs_arr, info = env.reset(seed=seed + ep_count)
                graph = _graph_from_info(info)
            else:
                graph = next_graph

            loss_info = None
            if buf.is_full():
                rollout = buf.compute_advantages()
                loss_info = agent.learn(rollout)
                buf.clear()
                if step % 1000 == 0:
                    elapsed = time.time() - t0
                    print(f"[gnn-mappo/{backbone_name}] step={step:>7} ep={ep_count:>4} "
                          f"rew={ep_reward:+.3f} t={elapsed:.0f}s")
                writer.writerow([step, ep_count, round(ep_reward, 4),
                                 round(loss_info.get("loss", 0), 6) if loss_info else ""])
                ep_reward = 0.0

    env.close()
    model_path = log_path.with_suffix(".pt")
    torch.save({"state_dict": agent.state_dict(), "backbone": backbone_name,
                "algo": "gnn-mappo", "steps": steps, "seed": seed}, model_path)
    print(f"[gnn-mappo/{backbone_name}] done → {log_path} | model → {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["gnn-madqn", "gnn-mappo"], default="gnn-madqn")
    parser.add_argument("--backbone", choices=list(BACKBONES.keys()), default="gat")
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    log_dir = Path("results/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{args.algo}_{args.backbone}_seed{args.seed}.csv"

    if args.algo == "gnn-madqn":
        train_gnn_dqn(args.backbone, args.steps, args.seed, log_path)
    else:
        train_gnn_ppo(args.backbone, args.steps, args.seed, log_path)
