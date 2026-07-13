"""
Train proposed GNN-MARL agents (GNN-MADQN, GNN-MAPPO).

Rich CSV logging (reward / loss / PPO breakdown / network KPIs) + periodic
resumable checkpoints. Use --resume to continue an interrupted run.

Usage:
  python training/train_proposed.py --algo gnn-madqn --backbone gat --steps 200000 --seed 42
  python training/train_proposed.py --algo gnn-mappo --backbone gat --steps 1000000 --seed 42 --resume
"""
from __future__ import annotations
import argparse
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.network_slicing_env import NetworkSlicingEnv
from gnn import BACKBONES
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from training.replay_buffer import ReplayBuffer
from training.rollout_buffer import RolloutBuffer
from training.metrics_logger import (
    MetricsLogger, EpisodeNetworkStats, CheckpointManager,
    capture_rng_state, restore_rng_state,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
REPLAY_START = 1000
ROLLOUT_STEPS = 512


def make_env(seed: int) -> NetworkSlicingEnv:
    env = NetworkSlicingEnv()
    env.reset(seed=seed)
    return env


def _graph_from_info(info: dict) -> dict:
    return info["graph"]


def _maybe_resume(ckpt: CheckpointManager, agent, resume: bool):
    """Restore (start_step, episode, ma_deque, elapsed) from last checkpoint."""
    if not resume:
        return 0, 0, deque(maxlen=100), 0.0
    state = ckpt.load_latest()
    if state is None:
        return 0, 0, deque(maxlen=100), 0.0
    agent.load_state_dict(state["model"])
    agent.optimizer.load_state_dict(state["optimizer"])
    if "epsilon" in state and hasattr(agent, "epsilon"):
        agent.epsilon = state["epsilon"]
    restore_rng_state(state.get("rng", {}))
    ma = deque(state.get("ma_deque", []), maxlen=100)
    print(f"[resume] {ckpt.run_name}: from step {state['step']} ep {state['episode']}")
    return int(state["step"]), int(state["episode"]), ma, float(state.get("elapsed_sec", 0.0))


def train_gnn_dqn(backbone_name, steps, seed, log_path, ckpt_interval, resume) -> None:
    run_name = f"gnn-madqn_{backbone_name}_seed{seed}"
    np.random.seed(seed)
    env = make_env(seed)

    backbone = BACKBONES[backbone_name]()
    agent = DQNAgent(backbone).to(DEVICE)
    print(f"[gnn-madqn/{backbone_name}] device={DEVICE}")
    buf = ReplayBuffer(capacity=50_000)

    ckpt = CheckpointManager(run_name)
    start_step, ep_count, ma_deque, elapsed_off = _maybe_resume(ckpt, agent, resume)

    logger = MetricsLogger(log_path, resume=resume, ma_deque=ma_deque)
    logger.set_elapsed_offset(elapsed_off)
    netstats = EpisodeNetworkStats()

    obs_arr, info = env.reset(seed=seed + ep_count)
    graph = _graph_from_info(info)
    ep_reward = 0.0
    t0 = time.time()

    for step in range(start_step, steps):
        actions = agent.act(graph)
        next_obs, reward, terminated, truncated, next_info = env.step(actions)
        done = terminated or truncated
        next_graph = _graph_from_info(next_info)
        netstats.add(next_info)

        buf.push(graph, actions, reward, next_graph, done)
        ep_reward += reward
        graph = next_graph

        loss = None
        if len(buf) >= REPLAY_START:
            loss = agent.learn(buf.sample(BATCH_SIZE))

        if done:
            ep_count += 1
            net = netstats.summary(env.bandwidth_hz, env.urllc_max_delay_ms)
            ma = logger.log(step, ep_count, ep_reward, loss=loss,
                            epsilon=agent.epsilon, **net)
            netstats.reset()
            if step % 1000 == 0:
                print(f"[gnn-madqn/{backbone_name}] step={step:>7} ep={ep_count:>4} "
                      f"rew={ep_reward:+.3f} ma={ma:+.3f} eps={agent.epsilon:.3f} "
                      f"loss={f'{loss:.4f}' if loss else 'N/A'} t={time.time()-t0:.0f}s")
            obs_arr, info = env.reset(seed=seed + ep_count)
            graph = _graph_from_info(info)
            ep_reward = 0.0

        if ckpt_interval and step > start_step and step % ckpt_interval == 0:
            _save_ckpt(ckpt, agent, step, ep_count, logger, run_name,
                       algo="gnn-madqn", backbone=backbone_name, seed=seed)

    _save_ckpt(ckpt, agent, steps, ep_count, logger, run_name,
               algo="gnn-madqn", backbone=backbone_name, seed=seed)
    logger.close()
    env.close()
    _save_model(log_path, agent, algo="gnn-madqn", backbone=backbone_name,
                steps=steps, seed=seed)
    print(f"[gnn-madqn/{backbone_name}] done -> {log_path}")


def train_gnn_ppo(backbone_name, steps, seed, log_path, ckpt_interval, resume) -> None:
    run_name = f"gnn-mappo_{backbone_name}_seed{seed}"
    np.random.seed(seed)
    env = make_env(seed)
    n_gnb = env.n_gnb

    backbone = BACKBONES[backbone_name]()
    agent = PPOAgent(backbone).to(DEVICE)
    print(f"[gnn-mappo/{backbone_name}] device={DEVICE}")
    buf = RolloutBuffer(n_steps=ROLLOUT_STEPS, n_agents=n_gnb)

    ckpt = CheckpointManager(run_name)
    start_step, ep_count, ma_deque, elapsed_off = _maybe_resume(ckpt, agent, resume)

    logger = MetricsLogger(log_path, resume=resume, ma_deque=ma_deque)
    logger.set_elapsed_offset(elapsed_off)
    netstats = EpisodeNetworkStats()

    obs_arr, info = env.reset(seed=seed + ep_count)
    graph = _graph_from_info(info)
    ep_reward = 0.0
    t0 = time.time()

    for step in range(start_step, steps):
        actions, log_probs, values = agent.act(graph)
        next_obs, reward, terminated, truncated, next_info = env.step(actions)
        done = terminated or truncated
        next_graph = _graph_from_info(next_info)
        ep_reward += reward
        netstats.add(next_info)

        buf.add(graph, actions, log_probs, reward, values, done)

        if done:
            ep_count += 1
            obs_arr, info = env.reset(seed=seed + ep_count)
            graph = _graph_from_info(info)
        else:
            graph = next_graph

        if buf.is_full():
            loss_info = agent.learn(buf.compute_advantages())
            buf.clear()
            net = netstats.summary(env.bandwidth_hz, env.urllc_max_delay_ms)
            ma = logger.log(step, ep_count, ep_reward,
                            loss=loss_info.get("loss"),
                            policy_loss=loss_info.get("policy_loss"),
                            value_loss=loss_info.get("value_loss"),
                            entropy=loss_info.get("entropy"), **net)
            netstats.reset()
            if step % 1000 < ROLLOUT_STEPS:
                print(f"[gnn-mappo/{backbone_name}] step={step:>7} ep={ep_count:>4} "
                      f"rew={ep_reward:+.3f} ma={ma:+.3f} t={time.time()-t0:.0f}s")
            ep_reward = 0.0

        if ckpt_interval and step > start_step and step % ckpt_interval == 0:
            _save_ckpt(ckpt, agent, step, ep_count, logger, run_name,
                       algo="gnn-mappo", backbone=backbone_name, seed=seed)

    _save_ckpt(ckpt, agent, steps, ep_count, logger, run_name,
               algo="gnn-mappo", backbone=backbone_name, seed=seed)
    logger.close()
    env.close()
    _save_model(log_path, agent, algo="gnn-mappo", backbone=backbone_name,
                steps=steps, seed=seed)
    print(f"[gnn-mappo/{backbone_name}] done -> {log_path}")


def _save_ckpt(ckpt, agent, step, episode, logger, run_name, **meta) -> None:
    ma = float(np.mean(logger.ma_deque)) if logger.ma_deque else None
    state = {
        "model": agent.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
        "step": step,
        "episode": episode,
        "rng": capture_rng_state(),
        "ma_deque": list(logger.ma_deque),
        "elapsed_sec": logger.elapsed(),
        "run_name": run_name,
    }
    if hasattr(agent, "epsilon"):
        state["epsilon"] = agent.epsilon
    state.update(meta)
    ckpt.save(state, ma_reward=ma)


def _save_model(log_path, agent, **meta) -> None:
    model_path = Path(log_path).with_suffix(".pt")
    torch.save({"state_dict": agent.state_dict(), **meta}, model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["gnn-madqn", "gnn-mappo"], default="gnn-madqn")
    parser.add_argument("--backbone", choices=list(BACKBONES.keys()), default="gat")
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt-interval", type=int, default=25_000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    log_dir = Path("results/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{args.algo}_{args.backbone}_seed{args.seed}.csv"

    if args.algo == "gnn-madqn":
        train_gnn_dqn(args.backbone, args.steps, args.seed, log_path,
                      args.ckpt_interval, args.resume)
    else:
        train_gnn_ppo(args.backbone, args.steps, args.seed, log_path,
                      args.ckpt_interval, args.resume)
