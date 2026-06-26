"""
Train baseline agents (B1: centralized-MLP, B2: IDQN/IPPO).

Usage:
  python training/train_baselines.py --algo idqn --steps 1000000 --seed 42
  python training/train_baselines.py --algo ippo --steps 1000000 --seed 42
  python training/train_baselines.py --algo central-dqn --steps 1000000 --seed 42
  python training/train_baselines.py --algo central-ppo --steps 1000000 --seed 42
"""
from __future__ import annotations
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.network_slicing_env import NetworkSlicingEnv
from agents.mlp_agent import MLPDQNAgent, MLPPPOAgent
from training.replay_buffer import ReplayBuffer
from training.rollout_buffer import RolloutBuffer

BATCH_SIZE = 64
REPLAY_START = 1000
ROLLOUT_STEPS = 512
PPO_EPOCHS = 4


def make_env(seed: int) -> NetworkSlicingEnv:
    env = NetworkSlicingEnv()
    env.reset(seed=seed)
    return env


def train_dqn(algo: str, steps: int, seed: int, log_path: Path) -> None:
    np.random.seed(seed)
    env = make_env(seed)
    n_gnb = env.n_gnb

    # B1 centralized: flat global state (n_gnb * 8)
    # B2 independent: per-agent obs (8,) — all agents share same MLP weights
    if algo == "central-dqn":
        obs_dim = n_gnb * 8
        agent = MLPDQNAgent(obs_dim)
    else:  # idqn
        obs_dim = 8
        agent = MLPDQNAgent(obs_dim)

    buf = ReplayBuffer(capacity=50_000)
    obs_arr, info = env.reset(seed=seed)
    ep_reward = 0.0
    ep_step = 0
    ep_count = 0

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "episode", "ep_reward", "epsilon", "loss"])
        t0 = time.time()

        for step in range(steps):
            if algo == "central-dqn":
                obs_in = obs_arr.flatten()
                actions_arr = agent.act(obs_in)
                actions = np.full(n_gnb, int(actions_arr), dtype=int) if np.isscalar(actions_arr) \
                    else actions_arr
            else:
                # per-agent, parameter sharing
                actions = np.array([agent.act(obs_arr[i]) for i in range(n_gnb)])

            next_obs, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated

            if algo == "central-dqn":
                buf.push(obs_arr.flatten(), actions, reward, next_obs.flatten(), done)
            else:
                for i in range(n_gnb):
                    buf.push(obs_arr[i], np.array([actions[i]]), reward / n_gnb,
                             next_obs[i], done)

            ep_reward += reward
            ep_step += 1
            obs_arr = next_obs

            loss = None
            if len(buf) >= REPLAY_START:
                batch = buf.sample(BATCH_SIZE)
                loss = agent.learn(batch)

            if done:
                ep_count += 1
                if step % 1000 == 0:
                    elapsed = time.time() - t0
                    print(f"[{algo}] step={step:>7} ep={ep_count:>4} "
                          f"rew={ep_reward:+.3f} eps={agent.epsilon:.3f} "
                          f"loss={loss:.4f if loss else 'N/A'} t={elapsed:.0f}s")
                writer.writerow([step, ep_count, round(ep_reward, 4), round(agent.epsilon, 4),
                                 round(loss, 6) if loss else ""])
                obs_arr, _ = env.reset(seed=seed + ep_count)
                ep_reward = 0.0
                ep_step = 0

    env.close()
    print(f"[{algo}] Training done → {log_path}")


def train_ppo(algo: str, steps: int, seed: int, log_path: Path) -> None:
    np.random.seed(seed)
    env = make_env(seed)
    n_gnb = env.n_gnb

    obs_dim = n_gnb * 8 if algo == "central-ppo" else 8
    agent = MLPPPOAgent(obs_dim)
    buf = RolloutBuffer(n_steps=ROLLOUT_STEPS, n_agents=n_gnb)

    obs_arr, _ = env.reset(seed=seed)
    ep_reward = 0.0
    ep_count = 0

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "episode", "ep_reward", "loss"])
        t0 = time.time()

        for step in range(steps):
            if algo == "central-ppo":
                obs_in = obs_arr.flatten()
                actions, log_probs, values = agent.act(obs_in)
                actions_env = np.full(n_gnb, int(actions[0]) if hasattr(actions, '__len__') else int(actions))
            else:
                obs_stack = obs_arr  # (N, 8)
                actions, log_probs, values = agent.act(obs_stack)
                actions_env = actions

            next_obs, reward, terminated, truncated, info = env.step(actions_env)
            done = terminated or truncated
            ep_reward += reward

            buf.add(obs_arr.copy(), actions, log_probs, reward, values, done)

            if done:
                ep_count += 1
                obs_arr, _ = env.reset(seed=seed + ep_count)
            else:
                obs_arr = next_obs

            loss_info = None
            if buf.is_full():
                rollout = buf.compute_advantages()
                loss_info = agent.learn(rollout) if hasattr(agent, "learn") else {}
                buf.clear()
                if step % 1000 == 0:
                    elapsed = time.time() - t0
                    print(f"[{algo}] step={step:>7} ep={ep_count:>4} "
                          f"rew={ep_reward:+.3f} t={elapsed:.0f}s")
                writer.writerow([step, ep_count, round(ep_reward, 4),
                                 round(loss_info.get("loss", 0), 6) if loss_info else ""])
                ep_reward = 0.0

    env.close()
    print(f"[{algo}] Training done → {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["idqn", "central-dqn", "ippo", "central-ppo"],
                        default="idqn")
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    log_dir = Path("results/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{args.algo}_seed{args.seed}.csv"

    if "dqn" in args.algo:
        train_dqn(args.algo, args.steps, args.seed, log_path)
    else:
        train_ppo(args.algo, args.steps, args.seed, log_path)
