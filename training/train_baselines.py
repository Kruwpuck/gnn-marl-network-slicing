"""Train baseline agents (B1: centralized-MLP, B2: IDQN/IPPO)."""
from __future__ import annotations
import argparse, csv, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.network_slicing_env import NetworkSlicingEnv
from agents.mlp_agent import MLPDQNAgent, MLPPPOAgent
from training.replay_buffer import ReplayBuffer
from training.rollout_buffer import RolloutBuffer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
REPLAY_START = 1000
ROLLOUT_STEPS = 512


def make_env(seed):
    env = NetworkSlicingEnv(); env.reset(seed=seed); return env


def train_dqn(algo, steps, seed, log_path):
    np.random.seed(seed)
    env = make_env(seed)
    n_gnb = env.n_gnb
    obs_dim = n_gnb * 8 if algo == "central-dqn" else 8
    agent = MLPDQNAgent(obs_dim).to(DEVICE)
    print(f"[{algo}] device={DEVICE}")
    buf = ReplayBuffer(50_000)
    obs_arr, _ = env.reset(seed=seed)
    ep_reward = ep_step = ep_count = 0

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "episode", "ep_reward", "epsilon", "loss"])
        t0 = time.time()
        for step in range(steps):
            if algo == "central-dqn":
                raw = agent.act(obs_arr.flatten())
                actions = np.full(n_gnb, int(raw.flat[0]))
                buf.push(obs_arr.flatten(), actions, 0.0, obs_arr.flatten(), False)  # placeholder
            else:
                actions = np.array([int(agent.act(obs_arr[i]).flat[0]) for i in range(n_gnb)])
                for i in range(n_gnb):
                    buf.push(obs_arr[i], np.array([actions[i]]), 0.0, obs_arr[i], False)

            next_obs, reward, terminated, truncated, _ = env.step(actions)
            done = terminated or truncated
            # overwrite reward in last-pushed items
            if algo == "central-dqn":
                buf._buf[-1] = (obs_arr.flatten(), actions, reward, next_obs.flatten(), done)
            else:
                for i in range(n_gnb):
                    buf._buf[-(n_gnb - i)] = (obs_arr[i], np.array([actions[i]]),
                                               reward/n_gnb, next_obs[i], done)

            ep_reward += reward; ep_step += 1; obs_arr = next_obs
            loss = agent.learn(buf.sample(BATCH_SIZE)) if len(buf) >= REPLAY_START else None

            if done:
                ep_count += 1
                if step % 1000 == 0:
                    print(f"[{algo}] step={step:>7} ep={ep_count:>4} rew={ep_reward:+.3f} "
                          f"eps={agent.epsilon:.3f} loss={f'{loss:.4f}' if loss else 'N/A'} "
                          f"t={time.time()-t0:.0f}s")
                writer.writerow([step, ep_count, round(ep_reward,4),
                                 round(agent.epsilon,4), round(loss,6) if loss else ""])
                obs_arr, _ = env.reset(seed=seed+ep_count)
                ep_reward = ep_step = 0

    env.close()
    model_path = log_path.with_suffix(".pt")
    torch.save({"state_dict": agent.state_dict(), "algo": algo,
                "obs_dim": obs_dim, "steps": steps, "seed": seed}, model_path)
    print(f"[{algo}] done → {log_path} | model → {model_path}")


def train_ppo(algo, steps, seed, log_path):
    np.random.seed(seed)
    env = make_env(seed)
    n_gnb = env.n_gnb
    obs_dim = n_gnb * 8 if algo == "central-ppo" else 8
    agent = MLPPPOAgent(obs_dim).to(DEVICE)
    print(f"[{algo}] device={DEVICE}")
    buf = RolloutBuffer(n_steps=ROLLOUT_STEPS, n_agents=n_gnb)
    obs_arr, _ = env.reset(seed=seed)
    ep_reward = ep_count = 0

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "episode", "ep_reward", "loss"])
        t0 = time.time()
        for step in range(steps):
            obs_in = obs_arr.flatten() if algo == "central-ppo" else obs_arr
            actions, log_probs, values = agent.act(obs_in)
            actions_env = (np.full(n_gnb, int(actions.flat[0]))
                           if algo == "central-ppo" else actions)
            next_obs, reward, terminated, truncated, _ = env.step(actions_env)
            done = terminated or truncated
            ep_reward += reward
            buf.add(obs_arr.copy(), actions, log_probs, reward, values, done)
            obs_arr = next_obs if not done else env.reset(seed=seed+(ep_count:=ep_count+1))[0]
            if buf.is_full():
                loss_info = agent.learn(buf.compute_advantages()) if hasattr(agent,"learn") else {}
                buf.clear()
                if step % 1000 == 0:
                    print(f"[{algo}] step={step:>7} ep={ep_count:>4} rew={ep_reward:+.3f} "
                          f"t={time.time()-t0:.0f}s")
                writer.writerow([step, ep_count, round(ep_reward,4),
                                 round(loss_info.get("loss",0),6) if loss_info else ""])
                ep_reward = 0

    env.close()
    model_path = log_path.with_suffix(".pt")
    torch.save({"state_dict": agent.state_dict(), "algo": algo,
                "obs_dim": obs_dim, "steps": steps, "seed": seed}, model_path)
    print(f"[{algo}] done → {log_path} | model → {model_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["idqn","central-dqn","ippo","central-ppo"], default="idqn")
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    log_dir = Path("results/logs"); log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{args.algo}_seed{args.seed}.csv"
    (train_dqn if "dqn" in args.algo else train_ppo)(args.algo, args.steps, args.seed, log_path)
