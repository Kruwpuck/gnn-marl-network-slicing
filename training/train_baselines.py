"""Train baseline agents (B1: centralized-MLP, B2: IDQN/IPPO).

Rich CSV logging + resumable checkpoints, mirroring training/train_proposed.py.
The MLP-PPO baselines (ippo / central-ppo) now actually update (see MLPPPOAgent.learn).
"""
from __future__ import annotations
import argparse, sys, time
from collections import deque
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.network_slicing_env import NetworkSlicingEnv
from agents.mlp_agent import MLPDQNAgent, MLPPPOAgent
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


def make_env(seed, config_path=None):
    env = NetworkSlicingEnv(config_path=config_path); env.reset(seed=seed); return env


def _maybe_resume(ckpt, agent, env, resume):
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
    env.set_cmdp_state(state.get("cmdp", {}))
    ma = deque(state.get("ma_deque", []), maxlen=100)
    print(f"[resume] {ckpt.run_name}: from step {state['step']} ep {state['episode']}")
    return int(state["step"]), int(state["episode"]), ma, float(state.get("elapsed_sec", 0.0))


def _save_ckpt(ckpt, agent, env, step, episode, logger, run_name, **meta):
    ma = float(np.mean(logger.ma_deque)) if logger.ma_deque else None
    state = {
        "model": agent.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
        "step": step,
        "episode": episode,
        "rng": capture_rng_state(),
        "cmdp": env.get_cmdp_state(),
        "ma_deque": list(logger.ma_deque),
        "elapsed_sec": logger.elapsed(),
        "run_name": run_name,
    }
    if hasattr(agent, "epsilon"):
        state["epsilon"] = agent.epsilon
    state.update(meta)
    ckpt.save(state, ma_reward=ma)


def _save_model(log_path, agent, **meta):
    torch.save({"state_dict": agent.state_dict(), **meta},
               Path(log_path).with_suffix(".pt"))


def train_dqn(algo, steps, seed, log_path, ckpt_interval, resume, config_path, run_name):
    np.random.seed(seed)
    env = make_env(seed, config_path)
    n_gnb = env.n_gnb
    obs_dim = n_gnb * 8 if algo == "central-dqn" else 8
    agent = MLPDQNAgent(obs_dim).to(DEVICE)
    print(f"[{algo}] device={DEVICE}")
    buf = ReplayBuffer(50_000)

    ckpt = CheckpointManager(run_name)
    start_step, ep_count, ma_deque, elapsed_off = _maybe_resume(ckpt, agent, env, resume)
    logger = MetricsLogger(log_path, resume=resume, ma_deque=ma_deque)
    logger.set_elapsed_offset(elapsed_off)
    netstats = EpisodeNetworkStats()

    obs_arr, _ = env.reset(seed=seed + ep_count)
    ep_reward = 0.0
    t0 = time.time()

    for step in range(start_step, steps):
        if algo == "central-dqn":
            raw = agent.act(obs_arr.flatten())
            central_action = int(raw) if np.isscalar(raw) else int(np.asarray(raw).flat[0])
            actions = np.full(n_gnb, central_action)
            buf.push(obs_arr.flatten(), np.array([central_action]), 0.0, obs_arr.flatten(), False)
        else:
            _acts = [agent.act(obs_arr[i]) for i in range(n_gnb)]
            actions = np.array([int(a) if np.isscalar(a) else int(np.asarray(a).flat[0]) for a in _acts])
            for i in range(n_gnb):
                buf.push(obs_arr[i], np.array([actions[i]]), 0.0, obs_arr[i], False)

        next_obs, reward, terminated, truncated, info = env.step(actions)
        done = terminated or truncated
        netstats.add(info)
        # overwrite placeholder reward in the just-pushed items
        if algo == "central-dqn":
            buf._buf[-1] = (obs_arr.flatten(), np.array([central_action]), reward, next_obs.flatten(), done)
        else:
            for i in range(n_gnb):
                buf._buf[-(n_gnb - i)] = (obs_arr[i], np.array([actions[i]]),
                                          reward / n_gnb, next_obs[i], done)

        ep_reward += reward
        obs_arr = next_obs
        loss = agent.learn(buf.sample(BATCH_SIZE)) if len(buf) >= REPLAY_START else None

        if done:
            ep_count += 1
            net = netstats.summary(env.bandwidth_hz, env.urllc_max_delay_ms, env.slot_s)
            ma = logger.log(step, ep_count, ep_reward, loss=loss,
                            epsilon=agent.epsilon, **net)
            netstats.reset()
            if step % 1000 == 0:
                print(f"[{algo}] step={step:>7} ep={ep_count:>4} rew={ep_reward:+.3f} "
                      f"ma={ma:+.3f} eps={agent.epsilon:.3f} "
                      f"loss={f'{loss:.4f}' if loss else 'N/A'} lam={env._lam:.3f} "
                      f"t={time.time()-t0:.0f}s")
            obs_arr, _ = env.reset(seed=seed + ep_count)
            ep_reward = 0.0

        if ckpt_interval and step > start_step and step % ckpt_interval == 0:
            _save_ckpt(ckpt, agent, env, step, ep_count, logger, run_name,
                       algo=algo, obs_dim=obs_dim, seed=seed)

    _save_ckpt(ckpt, agent, env, steps, ep_count, logger, run_name,
               algo=algo, obs_dim=obs_dim, seed=seed)
    logger.close()
    env.close()
    _save_model(log_path, agent, algo=algo, obs_dim=obs_dim, steps=steps, seed=seed)
    print(f"[{algo}] done -> {log_path}")


def train_ppo(algo, steps, seed, log_path, ckpt_interval, resume, config_path, run_name):
    np.random.seed(seed)
    env = make_env(seed, config_path)
    n_gnb = env.n_gnb
    is_central = algo == "central-ppo"
    obs_dim = n_gnb * 8 if is_central else 8
    agent = MLPPPOAgent(obs_dim).to(DEVICE)
    print(f"[{algo}] device={DEVICE}")
    buf = RolloutBuffer(n_steps=ROLLOUT_STEPS, n_agents=1 if is_central else n_gnb)

    ckpt = CheckpointManager(run_name)
    start_step, ep_count, ma_deque, elapsed_off = _maybe_resume(ckpt, agent, env, resume)
    logger = MetricsLogger(log_path, resume=resume, ma_deque=ma_deque)
    logger.set_elapsed_offset(elapsed_off)
    netstats = EpisodeNetworkStats()

    obs_arr, _ = env.reset(seed=seed + ep_count)
    ep_reward = 0.0
    t0 = time.time()

    for step in range(start_step, steps):
        if is_central:
            obs_in = obs_arr.flatten()
            actions, log_probs, values = agent.act(obs_in)   # each (1,)
            actions_env = np.full(n_gnb, int(actions[0]))
            stored_state = obs_in.reshape(1, -1)             # (1, obs_dim)
        else:
            actions, log_probs, values = agent.act(obs_arr)  # each (N,)
            actions_env = actions
            stored_state = obs_arr.copy()                    # (N, 8)

        next_obs, reward, terminated, truncated, info = env.step(actions_env)
        done = terminated or truncated
        ep_reward += reward
        netstats.add(info)
        buf.add(stored_state, actions, log_probs, reward, values, done)

        if done:
            ep_count += 1
            obs_arr, _ = env.reset(seed=seed + ep_count)
        else:
            obs_arr = next_obs

        if buf.is_full():
            loss_info = agent.learn(buf.compute_advantages())
            buf.clear()
            net = netstats.summary(env.bandwidth_hz, env.urllc_max_delay_ms, env.slot_s)
            ma = logger.log(step, ep_count, ep_reward,
                            loss=loss_info.get("loss"),
                            policy_loss=loss_info.get("policy_loss"),
                            value_loss=loss_info.get("value_loss"),
                            entropy=loss_info.get("entropy"), **net)
            netstats.reset()
            if step % 1000 < ROLLOUT_STEPS:
                print(f"[{algo}] step={step:>7} ep={ep_count:>4} rew={ep_reward:+.3f} "
                      f"ma={ma:+.3f} lam={env._lam:.3f} t={time.time()-t0:.0f}s")
            ep_reward = 0.0

        if ckpt_interval and step > start_step and step % ckpt_interval == 0:
            _save_ckpt(ckpt, agent, env, step, ep_count, logger, run_name,
                       algo=algo, obs_dim=obs_dim, seed=seed)

    _save_ckpt(ckpt, agent, env, steps, ep_count, logger, run_name,
               algo=algo, obs_dim=obs_dim, seed=seed)
    logger.close()
    env.close()
    _save_model(log_path, agent, algo=algo, obs_dim=obs_dim, steps=steps, seed=seed)
    print(f"[{algo}] done -> {log_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["idqn", "central-dqn", "ippo", "central-ppo"], default="idqn")
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ckpt-interval", type=int, default=25_000)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--config", type=str, default=None,
                   help="path to an experiment_config.yaml variant "
                        "(default: configs/experiment_config.yaml)")
    p.add_argument("--tag", type=str, default="",
                   help="suffix for run_name/log/checkpoint, e.g. _floornone")
    args = p.parse_args()

    run_name = f"{args.algo}{args.tag}_seed{args.seed}"
    log_dir = Path("results/logs"); log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_name}.csv"
    fn = train_dqn if "dqn" in args.algo else train_ppo
    fn(args.algo, args.steps, args.seed, log_path, args.ckpt_interval, args.resume,
       args.config, run_name)
