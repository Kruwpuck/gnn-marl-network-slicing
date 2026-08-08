"""
Why the greedy readout breaks: measure how much probability mass the argmax
action actually carries.

Gate A round 5 found greedy and stochastic evaluation of the SAME checkpoint
diverging by 2-3 pp across most of the lambda sweep and by 88 pp at lambda=30.
Training-time entropy alone does not explain it -- it drifts down gently
(2.29 -> 2.06 nats against a ln(11)=2.398 ceiling) while the gap explodes. The
sharper question is per-state: if the policy is near-uniform, argmax selects on
logit noise and the deterministic readout measures nothing about the policy.

Reported per checkpoint, over held-out eval states (same seeds as
evaluate_checkpoints.py, EVAL_SEED_BASE=10000):

  p_max      mean probability of the argmax action  (uniform = 1/n_actions)
  margin     mean p(top1) - p(top2)
  entropy    mean entropy of the action distribution, nats
  agree      fraction of steps where argmax == the sampled action

States are visited under the SAMPLED policy: that is the distribution training
optimised, and the one the dual controlled. Rolling out greedy instead would
measure the distribution the degenerate readout drags the policy into, which is
the thing under test, not the yardstick.

PPO only -- DQN has Q-values, not a distribution, so p_max/entropy have no
meaning there. Extend if a DQN readout ever needs the same defence.

Usage:
  python scripts/policy_confidence.py --runs "results/logs/ippo_calib*.pt" --episodes 20
"""
from __future__ import annotations
import argparse
import csv
import glob
import sys
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.network_slicing_env import NetworkSlicingEnv
from scripts.evaluate_checkpoints import EVAL_SEED_BASE, load_agent


def _logits(agent, kind: str, obs: np.ndarray, info: dict) -> torch.Tensor:
    """(n_gnb, n_actions) action logits for the current state."""
    if kind == "gnn-ppo":
        logits, _ = agent(info["graph"])
        return logits
    device = next(agent.parameters()).device
    flat = kind == "mlp-ppo-central"
    x = torch.as_tensor(obs.flatten() if flat else obs, dtype=torch.float32).to(device)
    logits, _ = agent(x)
    return logits.unsqueeze(0) if flat else logits


def measure(pt_path: Path, episodes: int, config_path: str | None) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent, algo, kind = load_agent(pt_path, device)
    if "ppo" not in kind:
        raise ValueError(f"{algo}: not a policy-gradient agent, no action distribution")

    env = NetworkSlicingEnv(config_path=config_path)
    env.cmdp_enabled = False

    p_max, margin, ent, agree = [], [], [], []
    for ep in range(episodes):
        obs, info = env.reset(seed=EVAL_SEED_BASE + ep)
        done = False
        while not done:
            with torch.no_grad():
                logits = _logits(agent, kind, obs, info)
                probs = torch.softmax(logits, dim=-1)
                top2 = probs.topk(2, dim=-1).values
                dist = Categorical(logits=logits)
                sampled = dist.sample()
            greedy = logits.argmax(dim=-1)
            p_max.append(float(top2[:, 0].mean()))
            margin.append(float((top2[:, 0] - top2[:, 1]).mean()))
            ent.append(float(dist.entropy().mean()))
            agree.append(float((greedy == sampled).float().mean()))

            actions = sampled.cpu().numpy()
            if kind == "mlp-ppo-central":
                actions = np.full(env.n_gnb, int(actions[0]))
            obs, _, terminated, truncated, info = env.step(actions)
            done = terminated or truncated

    n_actions = int(probs.shape[-1])
    env.close()
    return {
        "run": pt_path.stem, "algo": algo, "n_actions": n_actions,
        "p_uniform": 1.0 / n_actions, "ent_max": float(np.log(n_actions)),
        "p_max": float(np.mean(p_max)), "margin": float(np.mean(margin)),
        "entropy": float(np.mean(ent)), "agree": float(np.mean(agree)),
        "steps": len(p_max),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=str, required=True, help="glob for .pt checkpoints")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--out", type=str, default="results/policy_confidence.csv")
    args = p.parse_args()

    rows = []
    for pt_path in sorted(glob.glob(args.runs)):
        try:
            rows.append(measure(Path(pt_path), args.episodes, args.config))
        except ValueError as e:
            print(f"SKIP {e}")

    if not rows:
        return
    hdr = f"{'run':<28}{'p_max':>8}{'unif':>8}{'margin':>8}{'entropy':>9}{'ent_max':>9}{'agree':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['run']:<28}{r['p_max']:>8.3f}{r['p_uniform']:>8.3f}{r['margin']:>8.3f}"
              f"{r['entropy']:>9.3f}{r['ent_max']:>9.3f}{r['agree']:>8.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
