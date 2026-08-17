"""The YAML hyperparameters must stay bit-identical to the Python defaults the v4 wave ran on.

Moving them into configs/experiment_config.yaml (2026-08-16) was a reproducibility fix for
goal1.md C2, not a retune. If a value drifts, results/eval/*_v4_* and every Gate B number
silently stop being reproducible from the committed config -- the failure is invisible
without a check like this one. The literals below are the pre-move defaults, read off
agents/*.py at commit 04f4773.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.dqn_agent import DQNAgent
from agents.hparams import hparams, resolve
from agents.mlp_agent import MLPDQNAgent, MLPPPOAgent
from agents.ppo_agent import PPOAgent
from gnn.sage_backbone import SAGEBackbone

PRE_MOVE_DQN = {
    "n_actions": 11, "hidden": 128, "epsilon": 1.0, "epsilon_min": 0.05,
    "epsilon_decay": 0.9995, "lr": 1e-3, "gamma": 0.99,
    "max_grad_norm": 10.0,      # was the literal in DQNAgent.learn / MLPDQNAgent.learn
    "batch_size": 64, "replay_capacity": 50_000, "replay_start": 1000,
}
PRE_MOVE_PPO = {
    "n_actions": 11, "hidden": 128, "lr": 3e-4, "clip_eps": 0.2,
    "entropy_coef": 0.01, "value_coef": 0.5, "max_grad_norm": 0.5,
    "rollout_steps": 512,
}


# Keys added to the YAML after the move. Each one must be listed here deliberately, so a
# stray addition still fails the test below instead of slipping in unnoticed.
POST_MOVE_ADDITIONS = {"knn_k"}  # mlp-knn-ppo neighbourhood size, added 2026-08-16


@pytest.mark.parametrize("family,expected", [("dqn", PRE_MOVE_DQN), ("ppo", PRE_MOVE_PPO)])
def test_yaml_matches_pre_move_defaults(family, expected):
    hp = hparams(family)
    for key, value in expected.items():
        assert hp[key] == value, f"{family}.{key} drifted from the value the wave ran on"
    assert set(hp) - set(expected) == POST_MOVE_ADDITIONS


def test_dqn_agents_built_from_yaml():
    for agent in (DQNAgent(SAGEBackbone()), MLPDQNAgent(obs_dim=8)):
        assert agent.n_actions == PRE_MOVE_DQN["n_actions"]
        assert agent.epsilon == PRE_MOVE_DQN["epsilon"]
        assert agent.epsilon_min == PRE_MOVE_DQN["epsilon_min"]
        assert agent.epsilon_decay == PRE_MOVE_DQN["epsilon_decay"]
        assert agent.gamma == PRE_MOVE_DQN["gamma"]
        assert agent.max_grad_norm == PRE_MOVE_DQN["max_grad_norm"]
        assert agent.optimizer.param_groups[0]["lr"] == PRE_MOVE_DQN["lr"]
        assert agent.adv_stream[-1].out_features == PRE_MOVE_DQN["n_actions"]
        assert agent.adv_stream[0].out_features == PRE_MOVE_DQN["hidden"]


def test_ppo_agents_built_from_yaml():
    for agent in (PPOAgent(SAGEBackbone()), MLPPPOAgent(obs_dim=8)):
        assert agent.n_actions == PRE_MOVE_PPO["n_actions"]
        assert agent.clip_eps == PRE_MOVE_PPO["clip_eps"]
        assert agent.entropy_coef == PRE_MOVE_PPO["entropy_coef"]
        assert agent.value_coef == PRE_MOVE_PPO["value_coef"]
        assert agent.max_grad_norm == PRE_MOVE_PPO["max_grad_norm"]
        assert agent.optimizer.param_groups[0]["lr"] == PRE_MOVE_PPO["lr"]
        assert agent.actor[0].out_features == PRE_MOVE_PPO["hidden"]
        assert agent.actor[-1].out_features == PRE_MOVE_PPO["n_actions"]


def test_dqn_and_ppo_families_share_the_non_optimizer_settings():
    """C2's point: what differs between families must be structural, not tuned. n_actions
    and hidden are shared; the rest differ because the two solvers are different, the same
    way the 200K/1M step budgets do."""
    for key in ("n_actions", "hidden"):
        assert hparams("dqn")[key] == hparams("ppo")[key]


def test_unknown_override_is_rejected():
    """A silently-dropped typo would make a run report a hyperparameter it never used."""
    with pytest.raises(ValueError, match="unknown ppo hyperparameter"):
        resolve("ppo", None, {"clip_epsilon": 0.1})
