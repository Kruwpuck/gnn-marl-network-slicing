import sys
sys.path.insert(0, "/home/habb/Kuliah/gnn-marl-network-slicing")

import numpy as np
import pytest
import torch
from gnn import GATBackbone, SAGEBackbone
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from agents.mlp_agent import MLPDQNAgent, MLPPPOAgent
from training.replay_buffer import ReplayBuffer
from training.rollout_buffer import RolloutBuffer


def _make_graph(n: int = 5) -> dict:
    x = np.random.randn(n, 8).astype("float32")
    src = [i for i in range(n) for j in range(n) if i != j]
    dst = [j for i in range(n) for j in range(n) if i != j]
    edge_index = np.array([src, dst], dtype="int64")
    edge_attr = np.abs(np.random.randn(n * (n - 1), 1).astype("float32")) * 100
    return {"x": x, "edge_index": edge_index, "edge_attr": edge_attr}


@pytest.fixture
def gnn_dqn():
    return DQNAgent(GATBackbone())


@pytest.fixture
def gnn_ppo():
    return PPOAgent(GATBackbone())


# --- DQN ---

def test_dqn_act_shape(gnn_dqn):
    graph = _make_graph()
    actions = gnn_dqn.act(graph)
    assert actions.shape == (5,), f"Got {actions.shape}"


def test_dqn_act_values_in_range(gnn_dqn):
    graph = _make_graph()
    actions = gnn_dqn.act(graph)
    assert all(0 <= a <= 10 for a in actions), f"Out-of-range: {actions}"


def test_dqn_q_shape(gnn_dqn):
    graph = _make_graph()
    q = gnn_dqn.q_values(graph)
    assert q.shape == (5, 11)


def test_dqn_greedy_act(gnn_dqn):
    gnn_dqn.epsilon = 0.0
    graph = _make_graph()
    a1 = gnn_dqn.act(graph, greedy=True)
    a2 = gnn_dqn.act(graph, greedy=True)
    np.testing.assert_array_equal(a1, a2)


def test_dqn_learn_returns_float(gnn_dqn):
    graph = _make_graph()
    actions = np.array([5, 3, 7, 2, 9])
    batch = [(graph, actions, 0.5, graph, False)] * 4
    loss = gnn_dqn.learn(batch)
    assert isinstance(loss, float)
    assert not np.isnan(loss)


def test_dqn_epsilon_decay(gnn_dqn):
    gnn_dqn.epsilon = 1.0
    gnn_dqn.decay_epsilon()
    assert gnn_dqn.epsilon < 1.0


# --- PPO ---

def test_ppo_act_returns_tuple(gnn_ppo):
    graph = _make_graph()
    result = gnn_ppo.act(graph)
    assert len(result) == 3


def test_ppo_actions_shape(gnn_ppo):
    graph = _make_graph()
    actions, log_probs, values = gnn_ppo.act(graph)
    assert actions.shape == (5,)
    assert log_probs.shape == (5,)
    assert values.shape == (5,)


def test_ppo_actions_in_range(gnn_ppo):
    graph = _make_graph()
    actions, _, _ = gnn_ppo.act(graph)
    assert all(0 <= a <= 10 for a in actions)


def test_ppo_learn_returns_dict(gnn_ppo):
    graph = _make_graph()
    n, T = 5, 8
    actions, log_probs, values = gnn_ppo.act(graph)
    rollout = {
        "graph_dicts": [graph] * T,
        "actions": np.stack([actions] * T),
        "old_log_probs": np.stack([log_probs] * T),
        "advantages": np.random.randn(T, n).astype("float32"),
        "returns": np.random.randn(T, n).astype("float32"),
    }
    result = gnn_ppo.learn(rollout)
    assert "loss" in result
    assert not np.isnan(result["loss"])


# --- MLP baselines ---

def test_mlp_dqn_act():
    agent = MLPDQNAgent(obs_dim=8)
    obs = np.random.randn(5, 8).astype("float32")
    actions = agent.act(obs)
    assert actions.shape == (5,)


def test_mlp_ppo_act():
    agent = MLPPPOAgent(obs_dim=8)
    obs = np.random.randn(5, 8).astype("float32")
    actions, log_probs, values = agent.act(obs)
    assert actions.shape == (5,)


# --- Replay buffer ---

def test_replay_buffer_push_sample():
    buf = ReplayBuffer(capacity=100)
    graph = _make_graph()
    for _ in range(50):
        buf.push(graph, np.array([5]*5), 0.1, graph, False)
    assert len(buf) == 50
    batch = buf.sample(32)
    assert len(batch) == 32


def test_replay_buffer_circular():
    buf = ReplayBuffer(capacity=10)
    graph = _make_graph()
    for i in range(20):
        buf.push(graph, np.array([i % 11]*5), float(i), graph, False)
    assert len(buf) == 10  # old entries evicted


# --- Rollout buffer ---

def test_rollout_buffer_fill_and_gae():
    n_gnb = 5
    buf = RolloutBuffer(n_steps=20, n_agents=n_gnb)
    graph = _make_graph(n_gnb)
    for _ in range(20):
        buf.add(graph, np.zeros(n_gnb), np.zeros(n_gnb), 0.1, np.zeros(n_gnb), False)
    rollout = buf.compute_advantages()
    assert rollout["advantages"].shape == (20, n_gnb)
    assert rollout["returns"].shape == (20, n_gnb)
    assert rollout["actions"].shape == (20, n_gnb)
