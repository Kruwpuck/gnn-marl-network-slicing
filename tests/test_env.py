import sys
sys.path.insert(0, "/home/habb/Kuliah/gnn-marl-network-slicing")

import numpy as np
import pytest
from envs.network_slicing_env import NetworkSlicingEnv
from envs.multi_agent_env import NetworkSlicingMAEnv


@pytest.fixture
def env():
    e = NetworkSlicingEnv()
    yield e
    e.close()


@pytest.fixture
def ma_env():
    e = NetworkSlicingMAEnv()
    yield e


def test_reset_returns_correct_shape(env):
    obs, info = env.reset(seed=42)
    assert obs.shape == (env.n_gnb, 8), f"Unexpected obs shape: {obs.shape}"
    assert obs.dtype == np.float32


def test_step_returns_correct_types(env):
    env.reset(seed=42)
    actions = [5] * env.n_gnb
    obs, reward, terminated, truncated, info = env.step(actions)
    assert obs.shape == (env.n_gnb, 8)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_reward_in_valid_range(env):
    env.reset(seed=42)
    for _ in range(10):
        actions = [5] * env.n_gnb
        _, reward, _, _, _ = env.step(actions)
        assert -2.0 <= reward <= 2.0, f"Reward out of bounds: {reward}"


def test_pyg_graph_keys_and_shapes(env):
    _, info = env.reset(seed=42)
    graph = info["graph"]
    assert "x" in graph and "edge_index" in graph and "edge_attr" in graph
    assert graph["x"].shape == (env.n_gnb, 8)
    assert graph["edge_index"].shape[0] == 2
    n = env.n_gnb
    assert graph["edge_index"].shape[1] == n * (n - 1)


def test_determinism(env):
    obs1, _ = env.reset(seed=42)
    for _ in range(5):
        env.step([5] * env.n_gnb)
    obs_a, _ = env.reset(seed=42)
    for _ in range(5):
        obs_a, _, _, _, _ = env.step([5] * env.n_gnb)

    env2 = NetworkSlicingEnv()
    env2.reset(seed=42)
    for _ in range(5):
        obs_b, _, _, _, _ = env2.step([5] * env.n_gnb)
    env2.close()

    np.testing.assert_array_equal(obs_a, obs_b)


def test_ma_env_obs_count(ma_env):
    obs, _ = ma_env.reset(seed=0)
    assert len(obs) == len(ma_env.possible_agents)
    for a in ma_env.possible_agents:
        assert obs[a].shape == (8,)


def test_ma_env_rewards_count(ma_env):
    ma_env.reset(seed=0)
    actions = {a: 5 for a in ma_env.agents}
    _, rewards, _, _, _ = ma_env.step(actions)
    assert len(rewards) == len(ma_env.possible_agents)


def test_truncation_after_episode_length():
    env = NetworkSlicingEnv()
    env.reset(seed=0)
    truncated = False
    for _ in range(env.episode_length):
        _, _, _, truncated, _ = env.step([5] * env.n_gnb)
    assert truncated
    env.close()
