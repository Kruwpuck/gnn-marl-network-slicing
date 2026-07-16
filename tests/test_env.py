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
        assert -10.0 <= reward <= 10.0, f"Reward out of bounds: {reward}"


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


def test_neighbor_allocation_couples_reward(env):
    """Own action fixed at gNB 0, but changing a neighbor's URLLC allocation must
    change gNB 0's SINR/reward — proof interference is coupled across agents, not
    independent per-gNB (env.slice_coupled_interference=True)."""
    assert env.slice_coupled_interference

    env.reset(seed=7)
    actions_low_neighbor = [5] + [0] * (env.n_gnb - 1)
    _, reward_low, _, _, info_low = env.step(actions_low_neighbor)

    env.reset(seed=7)
    actions_high_neighbor = [5] + [10] * (env.n_gnb - 1)
    _, reward_high, _, _, info_high = env.step(actions_high_neighbor)

    assert not np.isclose(info_low["embb_rates"][0], info_high["embb_rates"][0]), (
        "gNB 0's own eMBB rate should change when neighbors' URLLC allocation changes "
        "under coupled interference"
    )


def test_reward_not_saturated_by_extreme_delay():
    """Reward must keep decreasing as delay worsens (no flat clip plateau like v1's
    [-2, 2] clip), otherwise there's no gradient to escape a starvation local optimum."""
    env = NetworkSlicingEnv()
    embb_rates = np.array([4e6] * env.n_gnb)
    r_moderate = env._compute_reward(embb_rates, np.full(env.n_gnb, env.urllc_max_delay_ms * 5))
    r_severe = env._compute_reward(embb_rates, np.full(env.n_gnb, env.urllc_max_delay_ms * 500))
    r_extreme = env._compute_reward(embb_rates, np.full(env.n_gnb, env.urllc_max_delay_ms * 50000))
    assert r_moderate > r_severe > r_extreme, (
        f"Reward should strictly worsen with delay, got moderate={r_moderate} "
        f"severe={r_severe} extreme={r_extreme}"
    )
    env.close()
