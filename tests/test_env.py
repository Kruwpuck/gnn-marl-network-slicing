import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    change gNB 0's SINR/served throughput — proof interference is coupled across
    agents, not independent per-gNB (env.slice_coupled_interference=True). eMBB queue
    is pre-loaded with a large backlog so service is capacity-limited (not
    demand-limited) — otherwise a small per-slot arrival drains fully regardless of
    interference and the coupling signal doesn't show up in served throughput."""
    assert env.slice_coupled_interference

    env.reset(seed=7)
    env._queue_embb[:] = 1e9
    actions_low_neighbor = [5] + [0] * (env.n_gnb - 1)
    _, reward_low, _, _, info_low = env.step(actions_low_neighbor)

    env.reset(seed=7)
    env._queue_embb[:] = 1e9
    actions_high_neighbor = [5] + [10] * (env.n_gnb - 1)
    _, reward_high, _, _, info_high = env.step(actions_high_neighbor)

    assert not np.isclose(info_low["embb_rates"][0], info_high["embb_rates"][0]), (
        "gNB 0's own eMBB throughput should change when neighbors' URLLC allocation "
        "changes under coupled interference"
    )


def test_urllc_deadline_drop_under_starvation(env):
    """URLLC allocation pinned at 0% (tier 0, floor disabled) for longer than the
    deadline: sustained 0% allocation means nothing is ever delivered, so once the
    FIFO reaches steady state, every packet resolved each slot (aged-out deadline
    drops) is a violation — averaged over several steps (Poisson arrival variance
    makes any single step's ratio noisy)."""
    env.floor_mode = "none"
    env.reset(seed=3)
    deadline_slots = int(env.urllc_max_delay_ms / env.slot_ms)
    viol_rates = []
    delays = []
    for _ in range(deadline_slots + 20):
        _, _, _, _, info = env.step([0] * env.n_gnb)
        viol_rates.append(info["urllc_violation_rate"])
        delays.extend(d for gnb_delays in info["urllc_delivered_delays"] for d in gnb_delays)
    steady_state = np.array(viol_rates[deadline_slots:])  # skip warm-up before first deadline hit
    assert steady_state.mean() >= 0.9, (
        f"Expected near-total violation under sustained 0% URLLC allocation, "
        f"got mean={steady_state.mean()}"
    )
    assert not delays, "0% URLLC allocation should never deliver a packet"


def test_urllc_packet_conservation(env):
    """arrived == delivered + dropped_late + dropped_overflow + still-in-flight."""
    env.reset(seed=11)
    total_arrived = np.zeros(env.n_gnb, dtype=np.int64)
    total_delivered = np.zeros(env.n_gnb, dtype=np.int64)
    total_dropped = np.zeros(env.n_gnb, dtype=np.int64)
    for _ in range(50):
        _, _, _, _, info = env.step([3] * env.n_gnb)
        total_arrived += info["urllc_arrived"]
        total_delivered += np.array([len(d) for d in info["urllc_delivered_delays"]])
        total_dropped += info["urllc_dropped_late"] + info["urllc_dropped_overflow"]
    in_flight = np.array([sum(1 for _ in q) for q in env._urllc_fifo])
    np.testing.assert_array_equal(total_arrived, total_delivered + total_dropped + in_flight)


def test_urllc_delivered_delay_bounded_by_deadline(env):
    """No delivered packet's delay can exceed the deadline — enforced by the
    deadline-drop step running before service every slot."""
    env.reset(seed=13)
    for _ in range(100):
        _, _, _, _, info = env.step([4] * env.n_gnb)
        for gnb_delays in info["urllc_delivered_delays"]:
            for d in gnb_delays:
                assert d <= env.urllc_max_delay_ms + 1e-6


def test_lambda_increases_when_violation_above_delta(env):
    """CMDP dual variable must rise toward the constraint when the violation rate
    stays above delta, and must persist across reset() (not be wiped per-episode)."""
    env.cmdp_enabled = True
    env.floor_mode = "none"  # keep starvation genuine; a floor would self-mitigate it
    env.dual_update_every = 50
    env.delta = 0.01
    env.reset(seed=5)
    lam_before = env._lam
    for ep in range(3):
        env.reset(seed=5 + ep)
        for _ in range(50):
            env.step([0] * env.n_gnb)  # 0% URLLC -> near-total violation
    assert env._lam > lam_before, "lambda should increase under sustained SLA violation"


def test_lambda_persists_across_reset(env):
    env.cmdp_enabled = True
    env.dual_update_every = 20
    env.reset(seed=1)
    for _ in range(20):
        env.step([0] * env.n_gnb)
    lam_after_window = env._lam
    env.reset(seed=2)
    assert env._lam == lam_after_window, "lambda must not reset with the episode"


def test_floor_static_raises_low_allocation(env):
    """floor_mode=static must lift a below-floor action to floor_base, visible via
    the applied fraction in prev_alloc (next observation) and in the info fracs."""
    env.floor_mode = "static"
    env.floor_base = 0.2
    env.reset(seed=9)
    obs, _, _, _, _ = env.step([0] * env.n_gnb)  # tier 0 = 0% raw
    assert np.allclose(env._prev_alloc, env.floor_base, atol=1e-6)


def test_floor_none_vs_dynamic_differ_under_backlog(env):
    """With backlog built up, floor_mode='dynamic' should apply a strictly higher
    minimum URLLC fraction than floor_mode='none' for the same 0%-URLLC action."""
    env.floor_mode = "none"
    env.reset(seed=17)
    for _ in range(15):
        env.step([0] * env.n_gnb)
    _, _, _, _, info_none = env.step([0] * env.n_gnb)
    frac_none = env._prev_alloc.copy()

    env.floor_mode = "dynamic"
    env.reset(seed=17)
    for _ in range(15):
        env.step([0] * env.n_gnb)
    _, _, _, _, info_dyn = env.step([0] * env.n_gnb)
    frac_dyn = env._prev_alloc.copy()

    assert np.all(frac_dyn >= frac_none)
    assert np.any(frac_dyn > frac_none), "dynamic floor should exceed none under backlog"


def test_embb_reward_uses_served_bits_not_capacity(env):
    """Regression for the v2 bug: reward/KPI must be based on bits actually drained
    from the eMBB queue, not raw Shannon capacity. An empty eMBB queue with no new
    arrivals and full eMBB bandwidth allocated must yield exactly zero served
    throughput, however large the channel capacity is."""
    env.reset(seed=21)
    env._queue_embb[:] = 0.0
    for traffic in env._embb_traffic:
        traffic.lambda_burst = 0.0
        traffic.lambda_idle = 0.0
    _, _, _, _, info = env.step([0] * env.n_gnb)  # tier 0 -> 100% eMBB bandwidth
    assert np.all(info["embb_rates"] < 1e-6), (
        f"Expected exactly zero served eMBB throughput with an empty queue and no "
        f"arrivals, got {info['embb_rates']}"
    )


def test_obs_shape_and_dtype(env):
    obs, _ = env.reset(seed=42)
    assert obs.shape == (env.n_gnb, 8)
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))
