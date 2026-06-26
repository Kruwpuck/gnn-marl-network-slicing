import sys
sys.path.insert(0, "/home/habb/Kuliah/gnn-marl-network-slicing")

import numpy as np
import pytest
from pathlib import Path
import tempfile, csv

from evaluation.convergence_eval import ConvergenceEvaluator
from evaluation.network_perf_eval import NetworkPerfEvaluator, jains_fairness
from evaluation.zero_shot_eval import ZeroShotEvaluator


# --- helpers ---

def _write_fake_log(path, n_rows=50):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "episode", "ep_reward", "epsilon", "loss"])
        for i in range(n_rows):
            w.writerow([i * 1000, i + 1, round(np.sin(i / 5) * 0.5 + 0.3, 4),
                        round(max(0.05, 1.0 - i * 0.02), 4), round(0.1 / (i + 1), 6)])


# --- ConvergenceEvaluator ---

def test_convergence_load_log():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "idqn_seed42.csv"
        _write_fake_log(p)
        ev = ConvergenceEvaluator(tmp)
        log = ev.load_log_path(p)
        assert len(log["steps"]) == 50
        assert len(log["rewards"]) == 50


def test_moving_average_shape():
    ev = ConvergenceEvaluator()
    arr = np.ones(200)
    ma = ev.moving_average(arr, window=50)
    assert len(ma) == 200 - 50 + 1


def test_moving_average_short_array():
    ev = ConvergenceEvaluator()
    arr = np.array([1.0, 2.0, 3.0])
    ma = ev.moving_average(arr, window=10)
    assert len(ma) == 3  # falls back to raw


def test_sample_efficiency():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.csv"
        _write_fake_log(p, n_rows=1100)
        ev = ConvergenceEvaluator(tmp)
        log = ev.load_log_path(p)
        se = ev.sample_efficiency(log)
        assert set(se.keys()) == {100_000, 500_000, 1_000_000}


def test_stability_is_float():
    ev = ConvergenceEvaluator()
    log = {"steps": np.arange(200) * 1000, "rewards": np.random.randn(200)}
    s = ev.stability(log)
    assert isinstance(s, float)


def test_compute_all_returns_keys():
    ev = ConvergenceEvaluator()
    log = {"steps": np.arange(50) * 1000, "rewards": np.random.randn(50), "losses": np.array([])}
    metrics = ev.compute_all({"algo_a": log, "algo_b": log})
    assert "algo_a" in metrics and "algo_b" in metrics
    assert "final_reward" in metrics["algo_a"]


# --- Jain's Fairness ---

def test_jains_perfect_fairness():
    rates = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    assert abs(jains_fairness(rates) - 1.0) < 1e-6


def test_jains_worst_fairness():
    rates = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    assert jains_fairness(rates) == pytest.approx(0.2, abs=1e-4)


def test_jains_all_zero():
    rates = np.zeros(5)
    assert jains_fairness(rates) == 0.0


# --- NetworkPerfEvaluator ---

def test_network_perf_gnn_agent_runs():
    from gnn import GATBackbone
    from agents.dqn_agent import DQNAgent
    backbone = GATBackbone()
    agent = DQNAgent(backbone)
    ev = NetworkPerfEvaluator()
    m = ev.evaluate_gnn_agent(agent, n_episodes=2, seed=0)
    assert "embb_throughput_mbps_mean" in m
    assert "sla_violation_rate_pct" in m
    assert 0.0 <= m["jains_fairness"] <= 1.0


def test_network_perf_mlp_agent_runs():
    from agents.mlp_agent import MLPDQNAgent
    agent = MLPDQNAgent(obs_dim=8)
    ev = NetworkPerfEvaluator()
    m = ev.evaluate_mlp_agent(agent, n_episodes=2, seed=0)
    assert "urllc_delay_ms_mean" in m
    assert m["n_episodes"] == 2


# --- ZeroShotEvaluator ---

def test_zero_shot_evaluator_n5():
    from gnn import SAGEBackbone
    from agents.dqn_agent import DQNAgent
    agent = DQNAgent(SAGEBackbone())
    ev = ZeroShotEvaluator()
    reward = ev.evaluate_one(agent, n_gnb=5, n_episodes=1, seed=0)
    assert isinstance(reward, float)


def test_zero_shot_evaluator_n10():
    from gnn import SAGEBackbone
    from agents.dqn_agent import DQNAgent
    agent = DQNAgent(SAGEBackbone())
    ev = ZeroShotEvaluator()
    reward = ev.evaluate_one(agent, n_gnb=10, n_episodes=1, seed=0)
    assert isinstance(reward, float)


def test_retention_ratio():
    rewards = {5: 1.0, 10: 0.8, 20: 0.6}
    prr = ZeroShotEvaluator.retention_ratio(rewards, base_n=5)
    assert prr[5] == pytest.approx(100.0)
    assert prr[10] == pytest.approx(80.0)
    assert prr[20] == pytest.approx(60.0)


def test_retention_ratio_zero_base():
    prr = ZeroShotEvaluator.retention_ratio({5: 0.0, 10: 0.5}, base_n=5)
    assert np.isnan(prr[10])
