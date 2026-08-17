"""The k-NN adaptation baseline is only useful if its observation width is invariant to
n_gnb -- that is the single property that lets one set of weights run at 5, 10 and 20 gNB.
If it breaks, the baseline silently stops being a zero-shot control and the transfer
comparison goes back to GNN-vs-nothing.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.hparams import hparams
from agents.mlp_agent import OBS_FEATURES, MLPKNNPPOAgent, knn_features
from envs.network_slicing_env import NetworkSlicingEnv
from scripts.run_wave import make_variant_config

K = hparams("ppo")["knn_k"]


def graph_at(n_gnb: int) -> dict:
    cfg = make_variant_config("none", {"env.n_gnb": n_gnb}) if n_gnb != 5 else None
    env = NetworkSlicingEnv(config_path=str(cfg) if cfg else None)
    _, info = env.reset(seed=10_000)
    env.close()
    return info["graph"]


@pytest.mark.parametrize("n_gnb", [5, 10, 20])
def test_observation_width_is_independent_of_topology(n_gnb):
    feats = knn_features(graph_at(n_gnb), K)
    assert feats.shape == (n_gnb, OBS_FEATURES * (K + 1))


def test_agent_runs_at_every_topology_with_the_same_weights():
    agent = MLPKNNPPOAgent()
    assert agent.obs_dim == OBS_FEATURES * (K + 1)
    for n_gnb in (5, 10, 20):
        actions, log_probs, values = agent.act(agent.features(graph_at(n_gnb)))
        assert actions.shape == (n_gnb,)
        assert np.all((actions >= 0) & (actions < agent.n_actions))


def test_neighbours_are_ordered_by_interference_not_index():
    """Ranking is by path-loss (lower dB = stronger interferer). Falling back to node order
    would make the feature meaningless and the ordering topology-dependent."""
    x = np.arange(3 * OBS_FEATURES, dtype=np.float32).reshape(3, OBS_FEATURES)
    # node 0's neighbours: node 1 at 90 dB, node 2 at 70 dB -> node 2 must come first
    graph = {
        "x": x,
        "edge_index": np.array([[0, 0, 1, 2], [1, 2, 0, 0]]),
        "edge_attr": np.array([[90.0], [70.0], [90.0], [70.0]], dtype=np.float32),
    }
    feats = knn_features(graph, k=2)
    assert np.array_equal(feats[0, OBS_FEATURES:2 * OBS_FEATURES], x[2])
    assert np.array_equal(feats[0, 2 * OBS_FEATURES:3 * OBS_FEATURES], x[1])


def test_missing_neighbours_are_zero_padded():
    x = np.ones((2, OBS_FEATURES), dtype=np.float32)
    graph = {
        "x": x,
        "edge_index": np.array([[0, 1], [1, 0]]),
        "edge_attr": np.array([[80.0], [80.0]], dtype=np.float32),
    }
    feats = knn_features(graph, k=4)
    assert feats.shape == (2, OBS_FEATURES * 5)
    assert np.all(feats[:, 2 * OBS_FEATURES:] == 0.0)
