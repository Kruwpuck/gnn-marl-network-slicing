import sys
sys.path.insert(0, "/home/habb/Kuliah/gnn-marl-network-slicing")

import numpy as np
import pytest
from envs.channel_model import ChannelModel


@pytest.fixture
def uma_model():
    return ChannelModel(scenario="UMa")


@pytest.fixture
def umi_model():
    return ChannelModel(scenario="UMi")


def test_path_loss_uma_los_increases_with_distance(uma_model):
    pl_near = uma_model.compute_path_loss(d3d=50, d2d=50, los=True)
    pl_far = uma_model.compute_path_loss(d3d=500, d2d=500, los=True)
    assert pl_far > pl_near


def test_path_loss_nlos_ge_los(uma_model):
    pl_los = uma_model.compute_path_loss(d3d=200, d2d=200, los=True)
    pl_nlos = uma_model.compute_path_loss(d3d=200, d2d=200, los=False)
    assert pl_nlos >= pl_los


def test_path_loss_umi_los(umi_model):
    pl = umi_model.compute_path_loss(d3d=100, d2d=100, los=True)
    assert 60 < pl < 160, f"Unexpected path-loss: {pl}"


def test_sinr_positive_for_close_ue(uma_model):
    rx_dbm = uma_model.compute_rx_power_dbm(pl_db=80.0)
    sinr = uma_model.compute_sinr(rx_dbm, interference_dbm=-200.0, bandwidth_hz=10e6)
    assert sinr > 0


def test_shannon_rate_positive(uma_model):
    rate = uma_model.compute_rate(sinr_linear=5.0, bandwidth_hz=10e6)
    assert rate > 0


def test_shadow_fading_changes_pl(uma_model):
    rng = np.random.default_rng(0)
    pl_base = uma_model.compute_path_loss(100, 100, True)
    pl_faded = uma_model.add_shadow_fading(pl_base, True, rng)
    assert pl_faded != pl_base


def test_build_channel_matrix_shape(uma_model):
    rng = np.random.default_rng(42)
    gnb_pos = rng.uniform(0, 500, (5, 2))
    ue_pos = rng.uniform(0, 500, (25, 2))
    mat = uma_model.build_channel_matrix(gnb_pos, ue_pos, rng)
    assert mat.shape == (5, 25)


def test_build_interference_graph(uma_model):
    rng = np.random.default_rng(0)
    gnb_pos = rng.uniform(0, 500, (5, 2))
    edge_index, edge_attr = uma_model.build_interference_graph(gnb_pos)
    assert edge_index.shape == (2, 20)  # 5*(5-1) = 20 directed edges
    assert edge_attr.shape == (20, 1)
