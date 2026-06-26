import sys
sys.path.insert(0, "/home/habb/Kuliah/gnn-marl-network-slicing")

import numpy as np
import pytest
import torch
from gnn import GATBackbone, SAGEBackbone, GCNBackbone, BACKBONES


def _make_graph(n: int) -> tuple:
    x = np.random.randn(n, 8).astype("float32")
    src = [i for i in range(n) for j in range(n) if i != j]
    dst = [j for i in range(n) for j in range(n) if i != j]
    edge_index = np.array([src, dst], dtype="int64")
    edge_attr = np.abs(np.random.randn(n * (n - 1), 1).astype("float32")) * 100
    return x, edge_index, edge_attr


@pytest.fixture(params=["gat", "sage", "gcn"])
def backbone(request):
    return BACKBONES[request.param]()


def test_forward_shape_n5(backbone):
    x, ei, ea = _make_graph(5)
    out = backbone(x, ei, ea)
    assert out.shape == (5, 64), f"Expected (5,64), got {out.shape}"


def test_forward_dtype(backbone):
    x, ei, ea = _make_graph(5)
    out = backbone(x, ei, ea)
    assert out.dtype == torch.float32


def test_output_dim_property(backbone):
    assert backbone.output_dim == 64


@pytest.mark.parametrize("BackboneCls", [SAGEBackbone, GCNBackbone])
def test_zero_shot_n10(BackboneCls):
    m = BackboneCls()
    x, ei, ea = _make_graph(10)
    out = m(x, ei, ea)
    assert out.shape == (10, 64)


@pytest.mark.parametrize("BackboneCls", [SAGEBackbone, GCNBackbone])
def test_zero_shot_n20(BackboneCls):
    m = BackboneCls()
    x, ei, ea = _make_graph(20)
    out = m(x, ei, ea)
    assert out.shape == (20, 64)


def test_gat_zero_shot_n10():
    m = GATBackbone()
    x, ei, ea = _make_graph(10)
    out = m(x, ei, ea)
    assert out.shape == (10, 64)


def test_no_nan_in_output(backbone):
    x, ei, ea = _make_graph(5)
    out = backbone(x, ei, ea)
    assert not torch.isnan(out).any(), "NaN in backbone output"


def test_backbone_factory():
    for name in ["gat", "sage", "gcn"]:
        b = BACKBONES[name]()
        assert hasattr(b, "output_dim")
        assert hasattr(b, "forward")


def test_eval_mode_deterministic():
    b = GATBackbone(dropout=0.5)
    b.eval()
    x, ei, ea = _make_graph(5)
    out1 = b(x, ei, ea)
    out2 = b(x, ei, ea)
    torch.testing.assert_close(out1, out2)
