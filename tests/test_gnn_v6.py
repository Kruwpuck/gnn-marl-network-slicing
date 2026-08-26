"""Wave-v6 arms: input-reaching residual and the interference-coupling edge column.

Covers docs/revisi/PLAN-03 sections 2 and 5. Two properties carry the most weight here:
the v4 `gat` variant must stay bit-identical (otherwise every v4 comparison is void), and
the new edge column must actually reach the output (D2b found v4's edge attribute unused in
0/25 checkpoints, and a new arm must not repeat that silently).
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.mlp_agent import knn_features                      # noqa: E402
from envs.network_slicing_env import NetworkSlicingEnv         # noqa: E402
from gnn import BACKBONES, GATBackbone                         # noqa: E402
from scripts.diag_input_separability import rel_spread         # noqa: E402
from scripts.rliable_report import (                          # noqa: E402
    MATCHED_BASELINES, parse_run_name, per_seed_means)

N = 5


def _graph(n: int = N, seed: int = 0, cols: int = 2):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 8)).astype("float32")
    src = [i for i in range(n) for j in range(n) if i != j]
    dst = [j for i in range(n) for j in range(n) if i != j]
    edge_index = np.array([src, dst], dtype="int64")
    edge_attr = (np.abs(rng.standard_normal((len(src), cols))) * 100).astype("float32")
    return x, edge_index, edge_attr


def _zero_all(module: torch.nn.Module) -> None:
    with torch.no_grad():
        for p in module.parameters():
            p.zero_()


# --- the v4 variant must not move -------------------------------------------------------

def test_gat_reads_only_column_zero_so_v4_is_unchanged():
    """A two-column env feeds the v4 backbone exactly the v4 feature."""
    torch.manual_seed(0)
    m = GATBackbone()
    x, ei, ea2 = _graph(cols=2)
    out_two = m(x, ei, ea2)
    out_one = m(x, ei, ea2[:, :1].copy())
    assert torch.equal(out_two, out_one)


def test_gat_state_dict_has_no_residual_keys():
    keys = set(GATBackbone().state_dict())
    res_keys = set(BACKBONES["gatres"]().state_dict())
    assert not any(k.startswith("proj") for k in keys)
    assert {"proj1.weight", "proj2.weight"} <= res_keys
    assert keys < res_keys, "the residual arm must add keys, never rename or drop them"


def test_knn_ranking_is_unchanged_by_the_second_column():
    """mlp-knn-ppo ranks by path loss; a coupling column must not disturb it."""
    x, ei, ea2 = _graph(cols=2)
    g2 = {"x": x, "edge_index": ei, "edge_attr": ea2}
    g1 = {"x": x, "edge_index": ei, "edge_attr": ea2[:, :1].copy()}
    assert np.array_equal(knn_features(g2, k=4), knn_features(g1, k=4))


# --- section 2: the new edge column must reach the output -------------------------------

def test_edge_attr_changes_the_output_of_the_edge_arm():
    """PLAN-03 section 2's mandatory test, on the arm that has two columns."""
    torch.manual_seed(0)
    m = BACKBONES["gatedge"]()
    x, ei, ea = _graph(cols=2)
    out_real = m(x, ei, ea)
    out_zero = m(x, ei, np.zeros_like(ea))
    assert not torch.allclose(out_real, out_zero), "edge_attr has no effect -- bug"


def test_coupling_column_is_not_a_copy_of_the_path_loss_column():
    env = NetworkSlicingEnv()
    env.reset(seed=7)
    ea = env._get_graph_dict()["edge_attr"]
    assert ea.shape[1] == 2
    pl, coup = ea[:, 0], ea[:, 1]
    assert np.corrcoef(pl, coup)[0, 1] < 0.999, (
        "the coupling column is a reparametrisation of path loss -- it adds nothing, and "
        "should be dropped the way distance_norm was (PLAN-03 P3)"
    )


def test_coupling_is_directional():
    """coupling[i->j] describes i's power at j's users; the reverse edge is a different
    quantity, and a symmetric column would mean the per-UE geometry was averaged away."""
    env = NetworkSlicingEnv()
    env.reset(seed=7)
    g = env._get_graph_dict()
    ei, coup = g["edge_index"], g["edge_attr"][:, 1]
    lookup = {(int(s), int(d)): c for s, d, c in zip(ei[0], ei[1], coup)}
    pairs = [(s, d) for (s, d) in lookup if (d, s) in lookup]
    assert pairs, "fully connected graph should have reverse edges"
    assert any(abs(lookup[(s, d)] - lookup[(d, s)]) > 1e-6 for s, d in pairs)


def test_column_zero_still_is_the_channel_models_path_loss():
    env = NetworkSlicingEnv()
    env.reset(seed=7)
    _, pl_only = env.channel_model.build_interference_graph(env._gnb_positions)
    assert np.array_equal(env._get_graph_dict()["edge_attr"][:, :1], pl_only)


# --- section 5: the residual has to reach the input -------------------------------------

def test_residual_reaches_the_input_not_the_previous_layer():
    """With both convolutions zeroed, the graph path carries nothing. `gat` then emits
    identical nodes; `gatres` still separates them, because its skip starts at x."""
    torch.manual_seed(0)
    plain, res = GATBackbone(), BACKBONES["gatres"]()
    for m in (plain, res):
        _zero_all(m.conv1)
        _zero_all(m.conv2)
    x, ei, ea = _graph(cols=2)
    out_plain = plain(x, ei, ea).detach().numpy()
    out_res = res(x, ei, ea).detach().numpy()
    assert rel_spread(out_plain) == pytest.approx(0.0, abs=1e-6)
    assert rel_spread(out_res) > 1e-3


def test_residual_arm_separates_nodes_more_than_the_v4_arm():
    """Same input, same seed, untrained: the arm that keeps a path from x must retain more
    node separation than the one that does not. This is the quantity D6 reports."""
    spreads = {}
    for key in ("gat", "gatres"):
        torch.manual_seed(0)
        m = BACKBONES[key]()
        x, ei, ea = _graph(cols=2)
        spreads[key] = rel_spread(m(x, ei, ea).detach().numpy())
    assert spreads["gatres"] > spreads["gat"]


@pytest.mark.parametrize("key", ["gat", "gatres", "gatedge", "gatres-edge"])
def test_every_arm_runs_at_a_different_topology(key):
    """Zero-shot transfer is the claim these arms must not break."""
    m = BACKBONES[key]()
    for n in (5, 10):
        x, ei, ea = _graph(n=n, cols=2)
        out = m(x, ei, ea)
        assert out.shape == (n, 64)
        assert not torch.isnan(out).any()


# --- the run-name parser must tell the arms apart ---------------------------------------

@pytest.mark.parametrize("backbone", ["gat", "gatres", "gatedge", "gatres-edge"])
def test_parse_run_name_keeps_each_arm_separate(backbone):
    algo, tag, seed = parse_run_name(f"gnn-mappo_{backbone}_floornone_seed42")
    assert (algo, tag, seed) == (f"gnn-mappo_{backbone}", "_floornone", 42)


@pytest.mark.parametrize("family,arm", [(f, a) for f in ("gnn-mappo", "gnn-madqn")
                                        for a in ("gatres", "gatedge", "gatres-edge")])
def test_every_v6_arm_has_a_report_section(family, arm):
    """Found by the v6 smoke run: an unregistered arm produces no section and no error, so
    the report exits 0 while silently omitting the whole wave."""
    key = f"{family}_{arm}"
    assert key in MATCHED_BASELINES, f"{key} would be missing from the rliable report"
    _, baselines = MATCHED_BASELINES[key]
    assert f"{family}_gat" in baselines, "PREREG-V6 names gat as THE comparator for the arms"


def test_one_algorithm_may_not_pool_two_waves():
    """The v6 report reads two tags at once (_v6 arms against the _v4 comparator). The hazard
    that opens is one algorithm silently averaging runs from both waves, so it aborts."""
    import pandas as pd
    df = pd.DataFrame({
        "algo_key": ["gnn-mappo_gat"] * 4,
        "tag": ["_v4", "_v4", "_v6", "_v6"],
        "seed_parsed": [42, 43, 42, 43],
        "embb_p5_mbps": [1.0, 2.0, 3.0, 4.0],
    })
    with pytest.raises(SystemExit, match="more than one tag"):
        per_seed_means(df, "gnn-mappo_gat", {"_v4", "_v6"}, "embb_p5_mbps")
    one = per_seed_means(df, "gnn-mappo_gat", {"_v4"}, "embb_p5_mbps")
    assert one is not None and one.shape == (2, 1)
