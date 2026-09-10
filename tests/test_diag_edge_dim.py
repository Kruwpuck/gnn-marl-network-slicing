"""Both diagnostics must survive an edge_attr with more than one column.

The v6 arms `gatedge` and `gatres-edge` are built with edge_dim=2 (gnn/__init__.py), and two
instruments still assumed the v4 width of 1. One of them crashed and one of them did not,
which is the worse half: scripts/attention_analysis.py flattened the (E,2) array to 2E values
and let zip() truncate it back against E edges, so every edge took a neighbour's path loss and
the correlation still printed as a plausible number. Both faults are pinned here.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.attention_analysis import build_pathloss_lookup
from scripts.diag_gnn_reliance import strip_neighbours


def _graph(n_nodes: int, n_edge_feat: int) -> dict:
    """Complete directed graph without self-loops, as envs/channel_model.py emits it."""
    pairs = [(s, d) for s in range(n_nodes) for d in range(n_nodes) if s != d]
    ei = np.array(pairs, dtype=np.int64).T
    ea = np.arange(len(pairs) * n_edge_feat, dtype=np.float32).reshape(len(pairs), n_edge_feat)
    return {"x": np.zeros((n_nodes, 8), dtype=np.float32), "edge_index": ei, "edge_attr": ea}


@pytest.mark.parametrize("n_edge_feat", [1, 2])
@pytest.mark.parametrize("attention", [True, False])
def test_strip_neighbours_keeps_edge_attr_width(n_edge_feat, attention):
    """D2a: the ablated graph keeps the arm's edge width, or GATv2Conv refuses it."""
    out = strip_neighbours(_graph(5, n_edge_feat), attention=attention)
    assert out["edge_attr"].shape[1] == n_edge_feat
    # self-loops for attention (one per node), no edges at all otherwise
    assert out["edge_attr"].shape[0] == (5 if attention else 0)
    assert out["edge_index"].shape[1] == out["edge_attr"].shape[0]


@pytest.mark.parametrize("n_edge_feat", [1, 2])
def test_pathloss_lookup_has_one_entry_per_edge(n_edge_feat):
    """One entry per edge, and it is column 0 -- not 2E entries, not column 1."""
    g = _graph(5, n_edge_feat)
    lookup = build_pathloss_lookup(g["edge_index"], g["edge_attr"])
    n_edges = g["edge_index"].shape[1]
    assert len(lookup) == n_edges
    for k in range(n_edges):
        s, d = int(g["edge_index"][0, k]), int(g["edge_index"][1, k])
        assert lookup[(s, d)] == pytest.approx(float(g["edge_attr"][k, 0]))


# --- D2b applicability gate -------------------------------------------------
# scripts/diag_gnn_reliance.py used to run D2b only when the backbone was literally named
# "gat". That spelling of "reads edge_attr" was correct in v4 and silently wrong in v6: the
# gatedge / gatres-edge arms -- the only ones carrying a second edge column -- reported N/A
# and the report still exited 0. The gate now reads the capability, so pin the capability.

D2B_EXPECTED = {
    "gat": True, "gatres": True, "gatedge": True, "gatres-edge": True,
    "sage": False, "gcn": False,
}


@pytest.mark.parametrize("name,expected", sorted(D2B_EXPECTED.items()))
def test_d2b_gate_matches_edge_attr_capability(name, expected):
    from gnn import BACKBONES

    backbone = BACKBONES[name](in_channels=8, hidden=16, out_dim=8)
    # the exact expression scripts/diag_gnn_reliance.py gates D2b on
    assert bool(getattr(backbone, "edge_dim", 0)) is expected


def test_d2b_gate_covers_every_registered_backbone():
    """A new backbone must be classified here, not silently default to N/A."""
    from gnn import BACKBONES

    assert set(BACKBONES) == set(D2B_EXPECTED)


# --- zero-shot wave label ----------------------------------------------------
# scripts/zeroshot_eval.py printed "wave v4" as a literal in the report header. Pointed at the
# v6 checkpoints it still said v4, and every number under that header was correct, so nothing
# looked wrong. The label is now read off the checkpoint glob; pin that it tracks the glob.

@pytest.mark.parametrize(
    "glob_pattern,expected",
    [
        ("results/logs/gnn-*_v6_seed*.pt", "v6"),
        ("results/logs/gnn-mappo_*_v6_seed*.pt", "v6"),
        ("results/logs/*_v4_seed*.pt", "v4"),
        ("results/logs/*_v6smoke_seed*.pt", "v6smoke"),
    ],
)
def test_wave_tag_follows_the_checkpoint_glob(glob_pattern, expected):
    from scripts.zeroshot_eval import wave_tag

    assert wave_tag(glob_pattern) == expected


def test_wave_tag_never_invents_a_wave():
    """No tag in the glob means the glob is printed, not a guessed wave name."""
    from scripts.zeroshot_eval import wave_tag

    assert wave_tag("results/logs/some_checkpoint.pt") == "results/logs/some_checkpoint.pt"
