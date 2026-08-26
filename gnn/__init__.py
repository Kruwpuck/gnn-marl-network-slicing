from functools import partial

from gnn.gat_backbone import GATBackbone
from gnn.sage_backbone import SAGEBackbone
from gnn.gcn_backbone import GCNBackbone

# The three v6 arms (docs/revisi/PLAN-03, PREREG-V6). New keys, never overwriting `gat`:
# the v4 variant keeps its name so its checkpoints, CSVs and reports stay referable
# (PLAN-03 Larangan 5). `sage` is deliberately left alone -- it is the natural control for
# D6's mechanistic prediction, and SAGEConv takes no edge_attr at all.
BACKBONES: dict[str, type] = {
    "gat": GATBackbone,
    "gatres": partial(GATBackbone, residual=True),
    "gatedge": partial(GATBackbone, edge_dim=2),
    "gatres-edge": partial(GATBackbone, edge_dim=2, residual=True),
    "sage": SAGEBackbone,
    "gcn": GCNBackbone,
}
