from gnn.gat_backbone import GATBackbone
from gnn.sage_backbone import SAGEBackbone
from gnn.gcn_backbone import GCNBackbone

BACKBONES: dict[str, type] = {
    "gat": GATBackbone,
    "sage": SAGEBackbone,
    "gcn": GCNBackbone,
}
