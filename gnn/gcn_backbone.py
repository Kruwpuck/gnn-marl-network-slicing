from __future__ import annotations
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from gnn.base_backbone import GNNBackbone


class GCNBackbone(GNNBackbone):
    """
    2-layer GCN baseline.
    Fixed degree-normalized aggregation; does not use edge_attr.
    """

    def __init__(self, in_channels: int = 8, hidden: int = 64, out_dim: int = 64):
        super().__init__()
        self._out_dim = out_dim
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, out_dim)

    @property
    def output_dim(self) -> int:
        return self._out_dim

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        x, edge_index, _ = self.to_tensors(x, edge_index, edge_attr)
        h = F.relu(self.conv1(x, edge_index))
        return self.conv2(h, edge_index)
