from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

from gnn.base_backbone import GNNBackbone


class GATBackbone(GNNBackbone):
    """
    2-layer Graph Attention Network v2.
    Layer 1: GATv2Conv(F, hidden, heads) concat → (N, hidden*heads)
    Layer 2: GATv2Conv(hidden*heads, out_dim, 1) no-concat → (N, out_dim)
    """

    def __init__(
        self,
        in_channels: int = 8,
        hidden: int = 32,
        out_dim: int = 64,
        heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self._out_dim = out_dim
        self.dropout = dropout

        self.conv1 = GATv2Conv(
            in_channels, hidden, heads=heads, edge_dim=1, concat=True, dropout=dropout
        )
        self.conv2 = GATv2Conv(
            hidden * heads, out_dim, heads=1, edge_dim=1, concat=False, dropout=dropout
        )

    @property
    def output_dim(self) -> int:
        return self._out_dim

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        x, edge_index, edge_attr = self.to_tensors(x, edge_index, edge_attr)
        h = F.elu(self.conv1(x, edge_index, edge_attr=edge_attr))
        if self.dropout > 0 and self.training:
            h = F.dropout(h, p=self.dropout)
        h = self.conv2(h, edge_index, edge_attr=edge_attr)
        return h
