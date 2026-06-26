from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from gnn.base_backbone import GNNBackbone


class SAGEBackbone(GNNBackbone):
    """
    2-layer GraphSAGE.  Inductive — handles arbitrary N at inference time.
    edge_attr not used (SAGE aggregation is degree-agnostic).
    """

    def __init__(self, in_channels: int = 8, hidden: int = 64, out_dim: int = 64):
        super().__init__()
        self._out_dim = out_dim
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, out_dim)

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
