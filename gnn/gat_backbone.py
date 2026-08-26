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

    Defaults reproduce the v4 variant exactly. `edge_dim` and `residual` select the v6 arms
    (docs/revisi/PLAN-03, PREREG-V6); see gnn/__init__.py for the registry keys.
    """

    def __init__(
        self,
        in_channels: int = 8,
        hidden: int = 32,
        out_dim: int = 64,
        heads: int = 4,
        dropout: float = 0.0,
        edge_dim: int = 1,
        residual: bool = False,
        alpha: float = 0.1,
    ):
        super().__init__()
        self._out_dim = out_dim
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.residual = residual
        self.alpha = alpha

        self.conv1 = GATv2Conv(
            in_channels, hidden, heads=heads, edge_dim=edge_dim, concat=True, dropout=dropout
        )
        self.conv2 = GATv2Conv(
            hidden * heads, out_dim, heads=1, edge_dim=edge_dim, concat=False, dropout=dropout
        )
        # Residual reaches the INPUT, not the previous layer (PLAN-03 section 5). D6 measured
        # conv1 discarding 98.6% of the node separation, so a connection that starts at h1 has
        # nothing left to rescue: what needs saving is x, and x is already gone. Projections
        # because the dimensions genuinely differ (8 -> hidden*heads -> out_dim). Created only
        # when residual=True, so the v4 variant's state_dict keys are untouched.
        if residual:
            self.proj1 = nn.Linear(in_channels, hidden * heads, bias=False)
            self.proj2 = nn.Linear(in_channels, out_dim, bias=False)

    @property
    def output_dim(self) -> int:
        return self._out_dim

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        x, edge_index, edge_attr = self.to_tensors(x, edge_index, edge_attr)
        # The env emits every edge column it has; a variant reads the first edge_dim of them.
        # edge_dim=1 therefore sees exactly the v4 feature (gNB-to-gNB path loss) no matter how
        # many columns the env grows later.
        edge_attr = edge_attr[:, :self.edge_dim]
        if not return_attention:
            h = F.elu(self.conv1(x, edge_index, edge_attr=edge_attr))
            if self.residual:
                h = h + self.alpha * self.proj1(x)
            if self.dropout > 0 and self.training:
                h = F.dropout(h, p=self.dropout)
            h = self.conv2(h, edge_index, edge_attr=edge_attr)
            if self.residual:
                h = h + self.alpha * self.proj2(x)
            return h

        h, attn1 = self.conv1(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
        h = F.elu(h)
        if self.residual:
            h = h + self.alpha * self.proj1(x)
        if self.dropout > 0 and self.training:
            h = F.dropout(h, p=self.dropout)
        h, attn2 = self.conv2(h, edge_index, edge_attr=edge_attr, return_attention_weights=True)
        if self.residual:
            h = h + self.alpha * self.proj2(x)
        return h, [attn1, attn2]
