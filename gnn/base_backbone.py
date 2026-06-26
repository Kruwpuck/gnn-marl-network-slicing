from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn


class GNNBackbone(ABC, nn.Module):
    """Common interface for all GNN backbones."""

    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def output_dim(self) -> int: ...

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:          (N, F)   node feature matrix
            edge_index: (2, E)   directed edges
            edge_attr:  (E, 1)   edge weights (path-loss / 100.0)
        Returns:
            node_embeddings: (N, output_dim)
        """
        ...

    @staticmethod
    def to_tensors(
        x: np.ndarray | torch.Tensor,
        edge_index: np.ndarray | torch.Tensor,
        edge_attr: np.ndarray | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert numpy arrays from env graph dict to float tensors."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if isinstance(edge_index, np.ndarray):
            edge_index = torch.from_numpy(edge_index).long()
        if isinstance(edge_attr, np.ndarray):
            edge_attr = torch.from_numpy(edge_attr).float() / 100.0
        return x, edge_index, edge_attr
