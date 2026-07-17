from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from gnn.base_backbone import GNNBackbone


class PPOAgent(nn.Module):
    """
    PPO actor-critic with shared GNN backbone and parameter sharing across agents.
    """

    def __init__(
        self,
        backbone: GNNBackbone,
        n_actions: int = 11,
        hidden: int = 128,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
    ):
        super().__init__()
        self.backbone = backbone
        self.n_actions = n_actions
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        D = backbone.output_dim
        self.actor = nn.Sequential(
            nn.Linear(D, hidden), nn.Tanh(), nn.Linear(hidden, n_actions)
        )
        self.critic = nn.Sequential(
            nn.Linear(D, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def _embed(self, graph_dict: dict) -> torch.Tensor:
        x, ei, ea = graph_dict["x"], graph_dict["edge_index"], graph_dict["edge_attr"]
        return self.backbone(x, ei, ea)

    def forward(self, graph_dict: dict) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self._embed(graph_dict)
        return self.actor(emb), self.critic(emb).squeeze(-1)

    def act(
        self, graph_dict: dict, greedy: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (actions, log_probs, values) each shape (N,)."""
        with torch.no_grad():
            emb = self._embed(graph_dict)
            logits = self.actor(emb)
            values = self.critic(emb).squeeze(-1)
            dist = Categorical(logits=logits)
            actions = logits.argmax(dim=-1) if greedy else dist.sample()
            log_probs = dist.log_prob(actions)
        return (
            actions.cpu().numpy(),
            log_probs.cpu().numpy(),
            values.cpu().numpy(),
        )

    def evaluate(
        self, graph_dict: dict, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (log_probs, values, entropy) for PPO update."""
        emb = self._embed(graph_dict)
        logits = self.actor(emb)
        values = self.critic(emb).squeeze(-1)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), values, dist.entropy()

    def learn(self, rollout: dict) -> dict:
        """
        rollout keys: graph_dicts, actions, old_log_probs, advantages, returns
        Returns loss dict.
        """
        graph_dicts = rollout["graph_dicts"]
        dev = next(self.parameters()).device
        actions = torch.as_tensor(rollout["actions"], dtype=torch.long).to(dev)
        old_log_probs = torch.as_tensor(rollout["old_log_probs"], dtype=torch.float32).to(dev)
        advantages = torch.as_tensor(rollout["advantages"], dtype=torch.float32).to(dev)
        returns = torch.as_tensor(rollout["returns"], dtype=torch.float32).to(dev)

        # Normalise advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Process all steps
        log_probs_list, values_list, entropy_list = [], [], []
        for t, gd in enumerate(graph_dicts):
            lp, v, ent = self.evaluate(gd, actions[t])
            log_probs_list.append(lp)
            values_list.append(v)
            entropy_list.append(ent)

        log_probs = torch.stack(log_probs_list)    # (T, N)
        values = torch.stack(values_list)           # (T, N)
        entropy = torch.stack(entropy_list).mean()  # scalar

        ratio = (log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(values, returns)
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            "loss": float(loss),
            "policy_loss": float(policy_loss),
            "value_loss": float(value_loss),
            "entropy": float(entropy),
        }
