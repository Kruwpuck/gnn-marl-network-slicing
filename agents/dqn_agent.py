from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gnn.base_backbone import GNNBackbone
from .hparams import resolve


class DQNAgent(nn.Module):
    """
    Dueling DQN with parameter sharing across gNB agents.
    One forward pass processes all N nodes simultaneously.
    """

    def __init__(self, backbone: GNNBackbone, config_path: str | None = None, **overrides):
        """Hyperparameters come from `agent.dqn` in configs/experiment_config.yaml
        (goal1.md C2), not from defaults here. **overrides is for tests and sweeps and
        raises on an unknown key rather than silently ignoring it."""
        super().__init__()
        hp = resolve("dqn", config_path, overrides)
        self.backbone = backbone
        self.n_actions = hp["n_actions"]
        self.epsilon = hp["epsilon"]
        self.epsilon_min = hp["epsilon_min"]
        self.epsilon_decay = hp["epsilon_decay"]
        self.gamma = hp["gamma"]
        self.max_grad_norm = hp["max_grad_norm"]

        D = backbone.output_dim
        hidden = hp["hidden"]
        n_actions = hp["n_actions"]

        # Dueling streams
        self.value_stream = nn.Sequential(
            nn.Linear(D, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(D, hidden), nn.ReLU(), nn.Linear(hidden, n_actions)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=hp["lr"])

    def q_values(self, graph_dict: dict) -> torch.Tensor:
        """Return Q-values tensor shape (N, n_actions)."""
        x, ei, ea = graph_dict["x"], graph_dict["edge_index"], graph_dict["edge_attr"]
        emb = self.backbone(x, ei, ea)      # (N, D)
        V = self.value_stream(emb)           # (N, 1)
        A = self.adv_stream(emb)             # (N, n_actions)
        return V + (A - A.mean(dim=-1, keepdim=True))

    def forward(self, graph_dict: dict) -> torch.Tensor:
        return self.q_values(graph_dict)

    def act(self, graph_dict: dict, greedy: bool = False) -> np.ndarray:
        """ε-greedy action selection. Returns (N,) int actions."""
        if not greedy and np.random.random() < self.epsilon:
            n = graph_dict["x"].shape[0] if isinstance(graph_dict["x"], np.ndarray) \
                else graph_dict["x"].shape[0]
            return np.random.randint(0, self.n_actions, size=n)
        with torch.no_grad():
            q = self.q_values(graph_dict)   # (N, n_actions)
            return q.argmax(dim=-1).cpu().numpy()

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def learn(self, batch: list) -> float:
        """
        batch: list of (graph_dict, actions, reward, next_graph_dict, done)
        Returns MSE/Huber loss value.
        """
        losses = []
        for graph_dict, actions, reward, next_graph_dict, done in batch:
            with torch.no_grad():
                q_next = self.q_values(next_graph_dict).max(dim=-1).values  # (N,)
                target = reward + self.gamma * q_next * (1.0 - float(done))
                target = target.detach()

            q_pred = self.q_values(graph_dict)                               # (N, n_actions)
            actions_t = torch.as_tensor(actions, dtype=torch.long).to(q_pred.device)
            q_taken = q_pred.gather(1, actions_t.unsqueeze(1)).squeeze(1)    # (N,)
            loss = F.huber_loss(q_taken, target)
            losses.append(loss)

        total_loss = torch.stack(losses).mean()
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.decay_epsilon()
        return float(total_loss)
