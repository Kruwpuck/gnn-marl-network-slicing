from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class MLPDQNAgent(nn.Module):
    """
    B1/B2 baseline: Dueling DQN with flat MLP (no GNN).
    B1 centralized: obs_dim = n_gnb * 8  (global state)
    B2 independent: obs_dim = 8          (local obs per agent)
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int = 11,
        hidden: int = 128,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.9995,
        lr: float = 1e-3,
        gamma: float = 0.99,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma

        self.shared = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU())
        self.value_stream = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.adv_stream = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_actions))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def q_values(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (batch, obs_dim) or (obs_dim,) → Q: (batch, n_actions)"""
        h = self.shared(obs)
        V = self.value_stream(h)
        A = self.adv_stream(h)
        return V + (A - A.mean(dim=-1, keepdim=True))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.q_values(obs)

    @property
    def _device(self):
        return next(self.parameters()).device

    def act(self, obs: np.ndarray, greedy: bool = False) -> np.ndarray | int:
        if not greedy and np.random.random() < self.epsilon:
            batch = obs.shape[0] if obs.ndim > 1 else 1
            return np.random.randint(0, self.n_actions, size=batch) if batch > 1 \
                else int(np.random.randint(0, self.n_actions))
        obs_t = torch.as_tensor(obs, dtype=torch.float32).to(self._device)
        with torch.no_grad():
            q = self.q_values(obs_t)
        return q.argmax(dim=-1).cpu().numpy()

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def learn(self, batch: list) -> float:
        dev = self._device
        losses = []
        for obs, actions, reward, next_obs, done in batch:
            obs_t = torch.as_tensor(
                np.asarray(obs, dtype=np.float32).reshape(-1, self.obs_dim)).to(dev)
            next_t = torch.as_tensor(
                np.asarray(next_obs, dtype=np.float32).reshape(-1, self.obs_dim)).to(dev)
            actions_t = torch.as_tensor(
                np.asarray(actions, dtype=np.int64).flatten()).to(dev)  # (B,)
            B = obs_t.shape[0]
            with torch.no_grad():
                q_next = self.q_values(next_t).max(dim=-1).values  # (B,)
                rew_t = torch.full((B,), float(reward), device=dev)
                target = rew_t + self.gamma * q_next * (1.0 - float(done))
            q_pred = self.q_values(obs_t)                            # (B, n_actions)
            q_taken = q_pred.gather(1, actions_t.unsqueeze(1)).squeeze(1)  # (B,)
            losses.append(F.huber_loss(q_taken, target.detach()))
        loss = torch.stack(losses).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 10.0)
        self.optimizer.step()
        self.decay_epsilon()
        return float(loss)


class MLPPPOAgent(nn.Module):
    """B1/B2 PPO baseline with flat MLP."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int = 11,
        hidden: int = 128,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, n_actions)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    @property
    def _device(self):
        return next(self.parameters()).device

    def act(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32).to(self._device)
        with torch.no_grad():
            logits = self.actor(obs_t)
            values = self.critic(obs_t).squeeze(-1)
            dist = Categorical(logits=logits)
            actions = dist.sample()
        return actions.cpu().numpy(), dist.log_prob(actions).cpu().numpy(), values.cpu().numpy()

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor(obs), self.critic(obs).squeeze(-1)
