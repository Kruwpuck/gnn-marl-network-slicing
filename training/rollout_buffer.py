from __future__ import annotations
import numpy as np


class RolloutBuffer:
    """
    On-policy rollout storage with Generalized Advantage Estimation (GAE).
    Stores N-agent transitions for PPO updates.
    """

    def __init__(
        self,
        n_steps: int = 2048,
        n_agents: int = 5,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        self.n_steps = n_steps
        self.n_agents = n_agents
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clear()

    def clear(self) -> None:
        self.graph_dicts: list[dict] = []
        self.actions: list[np.ndarray] = []
        self.log_probs: list[np.ndarray] = []
        self.rewards: list[float] = []
        self.values: list[np.ndarray] = []
        self.dones: list[bool] = []
        self._ptr = 0

    def add(
        self,
        graph_dict: dict,
        actions: np.ndarray,
        log_probs: np.ndarray,
        reward: float,
        values: np.ndarray,
        done: bool,
    ) -> None:
        self.graph_dicts.append(graph_dict)
        self.actions.append(np.asarray(actions))
        self.log_probs.append(np.asarray(log_probs))
        self.rewards.append(float(reward))
        self.values.append(np.asarray(values))
        self.dones.append(bool(done))
        self._ptr += 1

    def compute_advantages(self, last_value: np.ndarray | None = None) -> dict:
        """GAE-λ advantage estimation. Returns rollout dict for PPO.learn()."""
        T = len(self.rewards)
        advantages = np.zeros((T, self.n_agents), dtype=np.float32)
        returns = np.zeros((T, self.n_agents), dtype=np.float32)

        values_arr = np.array(self.values, dtype=np.float32)    # (T, N)
        rewards_arr = np.array(self.rewards, dtype=np.float32)  # (T,)
        dones_arr = np.array(self.dones, dtype=np.float32)      # (T,)

        next_value = np.zeros(self.n_agents) if last_value is None else np.asarray(last_value)
        gae = np.zeros(self.n_agents, dtype=np.float32)

        for t in reversed(range(T)):
            nv = next_value if t == T - 1 else values_arr[t + 1]
            delta = rewards_arr[t] + self.gamma * nv * (1.0 - dones_arr[t]) - values_arr[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones_arr[t]) * gae
            advantages[t] = gae
        returns = advantages + values_arr

        return {
            "graph_dicts": self.graph_dicts,
            "actions": np.array(self.actions),      # (T, N)
            "old_log_probs": np.array(self.log_probs),  # (T, N)
            "advantages": advantages,               # (T, N)
            "returns": returns,                     # (T, N)
        }

    def is_full(self) -> bool:
        return self._ptr >= self.n_steps
