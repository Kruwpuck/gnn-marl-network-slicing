from __future__ import annotations
import random
from collections import deque
from typing import Any


class ReplayBuffer:
    """Circular replay buffer for DQN experience replay."""

    def __init__(self, capacity: int = 50_000):
        self._buf: deque = deque(maxlen=capacity)

    def push(
        self,
        graph_dict: dict,
        actions,
        reward: float,
        next_graph_dict: dict,
        done: bool,
    ) -> None:
        self._buf.append((graph_dict, actions, reward, next_graph_dict, done))

    def sample(self, batch_size: int = 64) -> list:
        return random.sample(self._buf, min(batch_size, len(self._buf)))

    def __len__(self) -> int:
        return len(self._buf)
