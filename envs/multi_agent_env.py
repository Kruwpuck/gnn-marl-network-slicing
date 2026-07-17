from __future__ import annotations

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from envs.network_slicing_env import NetworkSlicingEnv


class NetworkSlicingMAEnv(ParallelEnv):
    """
    PettingZoo ParallelEnv wrapper around NetworkSlicingEnv.
    Each agent = one gNB. Agents act simultaneously each step.
    """

    metadata = {"render_modes": [], "name": "network_slicing_v0"}

    def __init__(self, config_path: str | None = None, action_type: str = "discrete"):
        self._inner = NetworkSlicingEnv(config_path=config_path, action_type=action_type)
        self.possible_agents = [f"gnb_{i}" for i in range(self._inner.n_gnb)]
        self.agents = list(self.possible_agents)
        self.action_type = action_type
        self._n_gnb = self._inner.n_gnb

        obs_shape = (8,)
        if action_type == "discrete":
            act_space = spaces.Discrete(self._inner.n_tiers)
        else:
            act_space = spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32)

        self.observation_spaces = {a: spaces.Box(-np.inf, np.inf, obs_shape, np.float32)
                                   for a in self.possible_agents}
        self.action_spaces = {a: act_space for a in self.possible_agents}

    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]

    def reset(self, seed: int | None = None, options: dict | None = None):
        self.agents = list(self.possible_agents)
        obs_all, info = self._inner.reset(seed=seed, options=options)
        graph = info.get("graph", {})
        obs = {a: obs_all[i] for i, a in enumerate(self.agents)}
        infos = {a: {} for a in self.agents}
        infos["__common__"] = {"graph": graph}
        return obs, infos

    def step(self, actions: dict):
        action_arr = np.array([actions[a] for a in self.agents])
        obs_all, reward, terminated, truncated, info = self._inner.step(action_arr)

        obs = {a: obs_all[i] for i, a in enumerate(self.agents)}
        rewards = {a: reward / self._n_gnb for a in self.agents}
        terminateds = {a: terminated for a in self.agents}
        truncateds = {a: truncated for a in self.agents}
        infos = {a: {} for a in self.agents}
        infos["__common__"] = {"graph": info.get("graph", {}),
                                "embb_rates": info.get("embb_rates"),
                                "urllc_violation_rate": info.get("urllc_violation_rate"),
                                "urllc_delivered_delays": info.get("urllc_delivered_delays")}

        if truncated or terminated:
            self.agents = []

        return obs, rewards, terminateds, truncateds, infos
