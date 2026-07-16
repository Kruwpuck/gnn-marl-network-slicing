from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import yaml
from pathlib import Path

from envs.channel_model import ChannelModel
from traffic.poisson_traffic import PoissonTraffic
from traffic.mmpp_traffic import MMPPTraffic


class NetworkSlicingEnv(gym.Env):
    """
    Centralized Gymnasium env for 5G/6G network slicing research.

    Observation: ndarray (n_gnb, 8) — one row per gNB:
        [channel_gain_db, sinr_embb_norm, sinr_urllc_norm,
         queue_embb, queue_urllc, last_delay_norm,
         neighbor_urllc_frac_mean, prev_alloc]

    Action (MultiDiscrete): tier index per gNB, 0..10 = 0%..100% URLLC PRB (step 10%).

    Co-slice interference (env.slice_coupled_interference=True, reuse-1): interference
    a gNB receives on a given slice's subband scales with how much of that subband its
    neighbors are also using for the same slice — allocation decisions couple across
    gNBs, so coordination (not just per-gNB optimization) matters.
    """

    metadata = {"render_modes": []}

    def __init__(self, config_path: str | None = None, action_type: str = "discrete"):
        super().__init__()
        cfg = self._load_config(config_path)

        self.n_gnb: int = cfg["env"]["n_gnb"]
        self.n_prb: int = cfg["env"]["n_prb"]
        self.bandwidth_hz: float = float(cfg["env"]["bandwidth_hz"])
        self.carrier_freq_ghz: float = float(cfg["env"]["carrier_freq_ghz"])
        self.scenario: str = cfg["env"]["scenario"]
        self.h_bs: float = float(cfg["env"]["h_bs"])
        self.h_ut: float = float(cfg["env"]["h_ut"])
        self.area_size: float = float(cfg["env"]["area_size"])
        self.n_ue_per_gnb: int = cfg["env"]["n_ue_per_gnb"]
        self.episode_length: int = cfg["training"]["episode_length"]
        self.slice_coupled_interference: bool = bool(
            cfg["env"].get("slice_coupled_interference", True)
        )

        self.w1 = float(cfg["reward"]["w1"])
        self.w2 = float(cfg["reward"]["w2"])
        self.w3 = float(cfg["reward"]["w3"])
        self.w4 = float(cfg["reward"]["w4"])

        self.urllc_max_delay_ms: float = float(cfg["slices"]["urllc"]["max_delay_ms"])
        self.embb_min_throughput: float = float(cfg["slices"]["embb"]["min_throughput_mbps"]) * 1e6

        self.tx_power_dbm: float = float(cfg["noise"]["tx_power_dbm"])
        self.noise_figure_db: float = float(cfg["noise"]["noise_figure_db"])
        self.thermal_dbm_per_hz: float = float(cfg["noise"]["thermal_dbm_per_hz"])

        self.urllc_lambda: float = float(cfg["traffic"]["urllc"]["lambda_arrival"])
        self.urllc_pkt_bits: int = int(cfg["traffic"]["urllc"]["packet_size_bits"])
        self.embb_lambda_burst: float = float(cfg["traffic"]["embb"]["lambda_burst"]) * 1e6
        self.embb_lambda_idle: float = float(cfg["traffic"]["embb"]["lambda_idle"]) * 1e6
        self.embb_r_b2i: float = float(cfg["traffic"]["embb"]["r_burst_to_idle"])
        self.embb_r_i2b: float = float(cfg["traffic"]["embb"]["r_idle_to_burst"])

        self.action_type = action_type
        self.n_tiers = 11

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_gnb, 8), dtype=np.float32
        )
        if action_type == "discrete":
            self.action_space = spaces.MultiDiscrete([self.n_tiers] * self.n_gnb)
        else:
            self.action_space = spaces.Box(0.0, 1.0, shape=(self.n_gnb, 2), dtype=np.float32)

        self.channel_model = ChannelModel(
            scenario=self.scenario, h_bs=self.h_bs, h_ut=self.h_ut,
            carrier_freq_ghz=self.carrier_freq_ghz, noise_figure_db=self.noise_figure_db,
            thermal_dbm_per_hz=self.thermal_dbm_per_hz, tx_power_dbm=self.tx_power_dbm,
        )

        self._rng: np.random.Generator | None = None
        self._gnb_positions: np.ndarray | None = None
        self._ue_positions: np.ndarray | None = None
        self._pl_matrix: np.ndarray | None = None
        self._edge_index: np.ndarray | None = None
        self._edge_attr: np.ndarray | None = None
        self._queue_embb: np.ndarray | None = None
        self._queue_urllc: np.ndarray | None = None
        self._prev_alloc: np.ndarray | None = None
        self._last_sinr_embb: np.ndarray | None = None
        self._last_sinr_urllc: np.ndarray | None = None
        self._last_delay_ms: np.ndarray | None = None
        self._urllc_traffic: list = []
        self._embb_traffic: list = []
        self._step_count: int = 0

    def _load_config(self, config_path: str | None) -> dict:
        if config_path is None:
            config_path = str(Path(__file__).parent.parent / "configs" / "experiment_config.yaml")
        with open(config_path) as f:
            return yaml.safe_load(f)

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)

        self._gnb_positions = self._rng.uniform(0, self.area_size, (self.n_gnb, 2))
        n_ue_total = self.n_gnb * self.n_ue_per_gnb
        self._ue_positions = self._rng.uniform(0, self.area_size, (n_ue_total, 2))

        self._pl_matrix = self.channel_model.build_channel_matrix(
            self._gnb_positions, self._ue_positions, self._rng
        )
        self._edge_index, self._edge_attr = self.channel_model.build_interference_graph(
            self._gnb_positions
        )

        self._queue_embb = np.zeros(self.n_gnb, dtype=np.float32)
        self._queue_urllc = np.zeros(self.n_gnb, dtype=np.float32)
        self._prev_alloc = np.zeros(self.n_gnb, dtype=np.float32)
        self._last_sinr_embb = np.zeros(self.n_gnb, dtype=np.float32)
        self._last_sinr_urllc = np.zeros(self.n_gnb, dtype=np.float32)
        self._last_delay_ms = np.zeros(self.n_gnb, dtype=np.float32)
        self._step_count = 0

        self._urllc_traffic = [
            PoissonTraffic(self.urllc_lambda, self.urllc_pkt_bits,
                           np.random.default_rng(None if seed is None else seed + i + 1))
            for i in range(self.n_gnb)
        ]
        self._embb_traffic = [
            MMPPTraffic(self.embb_lambda_burst, self.embb_lambda_idle,
                        self.embb_r_b2i, self.embb_r_i2b,
                        np.random.default_rng(None if seed is None else seed + self.n_gnb + i + 1))
            for i in range(self.n_gnb)
        ]

        return self._get_obs(), {"graph": self._get_graph_dict()}

    def step(self, actions):
        assert self._rng is not None, "Call reset() before step()"
        actions = np.asarray(actions)

        urllc_arrivals = np.array([self._urllc_traffic[i].step()[1] for i in range(self.n_gnb)], dtype=np.float32)
        embb_arrivals = np.array([self._embb_traffic[i].step() for i in range(self.n_gnb)], dtype=np.float32)
        self._queue_urllc += urllc_arrivals
        self._queue_embb += embb_arrivals

        if self.action_type == "discrete":
            urllc_fracs = actions / (self.n_tiers - 1)
        else:
            acts = np.clip(actions, 0.0, 1.0)
            acts = acts / acts.sum(axis=-1, keepdims=True).clip(min=1e-8)
            urllc_fracs = acts[:, 1]

        embb_fracs = 1.0 - urllc_fracs
        bw_embb = self.bandwidth_hz * embb_fracs
        bw_urllc = self.bandwidth_hz * urllc_fracs

        # Mean rx power per gNB->own-UEs link (dBm), used for both signal and interference.
        rx_dbm = np.array([
            self.channel_model.compute_rx_power_dbm(
                self._pl_matrix[i, i * self.n_ue_per_gnb:(i + 1) * self.n_ue_per_gnb].mean()
            )
            for i in range(self.n_gnb)
        ])
        # Interferer->my-UEs rx power (dBm), i.e. how strong gNB j's signal lands on my UEs.
        interferer_rx_dbm = np.array([
            [
                self.channel_model.compute_rx_power_dbm(
                    self._pl_matrix[j, i * self.n_ue_per_gnb:(i + 1) * self.n_ue_per_gnb].mean()
                )
                if j != i else -np.inf
                for j in range(self.n_gnb)
            ]
            for i in range(self.n_gnb)
        ])
        interferer_mw = 10.0 ** (interferer_rx_dbm / 10.0)
        interferer_mw = np.where(np.isfinite(interferer_rx_dbm), interferer_mw, 0.0)

        embb_rates = np.zeros(self.n_gnb)
        urllc_rates = np.zeros(self.n_gnb)
        sinr_embb = np.zeros(self.n_gnb)
        sinr_urllc = np.zeros(self.n_gnb)

        for i in range(self.n_gnb):
            if self.slice_coupled_interference:
                # Reuse-1: neighbor only interferes on a slice's subband to the extent
                # it also allocated that subband — this is what makes allocation a
                # coupled multi-agent decision instead of n independent bandits.
                interf_mw_embb = float(np.sum(interferer_mw[i] * embb_fracs))
                interf_mw_urllc = float(np.sum(interferer_mw[i] * urllc_fracs))
            else:
                interf_mw_embb = float(np.sum(interferer_mw[i]))
                interf_mw_urllc = float(np.sum(interferer_mw[i]))

            interf_dbm_embb = 10.0 * np.log10(interf_mw_embb + 1e-30) if interf_mw_embb > 0 else -200.0
            interf_dbm_urllc = 10.0 * np.log10(interf_mw_urllc + 1e-30) if interf_mw_urllc > 0 else -200.0

            sinr_embb[i] = self.channel_model.compute_sinr(rx_dbm[i], interf_dbm_embb, max(bw_embb[i], 1.0))
            sinr_urllc[i] = self.channel_model.compute_sinr(rx_dbm[i], interf_dbm_urllc, max(bw_urllc[i], 1.0))
            embb_rates[i] = self.channel_model.compute_rate(sinr_embb[i], max(bw_embb[i], 1.0))
            urllc_rates[i] = self.channel_model.compute_rate(sinr_urllc[i], max(bw_urllc[i], 1.0))

        self._queue_embb = np.maximum(self._queue_embb - np.minimum(self._queue_embb, embb_rates), 0.0).astype(np.float32)
        self._queue_urllc = np.maximum(self._queue_urllc - np.minimum(self._queue_urllc, urllc_rates), 0.0).astype(np.float32)

        urllc_delay_ms = np.where(
            urllc_rates > 0,
            self._queue_urllc / np.maximum(urllc_rates, 1.0) * 1000.0,
            self.urllc_max_delay_ms * 2.0,
        )

        reward = self._compute_reward(embb_rates, urllc_delay_ms)

        self._prev_alloc = urllc_fracs.astype(np.float32)
        self._last_sinr_embb = sinr_embb.astype(np.float32)
        self._last_sinr_urllc = sinr_urllc.astype(np.float32)
        self._last_delay_ms = urllc_delay_ms.astype(np.float32)
        self._step_count += 1

        obs = self._get_obs()
        info = {"graph": self._get_graph_dict(), "embb_rates": embb_rates,
                "urllc_delay_ms": urllc_delay_ms}
        return obs, reward, False, self._step_count >= self.episode_length, info

    def _compute_reward(self, embb_rates: np.ndarray, urllc_delay_ms: np.ndarray) -> float:
        t_ref = max(self.embb_min_throughput, 1.0)
        delay_ratio = urllc_delay_ms / self.urllc_max_delay_ms
        r = (self.w1 * min(float(embb_rates.mean() / t_ref), 5.0)
             - self.w2 * float(np.mean(urllc_delay_ms > self.urllc_max_delay_ms))
             - self.w3 * float(np.mean(np.log1p(delay_ratio)))
             + self.w4 * float(embb_rates.sum() / self.bandwidth_hz))
        return float(np.clip(r, -10.0, 10.0))

    def _get_obs(self) -> np.ndarray:
        pl_mean = self._pl_matrix.mean(axis=1)
        ch_gain = -pl_mean / 100.0
        sinr_embb_norm = np.clip(10.0 * np.log10(self._last_sinr_embb + 1e-12) / 40.0, -3.0, 3.0)
        sinr_urllc_norm = np.clip(10.0 * np.log10(self._last_sinr_urllc + 1e-12) / 40.0, -3.0, 3.0)
        q_e = np.log1p(self._queue_embb) / 20.0
        q_u = np.log1p(self._queue_urllc) / 20.0
        last_delay_norm = np.log1p(self._last_delay_ms / self.urllc_max_delay_ms) / 5.0
        n = self.n_gnb
        if n > 1:
            total = self._prev_alloc.sum()
            neighbor_mean = (total - self._prev_alloc) / (n - 1)
        else:
            neighbor_mean = np.zeros_like(self._prev_alloc)
        return np.stack([ch_gain, sinr_embb_norm, sinr_urllc_norm, q_e, q_u,
                         last_delay_norm, neighbor_mean, self._prev_alloc],
                        axis=1).astype(np.float32)

    def _get_graph_dict(self) -> dict:
        return {"x": self._get_obs(), "edge_index": self._edge_index, "edge_attr": self._edge_attr}
