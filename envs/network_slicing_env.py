from __future__ import annotations

from collections import deque

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
        [ch_gain, sinr_embb_db_norm, sinr_urllc_db_norm, q_embb_log,
         urllc_backlog_log, urllc_viol_rate_ewma, neighbor_urllc_frac_mean, prev_alloc]

    Action (MultiDiscrete): tier index per gNB, 0..10 = 0%..100% URLLC PRB (step 10%).

    Co-slice interference (env.slice_coupled_interference=True, reuse-1): interference
    a gNB receives on a given slice's subband scales with how much of that subband its
    neighbors are also using for the same slice — allocation decisions couple across
    gNBs, so coordination (not just per-gNB optimization) matters.

    BUG FIX (v2 -> v3), documented explicitly because it invalidates v1/v2 results:
    reset() used to draw gNB positions and UE positions independently from the same
    uniform(0, area_size) square, so a UE "belonging" to gNB i (by index slice, see
    pl_matrix indexing below) had a position uncorrelated with gNB i's position — it
    could easily be closer to a neighboring gNB than to its own serving cell. That is
    not a valid 3GPP UMa scenario (UEs are dropped within their serving cell), and
    diagnostics (scripts/diag_topology_sweep.py) confirmed most gNBs were
    interference-limited as a direct consequence: only ~28% had positive SIR,
    independent of area_size (uniform log-distance path-loss scaling shifts every
    link's PL by the same constant, so it cancels in the SIR ratio — area_size was
    never the lever) and only weakly dependent on n_gnb. Clustering each UE within
    env.ue_radius_m of its serving gNB (below) fixed it: ~92-98% of gNBs get positive
    SIR across seeds. All v1 (results/v1_uncoupled/) and v2 (results/v2_scalarized/)
    results used the buggy uncorrelated placement and are NOT representative of a
    valid scenario — they are archived for historical/ablation reference only, not as
    a baseline to build on. Only v3 results (this env) should be reported in the paper.

    URLLC is modeled at packet level with an explicit slot duration (env.slot_duration_ms):
    packets sit in a per-gNB FIFO, are dropped on deadline expiry (>slices.urllc.max_delay_ms
    wait) or buffer overflow (>buffer.urllc_max_bits backlog), and delay reported for
    delivered packets is therefore always <= the deadline by construction. eMBB stays a
    fluid queue (no per-packet deadline) but throughput is now *served* bits, not raw
    channel capacity.

    Objective/constraint split (CMDP, cmdp.enabled=true): the reward is throughput only
    (w1 eMBB + w4 spectral efficiency) minus a Lagrangian penalty lambda * violation_rate.
    lambda is a dual variable updated on a slow timescale (cmdp.dual_update_every steps)
    toward the target violation rate cmdp.delta, and persists across episode resets within
    a training run (see get_cmdp_state/set_cmdp_state) — it is NOT part of the per-episode
    reset. A minimum-PRB floor for URLLC (floor.mode: none|static|dynamic) is applied via
    action projection, identically regardless of which agent produced the action.
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
        self.ue_radius_m: float = float(cfg["env"].get("ue_radius_m", 100.0))
        self.episode_length: int = cfg["training"]["episode_length"]
        self.slice_coupled_interference: bool = bool(
            cfg["env"].get("slice_coupled_interference", True)
        )
        self.slot_ms: float = float(cfg["env"].get("slot_duration_ms", 1.0))
        self.slot_s: float = self.slot_ms / 1000.0

        self.w1 = float(cfg["reward"]["w1"])
        self.w2 = float(cfg["reward"].get("w2", 0.0))  # only used if cmdp.enabled=false
        self.w4 = float(cfg["reward"]["w4"])

        self.urllc_max_delay_ms: float = float(cfg["slices"]["urllc"]["max_delay_ms"])
        self.urllc_reliability_target: float = float(cfg["slices"]["urllc"].get("reliability", 0.999))
        self.embb_min_throughput: float = float(cfg["slices"]["embb"]["min_throughput_mbps"]) * 1e6

        self.urllc_max_bits: float = float(cfg.get("buffer", {}).get("urllc_max_bits", 40960.0))
        self.embb_max_bits: float = float(cfg.get("buffer", {}).get("embb_max_bits", 50_000_000.0))

        cmdp_cfg = cfg.get("cmdp", {})
        self.cmdp_enabled: bool = bool(cmdp_cfg.get("enabled", True))
        self.delta: float = float(cmdp_cfg.get("delta", 0.02))
        self.lambda_init: float = float(cmdp_cfg.get("lambda_init", 1.0))
        self.lambda_lr: float = float(cmdp_cfg.get("lambda_lr", 0.01))
        self.lambda_max: float = float(cmdp_cfg.get("lambda_max", 100.0))
        self.dual_update_every: int = int(cmdp_cfg.get("dual_update_every", 2000))

        floor_cfg = cfg.get("floor", {})
        self.floor_mode: str = floor_cfg.get("mode", "none")
        self.floor_base: float = float(floor_cfg.get("base", 0.10))
        self.floor_k: float = float(floor_cfg.get("k", 0.30))
        self.floor_lo: float = float(floor_cfg.get("lo", 0.10))
        self.floor_hi: float = float(floor_cfg.get("hi", 0.40))

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
        self._urllc_fifo: list[deque] | None = None
        self._prev_alloc: np.ndarray | None = None
        self._last_sinr_embb: np.ndarray | None = None
        self._last_sinr_urllc: np.ndarray | None = None
        self._last_backlog_bits: np.ndarray | None = None
        self._viol_ewma: np.ndarray | None = None
        self._urllc_traffic: list = []
        self._embb_traffic: list = []
        self._step_count: int = 0

        # CMDP dual state — persists across reset(), only re-initialized in __init__.
        self._lam: float = self.lambda_init
        self._total_steps: int = 0
        self._viol_window: deque = deque(maxlen=max(self.dual_update_every, 1))

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
        # Each UE is dropped within ue_radius_m of its OWN serving gNB (see bug-fix
        # note in the class docstring), not independently across the whole area.
        # UE block [i*n_ue_per_gnb : (i+1)*n_ue_per_gnb] belongs to gNB i (matches the
        # pl_matrix indexing used everywhere below), so np.repeat keeps that grouping.
        gnb_repeat = np.repeat(self._gnb_positions, self.n_ue_per_gnb, axis=0)
        ue_offsets = self._rng.uniform(-self.ue_radius_m, self.ue_radius_m, (n_ue_total, 2))
        # Clip (not reflect) to the map bounds: simplest way to guarantee no UE ends
        # up outside [0, area_size], which would otherwise happen for gNBs near an edge.
        self._ue_positions = np.clip(gnb_repeat + ue_offsets, 0.0, self.area_size)

        self._pl_matrix = self.channel_model.build_channel_matrix(
            self._gnb_positions, self._ue_positions, self._rng
        )
        self._edge_index, self._edge_attr = self.channel_model.build_interference_graph(
            self._gnb_positions
        )

        self._queue_embb = np.zeros(self.n_gnb, dtype=np.float32)
        self._urllc_fifo = [deque() for _ in range(self.n_gnb)]
        self._prev_alloc = np.zeros(self.n_gnb, dtype=np.float32)
        self._last_sinr_embb = np.zeros(self.n_gnb, dtype=np.float32)
        self._last_sinr_urllc = np.zeros(self.n_gnb, dtype=np.float32)
        self._last_backlog_bits = np.zeros(self.n_gnb, dtype=np.float32)
        self._viol_ewma = np.zeros(self.n_gnb, dtype=np.float32)
        self._step_count = 0
        # NOTE: self._lam / self._total_steps / self._viol_window are NOT reset here —
        # the CMDP dual variable must persist across episode boundaries within a run.

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

    def get_cmdp_state(self) -> dict:
        return {
            "lam": self._lam,
            "total_steps": self._total_steps,
            "viol_window": list(self._viol_window),
        }

    def set_cmdp_state(self, state: dict) -> None:
        if not state:
            return
        self._lam = float(state.get("lam", self.lambda_init))
        self._total_steps = int(state.get("total_steps", 0))
        self._viol_window = deque(state.get("viol_window", []), maxlen=max(self.dual_update_every, 1))

    def step(self, actions):
        assert self._rng is not None, "Call reset() before step()"
        actions = np.asarray(actions)
        n = self.n_gnb

        # --- 1. traffic arrivals for this slot ---
        for i in range(n):
            n_pkt, _ = self._urllc_traffic[i].step(dt=self.slot_s)
            for _ in range(n_pkt):
                self._urllc_fifo[i].append([self._step_count, float(self.urllc_pkt_bits)])
        embb_arrivals = np.array([self._embb_traffic[i].step(dt=self.slot_s) for i in range(n)],
                                  dtype=np.float32)
        self._queue_embb = np.minimum(self._queue_embb + embb_arrivals, self.embb_max_bits).astype(np.float32)

        arrived = np.array([sum(1 for e in self._urllc_fifo[i] if e[0] == self._step_count)
                             for i in range(n)], dtype=np.int64)

        # --- 2. raw action -> URLLC fraction, then apply PRB floor (action projection) ---
        if self.action_type == "discrete":
            urllc_fracs_raw = actions / (self.n_tiers - 1)
        else:
            acts = np.clip(actions, 0.0, 1.0)
            acts = acts / acts.sum(axis=-1, keepdims=True).clip(min=1e-8)
            urllc_fracs_raw = acts[:, 1]

        backlog_bits = np.array([sum(e[1] for e in self._urllc_fifo[i]) for i in range(n)],
                                 dtype=np.float64)
        f_min = self._compute_floor(backlog_bits)
        urllc_fracs = np.maximum(urllc_fracs_raw, f_min)
        embb_fracs = 1.0 - urllc_fracs
        bw_embb = self.bandwidth_hz * embb_fracs
        bw_urllc = self.bandwidth_hz * urllc_fracs

        # --- 3. SINR / rates (coupled interference, as in v2) ---
        rx_dbm = np.array([
            self.channel_model.compute_rx_power_dbm(
                self._pl_matrix[i, i * self.n_ue_per_gnb:(i + 1) * self.n_ue_per_gnb].mean()
            )
            for i in range(n)
        ])
        interferer_rx_dbm = np.array([
            [
                self.channel_model.compute_rx_power_dbm(
                    self._pl_matrix[j, i * self.n_ue_per_gnb:(i + 1) * self.n_ue_per_gnb].mean()
                )
                if j != i else -np.inf
                for j in range(n)
            ]
            for i in range(n)
        ])
        interferer_mw = 10.0 ** (interferer_rx_dbm / 10.0)
        interferer_mw = np.where(np.isfinite(interferer_rx_dbm), interferer_mw, 0.0)

        embb_cap = np.zeros(n)
        urllc_rates = np.zeros(n)
        sinr_embb = np.zeros(n)
        sinr_urllc = np.zeros(n)

        for i in range(n):
            if self.slice_coupled_interference:
                interf_mw_embb = float(np.sum(interferer_mw[i] * embb_fracs))
                interf_mw_urllc = float(np.sum(interferer_mw[i] * urllc_fracs))
            else:
                interf_mw_embb = float(np.sum(interferer_mw[i]))
                interf_mw_urllc = float(np.sum(interferer_mw[i]))

            interf_dbm_embb = 10.0 * np.log10(interf_mw_embb + 1e-30) if interf_mw_embb > 0 else -200.0
            interf_dbm_urllc = 10.0 * np.log10(interf_mw_urllc + 1e-30) if interf_mw_urllc > 0 else -200.0

            sinr_embb[i] = self.channel_model.compute_sinr(rx_dbm[i], interf_dbm_embb, max(bw_embb[i], 1.0))
            sinr_urllc[i] = self.channel_model.compute_sinr(rx_dbm[i], interf_dbm_urllc, max(bw_urllc[i], 1.0))
            embb_cap[i] = self.channel_model.compute_rate(sinr_embb[i], max(bw_embb[i], 1.0))
            urllc_rates[i] = self.channel_model.compute_rate(sinr_urllc[i], max(bw_urllc[i], 1.0))

        # --- 4. eMBB: serve fluid queue with bits actually delivered (not raw capacity) ---
        embb_served = np.minimum(self._queue_embb, embb_cap * self.slot_s)
        self._queue_embb = (self._queue_embb - embb_served).astype(np.float32)
        embb_thr_bps = (embb_served / self.slot_s).astype(np.float32)

        # --- 5. URLLC: per-gNB FIFO — deadline drop, overflow drop, then service ---
        deadline_slots = self.urllc_max_delay_ms / self.slot_ms
        dropped_late = np.zeros(n, dtype=np.int64)
        dropped_overflow = np.zeros(n, dtype=np.int64)
        delivered_count = np.zeros(n, dtype=np.int64)
        delivered_bits = np.zeros(n, dtype=np.float64)
        delivered_delays: list[list[float]] = [[] for _ in range(n)]

        for i in range(n):
            fifo = self._urllc_fifo[i]

            while fifo and (self._step_count - fifo[0][0]) > deadline_slots:
                fifo.popleft()
                dropped_late[i] += 1

            backlog = sum(e[1] for e in fifo)
            while backlog > self.urllc_max_bits and fifo:
                _, bits = fifo.pop()
                backlog -= bits
                dropped_overflow[i] += 1

            budget = urllc_rates[i] * self.slot_s
            while fifo and budget > 0:
                head = fifo[0]
                take = min(head[1], budget)
                head[1] -= take
                budget -= take
                delivered_bits[i] += take
                if head[1] <= 1e-9:
                    wait_ms = (self._step_count - head[0]) * self.slot_ms
                    delivered_delays[i].append(wait_ms)
                    delivered_count[i] += 1
                    fifo.popleft()

        # Normalize by packets *resolved this step* (delivered + dropped), not packets
        # *arrived* this step: a deadline-drop cohort aged out this step can be larger
        # or smaller than this step's fresh arrivals (FIFO ages one cohort per slot),
        # so arrived-based normalization can exceed 1.0. Resolved-based normalization
        # is bounded to [0, 1] by construction (dropped <= resolved).
        resolved = delivered_count + dropped_late + dropped_overflow
        denom = np.maximum(resolved, 1)
        urllc_violation_rate = (dropped_late + dropped_overflow).astype(np.float64) / denom
        urllc_violation_rate = np.where(resolved > 0, urllc_violation_rate, 0.0)
        self._viol_ewma = (0.9 * self._viol_ewma + 0.1 * urllc_violation_rate).astype(np.float32)

        mean_violation_rate = float(urllc_violation_rate.mean())
        reward = self._compute_reward(embb_thr_bps, mean_violation_rate)

        # --- 6. CMDP dual (Lagrangian) update — slow timescale, persists across episodes ---
        self._viol_window.append(mean_violation_rate)
        self._total_steps += 1
        if self.cmdp_enabled and self.dual_update_every > 0 and self._total_steps % self.dual_update_every == 0:
            mean_viol = float(np.mean(self._viol_window)) if self._viol_window else 0.0
            self._lam = float(np.clip(self._lam + self.lambda_lr * (mean_viol - self.delta),
                                       0.0, self.lambda_max))

        self._prev_alloc = urllc_fracs.astype(np.float32)
        self._last_sinr_embb = sinr_embb.astype(np.float32)
        self._last_sinr_urllc = sinr_urllc.astype(np.float32)
        self._last_backlog_bits = np.array(
            [sum(e[1] for e in self._urllc_fifo[i]) for i in range(n)], dtype=np.float32
        )
        self._step_count += 1

        obs = self._get_obs()
        info = {
            "graph": self._get_graph_dict(),
            "embb_rates": embb_thr_bps,          # served throughput, bps (NOT raw capacity)
            "embb_thr_bps": embb_thr_bps,
            "urllc_violation_rate": urllc_violation_rate,
            "urllc_delivered_delays": delivered_delays,
            "urllc_delivered_bits": delivered_bits,
            "urllc_dropped_late": dropped_late,
            "urllc_dropped_overflow": dropped_overflow,
            "urllc_arrived": arrived,
            "lam": self._lam,
        }
        return obs, reward, False, self._step_count >= self.episode_length, info

    def _compute_floor(self, backlog_bits: np.ndarray) -> np.ndarray:
        n = self.n_gnb
        if self.floor_mode == "none":
            return np.zeros(n)
        if self.floor_mode == "static":
            return np.full(n, self.floor_base)
        q_ref = max(self.urllc_max_bits, 1.0)
        f_min = self.floor_base + self.floor_k * (backlog_bits / q_ref)
        return np.clip(f_min, self.floor_lo, self.floor_hi)

    def _compute_reward(self, embb_thr_bps: np.ndarray, urllc_violation_rate: float) -> float:
        t_ref = max(self.embb_min_throughput, 1.0)
        r_obj = (self.w1 * min(float(embb_thr_bps.mean() / t_ref), 5.0)
                 + self.w4 * float(embb_thr_bps.sum() / self.bandwidth_hz))
        if not self.cmdp_enabled:
            r = r_obj - self.w2 * urllc_violation_rate
        else:
            r = r_obj - self._lam * urllc_violation_rate
        return float(np.clip(r, -10.0, 10.0))

    def _get_obs(self) -> np.ndarray:
        pl_mean = self._pl_matrix.mean(axis=1)
        ch_gain = -pl_mean / 100.0
        sinr_embb_norm = np.clip(10.0 * np.log10(self._last_sinr_embb + 1e-12) / 40.0, -3.0, 3.0)
        sinr_urllc_norm = np.clip(10.0 * np.log10(self._last_sinr_urllc + 1e-12) / 40.0, -3.0, 3.0)
        q_e = np.log1p(self._queue_embb) / 20.0
        urllc_backlog_log = np.log1p(self._last_backlog_bits) / 20.0
        n = self.n_gnb
        if n > 1:
            total = self._prev_alloc.sum()
            neighbor_mean = (total - self._prev_alloc) / (n - 1)
        else:
            neighbor_mean = np.zeros_like(self._prev_alloc)
        return np.stack([ch_gain, sinr_embb_norm, sinr_urllc_norm, q_e,
                         urllc_backlog_log, self._viol_ewma, neighbor_mean, self._prev_alloc],
                        axis=1).astype(np.float32)

    def _get_graph_dict(self) -> dict:
        return {"x": self._get_obs(), "edge_index": self._edge_index, "edge_attr": self._edge_attr}
