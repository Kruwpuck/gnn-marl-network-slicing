import numpy as np


class ChannelModel:
    """3GPP TR 38.901 path-loss, shadow fading, SINR, and interference graph."""

    SPEED_OF_LIGHT = 3e8

    def __init__(self, scenario: str = "UMa", h_bs: float = 25.0, h_ut: float = 1.5,
                 carrier_freq_ghz: float = 3.5, noise_figure_db: float = 7.0,
                 thermal_dbm_per_hz: float = -174.0, tx_power_dbm: float = 43.0):
        assert scenario in ("UMa", "UMi"), f"Unknown scenario: {scenario}"
        self.scenario = scenario
        self.h_bs = h_bs
        self.h_ut = h_ut
        self.fc = carrier_freq_ghz
        self.noise_figure_db = noise_figure_db
        self.thermal_dbm_per_hz = thermal_dbm_per_hz
        self.tx_power_dbm = tx_power_dbm

        if scenario == "UMa":
            self.sigma_los = 4.0
            self.sigma_nlos = 6.0
        else:
            self.sigma_los = 4.0
            self.sigma_nlos = 7.82

    def _breakpoint_distance(self) -> float:
        h_bs_eff = self.h_bs - 1.0
        h_ut_eff = self.h_ut - 1.0
        return 4.0 * h_bs_eff * h_ut_eff * self.fc * 1e9 / self.SPEED_OF_LIGHT

    def compute_path_loss(self, d3d: float, d2d: float, los: bool) -> float:
        """Return path-loss in dB (no shadow fading), per TR 38.901 Table 7.4.1."""
        fc = self.fc
        h_bs, h_ut = self.h_bs, self.h_ut
        d_bp = self._breakpoint_distance()

        if self.scenario == "UMa":
            if los:
                if d2d <= d_bp:
                    pl = 28.0 + 22.0 * np.log10(d3d) + 20.0 * np.log10(fc)
                else:
                    pl = (28.0 + 40.0 * np.log10(d3d) + 20.0 * np.log10(fc)
                          - 9.0 * np.log10(d_bp**2 + (h_bs - h_ut)**2))
            else:
                pl_los = self.compute_path_loss(d3d, d2d, los=True)
                pl_nlos = 13.54 + 39.08 * np.log10(d3d) + 20.0 * np.log10(fc) - 0.6 * (h_ut - 1.5)
                pl = max(pl_los, pl_nlos)
        else:  # UMi
            if los:
                if d2d <= d_bp:
                    pl = 32.4 + 21.0 * np.log10(d3d) + 20.0 * np.log10(fc)
                else:
                    pl = (32.4 + 40.0 * np.log10(d3d) + 20.0 * np.log10(fc)
                          - 9.5 * np.log10(d_bp**2 + (h_bs - h_ut)**2))
            else:
                pl_los = self.compute_path_loss(d3d, d2d, los=True)
                pl_nlos = (35.3 * np.log10(d3d) + 22.4 + 21.3 * np.log10(fc)
                           - 0.3 * (h_ut - 1.5))
                pl = max(pl_los, pl_nlos)

        return float(pl)

    def add_shadow_fading(self, pl_db: float, los: bool, rng: np.random.Generator) -> float:
        sigma = self.sigma_los if los else self.sigma_nlos
        return pl_db + float(rng.normal(0.0, sigma))

    def compute_rx_power_dbm(self, pl_db: float) -> float:
        return self.tx_power_dbm - pl_db

    def compute_noise_dbm(self, bandwidth_hz: float) -> float:
        return self.thermal_dbm_per_hz + 10.0 * np.log10(bandwidth_hz) + self.noise_figure_db

    def compute_sinr(self, rx_power_dbm: float, interference_dbm: float,
                     bandwidth_hz: float) -> float:
        """Return SINR as linear ratio."""
        noise_dbm = self.compute_noise_dbm(bandwidth_hz)

        def dbm_to_mw(x):
            return 10.0 ** (x / 10.0)

        signal_mw = dbm_to_mw(rx_power_dbm)
        noise_mw = dbm_to_mw(noise_dbm)
        interf_mw = dbm_to_mw(interference_dbm) if interference_dbm > -200 else 0.0
        denom = noise_mw + interf_mw
        return float(signal_mw / denom) if denom > 0 else 1e6

    def compute_rate(self, sinr_linear: float, bandwidth_hz: float) -> float:
        """Shannon capacity in bps."""
        return float(bandwidth_hz * np.log2(1.0 + max(sinr_linear, 0.0)))

    def build_channel_matrix(self, gnb_positions: np.ndarray, ue_positions: np.ndarray,
                              rng: np.random.Generator) -> np.ndarray:
        """Path-loss matrix (n_gnb, n_ue) in dB."""
        n_gnb, n_ue = gnb_positions.shape[0], ue_positions.shape[0]
        pl_matrix = np.zeros((n_gnb, n_ue))
        for i in range(n_gnb):
            for j in range(n_ue):
                d2d = float(np.linalg.norm(gnb_positions[i, :2] - ue_positions[j, :2]))
                d2d = max(d2d, 10.0)
                d3d = float(np.sqrt(d2d**2 + (self.h_bs - self.h_ut)**2))
                los = rng.random() < self._los_probability(d2d)
                pl = self.compute_path_loss(d3d, d2d, los)
                pl = self.add_shadow_fading(pl, los, rng)
                pl_matrix[i, j] = pl
        return pl_matrix

    def build_interference_graph(self, gnb_positions: np.ndarray
                                  ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fully-connected inter-gNB interference graph.
        Returns edge_index (2, E) and edge_attr (E, 1) path-loss in dB.
        """
        n = gnb_positions.shape[0]
        src, dst, attrs = [], [], []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d2d = float(np.linalg.norm(gnb_positions[i, :2] - gnb_positions[j, :2]))
                d2d = max(d2d, 10.0)
                d3d = float(np.sqrt(d2d**2))
                pl = self.compute_path_loss(d3d, d2d, los=True)
                src.append(i)
                dst.append(j)
                attrs.append([pl])
        edge_index = np.array([src, dst], dtype=np.int64)
        edge_attr = np.array(attrs, dtype=np.float32)
        return edge_index, edge_attr

    def _los_probability(self, d2d: float) -> float:
        """TR 38.901 LOS probability (simplified)."""
        if d2d <= 18.0:
            return 1.0
        if self.scenario == "UMa":
            return 18.0 / d2d + np.exp(-d2d / 63.0) * (1.0 - 18.0 / d2d)
        return 18.0 / d2d + np.exp(-d2d / 36.0) * (1.0 - 18.0 / d2d)
