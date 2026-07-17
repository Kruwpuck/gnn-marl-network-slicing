import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from envs.network_slicing_env import NetworkSlicingEnv

URLLC_FRAC = 0.5  # representative mid allocation, matches the flat-violation regime

for seed in range(5):
    env = NetworkSlicingEnv()
    env.floor_mode = "none"
    env.cmdp_enabled = False
    env.reset(seed=seed)
    n = env.n_gnb

    # nearest-neighbor gNB-to-gNB distance
    pos = env._gnb_positions
    dmat = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(dmat, np.inf)
    nn_dist = dmat.min(axis=1)

    rx_dbm = np.array([
        env.channel_model.compute_rx_power_dbm(
            env._pl_matrix[i, i * env.n_ue_per_gnb:(i + 1) * env.n_ue_per_gnb].mean()
        ) for i in range(n)
    ])
    interferer_rx_dbm = np.array([
        [
            env.channel_model.compute_rx_power_dbm(
                env._pl_matrix[j, i * env.n_ue_per_gnb:(i + 1) * env.n_ue_per_gnb].mean()
            ) if j != i else -np.inf
            for j in range(n)
        ] for i in range(n)
    ])
    interferer_mw = np.where(np.isfinite(interferer_rx_dbm), 10.0 ** (interferer_rx_dbm / 10.0), 0.0)
    bw_urllc = env.bandwidth_hz * URLLC_FRAC

    print(f"\n=== seed={seed} ===")
    print(f"{'gNB':>4}{'nn_dist_m':>11}{'SINR_dB':>10}{'SIR_dB':>10}{'SNR_only_dB':>13}{'label':>22}")
    for i in range(n):
        signal_mw = 10.0 ** (rx_dbm[i] / 10.0)

        # full interference, all neighbors at URLLC_FRAC (current regime)
        interf_mw_full = float(np.sum(interferer_mw[i] * URLLC_FRAC))
        interf_dbm_full = 10.0 * np.log10(interf_mw_full + 1e-30) if interf_mw_full > 0 else -200.0
        sinr = env.channel_model.compute_sinr(rx_dbm[i], interf_dbm_full, bw_urllc)
        sinr_db = 10 * np.log10(sinr + 1e-12)

        # SIR: signal vs interference only, no thermal noise
        sir = signal_mw / interf_mw_full if interf_mw_full > 0 else float("inf")
        sir_db = 10 * np.log10(sir) if np.isfinite(sir) else float("inf")

        # best case: all neighbors silent on URLLC (interf=0) -> noise-limited only
        snr_only = env.channel_model.compute_sinr(rx_dbm[i], -200.0, bw_urllc)
        snr_only_db = 10 * np.log10(snr_only + 1e-12)

        if snr_only_db < 0:
            label = "noise-limited"
        elif sinr_db < 0:
            label = "interference-limited"
        else:
            label = "ok"
        print(f"{i:>4}{nn_dist[i]:>11.1f}{sinr_db:>10.1f}"
              f"{sir_db:>10.1f}{snr_only_db:>13.1f}{label:>22}")
