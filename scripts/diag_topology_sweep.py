"""
Diagnostic-only (not part of the committed env): compare SIR distributions across
three topology variants to find which lever actually fixes the interference-limited
finding from diag_channel3.py, before touching the real config/env.

Root cause under test: NetworkSlicingEnv.reset() draws gNB positions AND UE positions
from the SAME uniform(0, area_size) independently — a UE "belonging" to gNB i (by
index slice) is not spatially clustered around gNB i at all. Under a log-distance
path-loss model, uniformly scaling area_size shifts every link's path loss by the
same additive constant (since PL(k*d) = PL(d) + B*log10(k)), so it should be close to
a no-op for SIR/SINR (own-link and interference-link both shift together) — this
script checks whether that's actually true here (breakpoint-distance model + LOS
probability + shadow fading make it not perfectly scale-invariant) versus whether
separating "UE-to-own-gNB radius" from "gNB-to-gNB spacing" (the real fix if the
above holds) or reducing n_gnb (fewer simultaneous interferers) does better.

Usage: python scripts/diag_topology_sweep.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from envs.channel_model import ChannelModel

N_SEEDS = 10
N_UE_PER_GNB = 5
URLLC_FRAC = 0.5


def build_positions(rng, n_gnb, area_size, ue_mode, ue_radius=None):
    gnb_pos = rng.uniform(0, area_size, (n_gnb, 2))
    if ue_mode == "independent":
        ue_pos = rng.uniform(0, area_size, (n_gnb * N_UE_PER_GNB, 2))
    elif ue_mode == "clustered":
        ue_pos = np.zeros((n_gnb * N_UE_PER_GNB, 2))
        for i in range(n_gnb):
            offset = rng.uniform(-ue_radius, ue_radius, (N_UE_PER_GNB, 2))
            ue_pos[i * N_UE_PER_GNB:(i + 1) * N_UE_PER_GNB] = gnb_pos[i] + offset
    else:
        raise ValueError(ue_mode)
    return gnb_pos, ue_pos


def measure(cm: ChannelModel, gnb_pos, ue_pos, n_gnb, urllc_frac, rng):
    pl_matrix = cm.build_channel_matrix(gnb_pos, ue_pos, rng)
    rx_dbm = np.array([
        cm.compute_rx_power_dbm(pl_matrix[i, i * N_UE_PER_GNB:(i + 1) * N_UE_PER_GNB].mean())
        for i in range(n_gnb)
    ])
    interferer_rx_dbm = np.array([
        [
            cm.compute_rx_power_dbm(pl_matrix[j, i * N_UE_PER_GNB:(i + 1) * N_UE_PER_GNB].mean())
            if j != i else -np.inf
            for j in range(n_gnb)
        ] for i in range(n_gnb)
    ])
    interferer_mw = np.where(np.isfinite(interferer_rx_dbm), 10.0 ** (interferer_rx_dbm / 10.0), 0.0)

    sir_db = np.zeros(n_gnb)
    for i in range(n_gnb):
        signal_mw = 10.0 ** (rx_dbm[i] / 10.0)
        interf_mw = float(np.sum(interferer_mw[i] * urllc_frac))
        sir = signal_mw / interf_mw if interf_mw > 0 else float("inf")
        sir_db[i] = 10 * np.log10(sir) if np.isfinite(sir) else 99.0
    return sir_db


def run_variant(label, n_gnb, area_size, ue_mode, ue_radius=None):
    cm = ChannelModel(scenario="UMa", h_bs=25.0, h_ut=1.5, carrier_freq_ghz=3.5,
                       noise_figure_db=7.0, thermal_dbm_per_hz=-174.0, tx_power_dbm=43.0)
    all_sir = []
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        gnb_pos, ue_pos = build_positions(rng, n_gnb, area_size, ue_mode, ue_radius)
        sir_db = measure(cm, gnb_pos, ue_pos, n_gnb, URLLC_FRAC, rng)
        all_sir.append(sir_db)
    all_sir = np.concatenate(all_sir)
    good_frac = float((all_sir > 0).mean())
    print(f"{label:38s} n_gnb={n_gnb:>2} good_SIR_frac={good_frac:.2%}  "
          f"mean={all_sir.mean():7.1f}dB  median={np.median(all_sir):7.1f}dB  "
          f"p10={np.percentile(all_sir,10):7.1f}dB")


if __name__ == "__main__":
    print(f"{'variant':38s} {'':>7} {'':>18}")
    run_variant("baseline (area=500, independent)",  5, 500, "independent")
    run_variant("area=1000 (independent)",            5, 1000, "independent")
    run_variant("area=2000 (independent)",            5, 2000, "independent")
    run_variant("area=500, UE clustered r=50m",        5, 500, "clustered", ue_radius=50)
    run_variant("area=500, UE clustered r=100m",       5, 500, "clustered", ue_radius=100)
    run_variant("area=1000, UE clustered r=100m",      5, 1000, "clustered", ue_radius=100)
    run_variant("n_gnb=3, area=500 (independent)",     3, 500, "independent")
    run_variant("n_gnb=3, area=500, UE clustered r=100m", 3, 500, "clustered", ue_radius=100)
