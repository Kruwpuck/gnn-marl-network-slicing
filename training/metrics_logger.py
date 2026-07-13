"""
Rich training-metrics logging + resumable checkpointing.

Shared by training/train_proposed.py and training/train_baselines.py so every run
produces a uniform CSV (easy to plot for the paper) and periodic .pt checkpoints
that survive crashes and support --resume.
"""
from __future__ import annotations

import csv
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation.network_perf_eval import jains_fairness  # reuse, don't duplicate

# Union schema across all algos. Cells left blank ("") where a metric doesn't apply.
CSV_COLUMNS = [
    "wall_time",           # ISO-ish timestamp of the row
    "elapsed_sec",         # seconds since training start (survives resume)
    "step",                # environment step index
    "episode",             # completed-episode counter
    "ep_reward",           # reward of the logged episode / rollout window
    "ep_reward_ma100",     # moving average (window=100) of ep_reward
    "sps",                 # steps per second (throughput of training)
    "loss",                # total loss
    "policy_loss",         # PPO only
    "value_loss",          # PPO only
    "entropy",             # PPO only
    "epsilon",             # DQN only
    "embb_mbps_mean",      # eMBB throughput (Mbps)
    "spectral_eff_bps_hz", # spectral efficiency (bps/Hz)
    "urllc_delay_ms_mean", # URLLC mean delay (ms)
    "sla_violation_pct",   # % gNB-steps violating URLLC delay bound
    "sla_satisfaction_pct",# 100 - violation  ("accuracy" analog)
    "jains_fairness",      # Jain's fairness index over eMBB rates
]


class MetricsLogger:
    """Append-safe rich CSV writer with a rolling reward moving-average."""

    def __init__(self, log_path: str | Path, resume: bool = False,
                 ma_window: int = 100, ma_deque: deque | None = None):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._ma = ma_deque if ma_deque is not None else deque(maxlen=ma_window)
        self._t0 = time.time()
        self._elapsed_offset = 0.0

        fresh = not (resume and self.log_path.exists())
        mode = "w" if fresh else "a"
        self._f = open(self.log_path, mode, newline="", buffering=1)
        self._writer = csv.DictWriter(self._f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if fresh:
            self._writer.writeheader()

    def set_elapsed_offset(self, seconds: float) -> None:
        """On resume, account for time already spent in prior runs."""
        self._elapsed_offset = float(seconds)

    @property
    def ma_deque(self) -> deque:
        return self._ma

    def elapsed(self) -> float:
        return self._elapsed_offset + (time.time() - self._t0)

    def log(self, step: int, episode: int, ep_reward: float, **extra) -> float:
        """Write one row. Updates + returns the reward moving average."""
        self._ma.append(float(ep_reward))
        ma = float(np.mean(self._ma)) if self._ma else float(ep_reward)
        elapsed = self.elapsed()
        sps = step / elapsed if elapsed > 0 else 0.0

        row = {c: "" for c in CSV_COLUMNS}
        row.update({
            "wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_sec": round(elapsed, 2),
            "step": step,
            "episode": episode,
            "ep_reward": round(float(ep_reward), 4),
            "ep_reward_ma100": round(ma, 4),
            "sps": round(sps, 1),
        })
        for k, v in extra.items():
            if k in row and v is not None and v != "":
                row[k] = round(float(v), 6) if isinstance(v, (int, float, np.floating)) else v
        self._writer.writerow(row)
        return ma

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass


class EpisodeNetworkStats:
    """Accumulate per-step env `info` and summarize network KPIs over a window."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._embb_bps: list[float] = []   # per-gNB eMBB rates (bps)
        self._delay_ms: list[float] = []   # per-gNB URLLC delay (ms)
        self._fairness: list[float] = []   # per-step Jain's index over eMBB rates

    def add(self, info: dict) -> None:
        embb = info.get("embb_rates")
        delay = info.get("urllc_delay_ms")
        if embb is not None:
            embb = np.asarray(embb, dtype=np.float64).ravel()
            self._embb_bps.extend(embb.tolist())
            self._fairness.append(jains_fairness(embb))
        if delay is not None:
            self._delay_ms.extend(np.asarray(delay, dtype=np.float64).ravel().tolist())

    def summary(self, bandwidth_hz: float, urllc_max_delay_ms: float) -> dict:
        if not self._embb_bps and not self._delay_ms:
            return {}
        embb = np.asarray(self._embb_bps) if self._embb_bps else np.array([0.0])
        delay = np.asarray(self._delay_ms) if self._delay_ms else np.array([0.0])
        viol = float(np.mean(delay > urllc_max_delay_ms) * 100.0)
        return {
            "embb_mbps_mean": float(embb.mean() / 1e6),
            "spectral_eff_bps_hz": float((embb / bandwidth_hz).mean()),
            "urllc_delay_ms_mean": float(delay.mean()),
            "sla_violation_pct": viol,
            "sla_satisfaction_pct": 100.0 - viol,
            "jains_fairness": float(np.mean(self._fairness)) if self._fairness else 0.0,
        }


class CheckpointManager:
    """Periodic, atomic, resumable checkpoints under results/checkpoints/."""

    def __init__(self, run_name: str, ckpt_dir: str | Path = "results/checkpoints"):
        self.run_name = run_name
        self.dir = Path(ckpt_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.last_path = self.dir / f"{run_name}_last.pt"
        self.best_path = self.dir / f"{run_name}_best.pt"
        self._best_ma = -float("inf")

    def _atomic_save(self, state: dict, path: Path) -> None:
        tmp = path.with_suffix(".pt.tmp")
        torch.save(state, tmp)
        os.replace(tmp, path)

    def save(self, state: dict, ma_reward: float | None = None) -> None:
        self._atomic_save(state, self.last_path)
        if ma_reward is not None and ma_reward > self._best_ma:
            self._best_ma = ma_reward
            self._atomic_save(state, self.best_path)

    def load_latest(self) -> dict | None:
        if self.last_path.exists():
            return torch.load(self.last_path, map_location="cpu", weights_only=False)
        return None


def capture_rng_state() -> dict:
    return {
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict) -> None:
    if not state:
        return
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
