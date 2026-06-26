"""
Phase 2a: Training convergence evaluation.

Reads CSV logs from results/logs/ and computes:
  - Cumulative reward curves (moving average, window=100 episodes)
  - Sample efficiency table (reward at 100K, 500K, 1M steps)
  - Training stability (std-dev of moving average)

Usage:
  python evaluation/convergence_eval.py --log-dir results/logs --out-dir results/figures
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


class ConvergenceEvaluator:
    """Load training logs and compute convergence metrics."""

    SAMPLE_CHECKPOINTS = [100_000, 500_000, 1_000_000]

    def __init__(self, log_dir: str | Path = "results/logs"):
        self.log_dir = Path(log_dir)

    def load_log(self, algo: str, seed: int) -> dict:
        """Load CSV log for one algo/seed. Returns dict with arrays."""
        # Try both naming conventions
        candidates = [
            self.log_dir / f"{algo}_seed{seed}.csv",
            self.log_dir / f"{algo}_gat_seed{seed}.csv",
            self.log_dir / f"{algo}_sage_seed{seed}.csv",
            self.log_dir / f"{algo}_gcn_seed{seed}.csv",
        ]
        for path in candidates:
            if path.exists():
                return self._parse_csv(path)
        raise FileNotFoundError(f"No log found for algo={algo} seed={seed}")

    def load_log_path(self, path: str | Path) -> dict:
        return self._parse_csv(Path(path))

    def _parse_csv(self, path: Path) -> dict:
        steps, rewards, losses = [], [], []
        with open(path) as f:
            header = f.readline().strip().split(",")
            step_idx = header.index("step")
            rew_idx = header.index("ep_reward")
            loss_idx = header.index("loss") if "loss" in header else -1
            for line in f:
                parts = line.strip().split(",")
                if len(parts) <= rew_idx:
                    continue
                try:
                    steps.append(int(parts[step_idx]))
                    rewards.append(float(parts[rew_idx]))
                    if loss_idx >= 0 and parts[loss_idx]:
                        losses.append(float(parts[loss_idx]))
                except ValueError:
                    continue
        return {
            "path": str(path),
            "steps": np.array(steps),
            "rewards": np.array(rewards),
            "losses": np.array(losses) if losses else np.array([]),
        }

    def moving_average(self, arr: np.ndarray, window: int = 100) -> np.ndarray:
        if len(arr) < window:
            return arr
        kernel = np.ones(window) / window
        return np.convolve(arr, kernel, mode="valid")

    def sample_efficiency(self, log: dict) -> dict:
        """Reward at each checkpoint step."""
        result = {}
        for ck in self.SAMPLE_CHECKPOINTS:
            mask = log["steps"] <= ck
            if mask.any():
                result[ck] = float(log["rewards"][mask][-1])
            else:
                result[ck] = float("nan")
        return result

    def stability(self, log: dict, window: int = 100) -> float:
        """Std-dev of moving-average reward."""
        ma = self.moving_average(log["rewards"], window)
        return float(np.std(ma)) if len(ma) > 0 else float("nan")

    def compute_all(self, logs: dict[str, dict]) -> dict:
        """
        logs: {algo_label: log_dict}
        Returns nested metrics dict.
        """
        metrics = {}
        for label, log in logs.items():
            metrics[label] = {
                "sample_efficiency": self.sample_efficiency(log),
                "stability_std": self.stability(log),
                "final_reward": float(log["rewards"][-1]) if len(log["rewards"]) > 0 else float("nan"),
                "n_episodes": len(log["rewards"]),
            }
        return metrics

    def print_table(self, metrics: dict) -> None:
        """Print sample-efficiency table to stdout."""
        ck_labels = [f"{c//1000}K" for c in self.SAMPLE_CHECKPOINTS]
        header = f"{'Algorithm':<20} " + " ".join(f"{l:>10}" for l in ck_labels) + f" {'Stability':>12} {'Final':>10}"
        print(header)
        print("-" * len(header))
        for label, m in metrics.items():
            se = m["sample_efficiency"]
            vals = " ".join(f"{se.get(c, float('nan')):>10.4f}" for c in self.SAMPLE_CHECKPOINTS)
            print(f"{label:<20} {vals} {m['stability_std']:>12.4f} {m['final_reward']:>10.4f}")

    def plot_curves(self, logs: dict[str, dict], out_path: str | Path | None = None,
                    window: int = 100) -> None:
        if not HAS_MPL:
            print("matplotlib not installed — skipping plot")
            return
        fig, ax = plt.subplots(figsize=(10, 5))
        for label, log in logs.items():
            ma = self.moving_average(log["rewards"], window)
            x = log["steps"][window - 1:] if len(log["steps"]) >= window else log["steps"]
            ax.plot(x[:len(ma)], ma, label=label)
        ax.set_xlabel("Environment Steps")
        ax.set_ylabel(f"Reward (MA-{window})")
        ax.set_title("Training Convergence")
        ax.legend()
        ax.grid(True, alpha=0.3)
        if out_path:
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved → {out_path}")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="results/logs")
    parser.add_argument("--out-dir", default="results/figures")
    parser.add_argument("--window", type=int, default=100)
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ev = ConvergenceEvaluator(log_dir)
    logs = {}
    for csv_path in sorted(log_dir.glob("*.csv")):
        logs[csv_path.stem] = ev.load_log_path(csv_path)

    if not logs:
        print(f"No CSV logs found in {log_dir}. Run training first.")
    else:
        metrics = ev.compute_all(logs)
        ev.print_table(metrics)
        ev.plot_curves(logs, out_dir / "convergence.png", window=args.window)
        with open(out_dir / "convergence_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics → {out_dir / 'convergence_metrics.json'}")
