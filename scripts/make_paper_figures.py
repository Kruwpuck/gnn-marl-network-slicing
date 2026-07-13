"""Script 4/4 — turn the training CSV logs into paper-ready figures + tidy data.

Reads every results/logs/*.csv (rich schema from the new logger; tolerant of the
old thin schema) and writes into results/figures/:
  - one PNG per metric (proposed vs baselines overlaid, moving-average smoothed)
  - paper_metrics_long.csv  (tidy: algo, family, backbone, step, metric, value)
  - summary_table.csv       (final + reward at 100K/500K/1M via ConvergenceEvaluator)

  python scripts/make_paper_figures.py [--log-dir results/logs] [--out-dir results/figures]
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not installed. Run:  pip install matplotlib")
    sys.exit(1)

from evaluation.convergence_eval import ConvergenceEvaluator

# metric key -> (nice title, y-axis label)
METRICS = {
    "ep_reward_ma100":      ("Reward Learning Curve", "Reward (MA-100)"),
    "loss":                 ("Training Loss", "Loss"),
    "policy_loss":          ("PPO Policy Loss", "Policy loss"),
    "value_loss":           ("PPO Value Loss", "Value loss"),
    "entropy":              ("PPO Policy Entropy", "Entropy"),
    "embb_mbps_mean":       ("eMBB Throughput", "Throughput (Mbps)"),
    "spectral_eff_bps_hz":  ("Spectral Efficiency", "bps/Hz"),
    "urllc_delay_ms_mean":  ("URLLC Delay", "Delay (ms)"),
    "sla_satisfaction_pct": ("SLA Satisfaction (accuracy)", "SLA satisfaction (%)"),
    "jains_fairness":       ("Jain's Fairness", "Jain's index"),
    "epsilon":              ("Epsilon Decay", "epsilon"),
}
FIG_NAME = {
    "ep_reward_ma100": "reward_learning_curve", "loss": "loss_curve",
    "policy_loss": "ppo_policy_loss", "value_loss": "ppo_value_loss",
    "entropy": "ppo_entropy", "embb_mbps_mean": "throughput_curve",
    "spectral_eff_bps_hz": "spectral_efficiency_curve",
    "urllc_delay_ms_mean": "delay_curve", "sla_satisfaction_pct": "sla_satisfaction_curve",
    "jains_fairness": "jains_fairness_curve", "epsilon": "epsilon_decay",
}


def classify(stem: str):
    proposed = stem.startswith("gnn-")
    backbone = ""
    for b in ("gat", "sage", "gcn"):
        if f"_{b}_" in stem or stem.endswith(f"_{b}"):
            backbone = b
    algo = stem.split("_seed")[0]
    return algo, ("proposed" if proposed else "baseline"), backbone


def load_rich(path: Path) -> dict:
    """Return {'step': np.array, metric: np.array(masked to same length)}.
    Each metric keeps only rows where it is present; paired with its own steps."""
    cols: dict[str, list] = {}
    steps: list[int] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        series = {m: ([], []) for m in METRICS if m in header}
        for row in reader:
            try:
                st = int(float(row["step"]))
            except (ValueError, KeyError, TypeError):
                continue
            for m in series:
                v = row.get(m, "")
                if v not in ("", None):
                    try:
                        series[m][0].append(st)
                        series[m][1].append(float(v))
                    except ValueError:
                        pass
    return {m: (np.array(xs), np.array(ys)) for m, (xs, ys) in series.items() if xs}


def moving_average(y: np.ndarray, window: int = 50) -> np.ndarray:
    if len(y) < window or window <= 1:
        return y
    return np.convolve(y, np.ones(window) / window, mode="valid")


def plot_metric(metric: str, runs: dict, out_dir: Path, smooth: int):
    title, ylabel = METRICS[metric]
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False
    for stem, data in sorted(runs.items()):
        if metric not in data:
            continue
        xs, ys = data[metric]
        # reward already MA in-log; smooth the noisier network/loss curves lightly
        if metric != "ep_reward_ma100":
            ys_s = moving_average(ys, smooth)
            xs_s = xs[len(xs) - len(ys_s):]
        else:
            xs_s, ys_s = xs, ys
        _, fam, _ = classify(stem)
        ax.plot(xs_s, ys_s, label=stem, alpha=0.9,
                linestyle="-" if fam == "proposed" else "--")
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("Environment steps")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    out = out_dir / f"{FIG_NAME[metric]}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("figure ->", out.name)


def write_long_csv(runs: dict, out_path: Path):
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algo", "family", "backbone", "step", "metric", "value"])
        for stem, data in sorted(runs.items()):
            algo, fam, bb = classify(stem)
            for metric, (xs, ys) in data.items():
                for x, y in zip(xs, ys):
                    w.writerow([algo, fam, bb, int(x), metric, y])
    print("tidy data ->", out_path.name)


def write_summary(log_dir: Path, runs: dict, out_path: Path):
    ev = ConvergenceEvaluator(log_dir)
    fields = ["algo", "family", "backbone", "n_points",
              "reward_100k", "reward_500k", "reward_1m", "final_reward",
              "final_sla_satisfaction_pct", "final_embb_mbps", "final_delay_ms",
              "final_jains_fairness"]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for stem, data in sorted(runs.items()):
            algo, fam, bb = classify(stem)
            log = ev.load_log_path(log_dir / f"{stem}.csv")
            se = ev.sample_efficiency(log)
            reward = data.get("ep_reward_ma100", (np.array([]), np.array([])))[1]
            row = [algo, fam, bb, len(reward),
                   se.get(100_000, float("nan")), se.get(500_000, float("nan")),
                   se.get(1_000_000, float("nan")),
                   float(reward[-1]) if len(reward) else float("nan")]
            for m in ("sla_satisfaction_pct", "embb_mbps_mean",
                      "urllc_delay_ms_mean", "jains_fairness"):
                ys = data.get(m, (None, np.array([])))[1]
                row.append(float(ys[-1]) if len(ys) else float("nan"))
            w.writerow(row)
    print("summary ->", out_path.name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default="results/logs")
    ap.add_argument("--out-dir", default="results/figures")
    ap.add_argument("--smooth", type=int, default=50)
    args = ap.parse_args()

    log_dir = (ROOT / args.log_dir) if not Path(args.log_dir).is_absolute() else Path(args.log_dir)
    out_dir = (ROOT / args.out_dir) if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csvs = sorted(log_dir.glob("*.csv"))
    if not csvs:
        print(f"No CSV logs in {log_dir}. Run training first.")
        sys.exit(0)

    runs = {p.stem: load_rich(p) for p in csvs}
    runs = {k: v for k, v in runs.items() if v}
    print(f"Loaded {len(runs)} runs: {', '.join(sorted(runs))}\n")

    for metric in METRICS:
        plot_metric(metric, runs, out_dir, args.smooth)

    write_long_csv(runs, out_dir / "paper_metrics_long.csv")
    write_summary(log_dir, runs, out_dir / "summary_table.csv")
    print(f"\nAll outputs in {out_dir}")
