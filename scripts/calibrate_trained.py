"""
Calibrate lambda_arrival/delta against a TRAINED policy, and evaluate Gate A of
handoff/goal1.md (A1 constraint binding, A2 dual active, A3 drop mechanism, A4
window variance).

scripts/calibrate_load.py only checks a STATIC policy -- that gate passed while
trained policies ran far below delta, so the CMDP constraint never actually bound
during training (v3 root cause). This trains the *reference baseline* to
convergence and measures its held-out violation against cmdp.delta.

Reference baseline is `central-ppo`, frozen in handoff/goal1.md Gate A1 *before*
calibration and required to be non-GNN, so the operating point is never set by
proposed-model behaviour. Note for the preregistration: central-ppo sees global
state but emits ONE PRB tier broadcast to all gNB
(training/train_baselines.py: `actions = np.full(n_gnb, central_action)`), so the
operating point is fixed by a policy with a narrower action space than the
per-gNB algorithms. That is legitimate under A1 but must be stated in the paper.

One training + one eval per invocation. Not a fully automated multi-round loop:
each round costs a full training, and the config being tuned is checked into
version control -- deciding the next lambda_arrival/delta and editing the YAML
should be a deliberate human step between rounds, not something this script does
unattended.

Usage:
  python scripts/calibrate_trained.py --seed 42
  python scripts/calibrate_trained.py --seed 42 --skip-train   # re-eval an existing run
  python scripts/calibrate_trained.py --seed 42 --algo ippo    # cheaper probe, NOT Gate A1
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
CALIB_TAG = "_calib"  # distinct from the main wave's <algo>_seed{N} -- must not collide
REFERENCE_BASELINE = "central-ppo"  # handoff/goal1.md Gate A1, frozen before calibration


def gate_a234(csv_path: Path, dual_update_every: int) -> dict:
    """A2/A3/A4 from the training CSV (training/metrics_logger.py CSV_COLUMNS)."""
    df = pd.read_csv(csv_path)
    for col in ("step", "lam", "sla_violation_pct",
                "urllc_drop_late_pct", "urllc_drop_overflow_pct"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["step", "sla_violation_pct"])
    if df.empty:
        raise SystemExit(f"no usable network-metric rows in {csv_path}")

    # A2: mean lambda over the last 20% of steps
    cutoff = df["step"].max() * 0.8
    lam_ss = float(df.loc[df["step"] >= cutoff, "lam"].mean())

    # A3: aggregate deadline:overflow ratio. Rows are equal-length episode windows
    # (env.episode_length steps each), so the mean of the per-window percentages is
    # the aggregate ratio.
    late = float(df["urllc_drop_late_pct"].mean())
    overflow = float(df["urllc_drop_overflow_pct"].mean())
    ratio = late / overflow if overflow > 0 else float("inf")

    # A4: std of violation across dual-update windows
    windows = df.groupby(df["step"] // dual_update_every)["sla_violation_pct"].mean()
    window_std = float(windows.std(ddof=1)) if len(windows) > 1 else float("nan")

    return {"lam_ss": lam_ss, "drop_late_pct": late, "drop_overflow_pct": overflow,
            "drop_ratio": ratio, "window_std_pp": window_std, "n_windows": len(windows)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--algo", type=str, default=REFERENCE_BASELINE,
                   help=f"reference baseline (Gate A1 requires {REFERENCE_BASELINE}); "
                        "another algo runs as an unofficial probe only")
    p.add_argument("--steps", type=int, default=1_000_000, help="PPO family -> 1M steps, matches main wave")
    p.add_argument("--config", type=str, default=None, help="default: configs/experiment_config.yaml")
    p.add_argument("--episodes", type=int, default=30, help="held-out eval episodes")
    p.add_argument("--skip-train", action="store_true", help="reuse an existing calibration checkpoint, only re-eval")
    args = p.parse_args()

    config_path = args.config or str(ROOT / "configs" / "experiment_config.yaml")
    cfg = yaml.safe_load(Path(config_path).read_text())
    delta = float(cfg["cmdp"]["delta"])
    dual_update_every = int(cfg["cmdp"]["dual_update_every"])
    lambda_arrival = float(cfg["traffic"]["urllc"]["lambda_arrival"])

    if args.algo != REFERENCE_BASELINE:
        print(f"WARNING: --algo {args.algo} != {REFERENCE_BASELINE}. Gate A1 of handoff/goal1.md "
              f"is defined on {REFERENCE_BASELINE}; this run is a probe, not a gate result.")

    run_name = f"{args.algo}{CALIB_TAG}_seed{args.seed}"
    csv_path = ROOT / "results" / "logs" / f"{run_name}.csv"
    pt_path = ROOT / "results" / "logs" / f"{run_name}.pt"
    eval_dir = ROOT / "results" / "eval_calib"

    if not args.skip_train:
        cmd = [sys.executable, "training/train_baselines.py", "--algo", args.algo,
               "--steps", str(args.steps), "--seed", str(args.seed),
               "--config", config_path, "--tag", CALIB_TAG,
               "--resume", "--ckpt-interval", "25000"]
        print(f"training {run_name}: steps={args.steps} config={config_path} "
              f"lambda_arrival={lambda_arrival} delta={delta}")
        r = subprocess.run(cmd, cwd=str(ROOT))
        if r.returncode != 0:
            print(f"training failed, rc={r.returncode}")
            return 1

    if not pt_path.exists():
        print(f"no checkpoint at {pt_path} -- run without --skip-train first")
        return 1

    eval_cmd = [sys.executable, "scripts/evaluate_checkpoints.py", "--run", str(pt_path),
                "--episodes", str(args.episodes), "--config", config_path, "--out-dir", str(eval_dir)]
    r = subprocess.run(eval_cmd, cwd=str(ROOT))
    if r.returncode != 0:
        print(f"eval failed, rc={r.returncode}")
        return 1

    eval_csv = eval_dir / f"{run_name}_eval.csv"
    df = pd.read_csv(eval_csv)
    violation = float(df["sla_violation_pct"].mean()) / 100.0  # fraction, matches delta's units
    a = gate_a234(csv_path, dual_update_every)

    lo, hi = 0.7 * delta, 1.0 * delta
    gates = {
        "A1 violation in [0.7d, 1.0d]": (lo <= violation <= hi,
            f"{violation*100:.2f}% vs [{lo*100:.2f}%, {hi*100:.2f}%]"),
        "A2 lam_ss >= 5.0": (a["lam_ss"] >= 5.0, f"{a['lam_ss']:.3f}"),
        "A3 deadline:overflow >= 3:1": (a["drop_ratio"] >= 3.0,
            f"{a['drop_ratio']:.2f}:1  (late {a['drop_late_pct']:.2f}%, overflow {a['drop_overflow_pct']:.2f}%)"),
        "A4 window std < 2.0pp": (a["window_std_pp"] < 2.0,
            f"{a['window_std_pp']:.2f}pp over {a['n_windows']} windows"),
    }

    print()
    print(f"GATE A -- {run_name}  (delta={delta:.4f}, lambda_arrival={lambda_arrival:.0f}, "
          f"dual_update_every={dual_update_every})")
    for name, (ok, detail) in gates.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<32} {detail}")

    if all(ok for ok, _ in gates.values()):
        print("\nGate A PASS. Freeze lambda_arrival/delta/urllc_max_bits and commit "
              "BEFORE the full wave (goal1.md C6).")
        return 0

    print("\nGate A FAIL. Adjust the operating point on TASK grounds only "
          "(goal1.md Larangan integritas #1), then re-run.")
    if not gates["A1 violation in [0.7d, 1.0d]"][0]:
        direction = "raise" if violation < lo else "lower"
        print(f"  A1: violation {'below' if violation < lo else 'above'} the band -> {direction} "
              f"traffic/urllc/lambda_arrival (now {lambda_arrival:.0f})")
    if a["drop_ratio"] < 3.0:
        print("  A3: overflow-dominated -> raise buffer.urllc_max_bits so the deadline, "
              "not the buffer, is the binding mechanism. NOTE: urllc_max_bits is also "
              "q_ref in env._compute_floor(), so raising it weakens floor=dynamic; "
              "record this in the preregistration.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
