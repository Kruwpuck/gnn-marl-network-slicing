"""
Calibrate lambda_arrival/delta against a TRAINED policy, and evaluate Gate A of
handoff/goal1.md (A1 constraint binding, A2 dual active, A3 drop mechanism, A4
window variance).

scripts/calibrate_load.py only checks a STATIC policy -- that gate passed while
trained policies ran far below delta, so the CMDP constraint never actually bound
during training (v3 root cause). This trains the *reference baseline* to
convergence and measures its held-out violation against cmdp.delta.

Reference baseline is `ippo`, frozen in handoff/goal1.md Gate A1 and required to
be non-GNN, so the operating point is never set by proposed-model behaviour. It
was `central-ppo` until the 2026-08-06 amendment: central-ppo sees global state
but emits ONE PRB tier broadcast to all gNB (training/train_baselines.py:
`actions = np.full(n_gnb, central_action)`), so it would have fixed the operating
point at the floor of the NARROWEST action space in the wave. `ippo` is per-gNB
and still non-GNN.

Gate A2 was split by the same amendment, because "lambda >= 5.0" measures the
SIZE of the dual, not whether the constraint binds -- round 3 passed it with
lambda 1.03 -> 18.59 and no change in policy behaviour at all. A2a (feasibility)
takes the action-space violation floor from scripts/probe_action_floor.py via
--floor-pct. A2b (sensitivity) needs a lambda-frozen control run -- identical
config with cmdp.lambda_lr = 0 -- passed as --control-tag. Both are measured on
the deterministic held-out policy; A3/A4 stay on the stochastic training log
(goal1.md, "Mode pengukuran").

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
REFERENCE_BASELINE = "ippo"  # handoff/goal1.md Gate A1 (amended 2026-08-06, see docstring)


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
    p.add_argument("--tag", type=str, default=CALIB_TAG,
                   help="run tag; use a fresh one per calibration round (_calib2, _calib3, ...) so "
                        "--resume never mixes a new operating point into an old checkpoint, and so "
                        "earlier rounds stay on disk as evidence (goal1.md integritas #3)")
    p.add_argument("--floor-pct", type=float, default=None,
                   help="Gate A2a: violation floor of this algo's action space, in percent, from "
                        "scripts/probe_action_floor.py. Not measured -> A2a cannot be evaluated")
    p.add_argument("--control-tag", type=str, default=None,
                   help="Gate A2b: tag of the lambda-frozen control run (same config, "
                        "cmdp.lambda_lr = 0). Not given -> A2b cannot be evaluated")
    p.add_argument("--as-control", action="store_true",
                   help="train THIS run as the lambda-frozen control for A2b: same config with "
                        "cmdp.lambda_lr = 0 (lambda stays at lambda_init), derived from the frozen "
                        "config at runtime so the two runs can never drift apart. Gates are not "
                        "meaningful for a control run; only its held-out violation is used")
    args = p.parse_args()

    config_path = args.config or str(ROOT / "configs" / "experiment_config.yaml")
    cfg = yaml.safe_load(Path(config_path).read_text())

    if args.as_control:
        cfg["cmdp"]["lambda_lr"] = 0.0
        control_path = ROOT / "results" / "eval_calib" / "config_lamfrozen.yaml"
        control_path.parent.mkdir(parents=True, exist_ok=True)
        control_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        config_path = str(control_path)
        print(f"control run: lambda_lr = 0 (lambda pinned at {cfg['cmdp']['lambda_init']}), "
              f"config written to {control_path}")
    delta = float(cfg["cmdp"]["delta"])
    dual_update_every = int(cfg["cmdp"]["dual_update_every"])
    lambda_arrival = float(cfg["traffic"]["urllc"]["lambda_arrival"])

    if args.algo != REFERENCE_BASELINE:
        print(f"WARNING: --algo {args.algo} != {REFERENCE_BASELINE}. Gate A1 of handoff/goal1.md "
              f"is defined on {REFERENCE_BASELINE}; this run is a probe, not a gate result.")

    run_name = f"{args.algo}{args.tag}_seed{args.seed}"
    csv_path = ROOT / "results" / "logs" / f"{run_name}.csv"
    pt_path = ROOT / "results" / "logs" / f"{run_name}.pt"
    eval_dir = ROOT / "results" / "eval_calib"

    if not args.skip_train:
        cmd = [sys.executable, "training/train_baselines.py", "--algo", args.algo,
               "--steps", str(args.steps), "--seed", str(args.seed),
               "--config", config_path, "--tag", args.tag,
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

    if args.as_control:
        print(f"\nCONTROL (lambda frozen) -- {run_name}: held-out violation "
              f"{violation*100:.2f}%. Feed it to the real run as --control-tag {args.tag}")
        return 0

    a = gate_a234(csv_path, dual_update_every)

    # A2a/A2b are both scaled by the window std: a shift smaller than the noise the dual
    # itself sees is not a shift, and a feasibility margin smaller than it is not a margin.
    std_pp = a["window_std_pp"]
    lo, hi = 0.7 * delta, 1.0 * delta
    gates = {
        "A1 violation in [0.7d, 1.0d]": (lo <= violation <= hi,
            f"{violation*100:.2f}% vs [{lo*100:.2f}%, {hi*100:.2f}%]"),
    }

    if args.floor_pct is None:
        gates["A2a floor <= delta - 1 std"] = (None, "not measured -- pass --floor-pct "
                                                     "(scripts/probe_action_floor.py)")
    else:
        margin_pp = delta * 100.0 - args.floor_pct
        gates["A2a floor <= delta - 1 std"] = (margin_pp >= std_pp,
            f"floor {args.floor_pct:.2f}%, margin {margin_pp:.2f}pp vs std {std_pp:.2f}pp")

    if args.control_tag is None:
        gates["A2b |viol shift| >= 1 std"] = (None, "not measured -- pass --control-tag "
                                                    "(lambda-frozen run, cmdp.lambda_lr = 0)")
    else:
        ctl_csv = eval_dir / f"{args.algo}{args.control_tag}_seed{args.seed}_eval.csv"
        if not ctl_csv.exists():
            print(f"no control eval at {ctl_csv} -- train the lambda-frozen run first")
            return 1
        ctl_viol = float(pd.read_csv(ctl_csv)["sla_violation_pct"].mean()) / 100.0
        shift_pp = abs(ctl_viol - violation) * 100.0
        gates["A2b |viol shift| >= 1 std"] = (shift_pp >= std_pp,
            f"{shift_pp:.2f}pp vs std {std_pp:.2f}pp  "
            f"(lam-frozen {ctl_viol*100:.2f}% -> lam-active {violation*100:.2f}%)")

    gates["A3 deadline:overflow >= 3:1"] = (a["drop_ratio"] >= 3.0,
        f"{a['drop_ratio']:.2f}:1  (late {a['drop_late_pct']:.2f}%, overflow {a['drop_overflow_pct']:.2f}%)")
    gates["A4 window std < 2.0pp"] = (a["window_std_pp"] < 2.0,
        f"{a['window_std_pp']:.2f}pp over {a['n_windows']} windows")

    print()
    print(f"GATE A -- {run_name}  (delta={delta:.4f}, lambda_arrival={lambda_arrival:.0f}, "
          f"dual_update_every={dual_update_every})")
    for name, (ok, detail) in gates.items():
        mark = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
        print(f"  [{mark}] {name:<32} {detail}")
    # lam_ss is no longer a gate (it measured the dual's size, not whether it binds), but
    # it stays in the report: a lambda still climbing at the end of training is evidence
    # the constraint has no fixed point.
    print(f"  [----] lam steady-state (report only)   {a['lam_ss']:.3f}")

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
