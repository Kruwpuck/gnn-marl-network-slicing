"""
Calibrate lambda_arrival/delta against a TRAINED policy (Fase 2 #5.2,
docs/rev2-implementation-plan.md). scripts/calibrate_load.py only checks a STATIC
policy -- that gate passed while trained policies ran far below delta, so the CMDP
constraint never actually bound during training. This trains the cheapest baseline
(ippo -- chosen because it's cheapest to iterate on, not because of its results) to
convergence and measures its held-out violation against cmdp.delta.

One training + one eval per invocation. Not a fully automated multi-round loop:
each round costs a full ippo training, and the config being tuned is checked into
version control -- deciding the next lambda_arrival/delta and editing the YAML
should be a deliberate human step between rounds, not something this script does
unattended.

Usage:
  python scripts/calibrate_trained.py --seed 42
  python scripts/calibrate_trained.py --seed 42 --skip-train   # re-eval an existing run
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
CALIB_TAG = "_calib"  # distinct from the main wave's ippo_seed{N} -- must not collide


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=1_000_000, help="ippo is PPO family -> 1M steps, matches main wave")
    p.add_argument("--config", type=str, default=None, help="default: configs/experiment_config.yaml")
    p.add_argument("--episodes", type=int, default=30, help="held-out eval episodes")
    p.add_argument("--skip-train", action="store_true", help="reuse an existing calibration checkpoint, only re-eval")
    args = p.parse_args()

    config_path = args.config or str(ROOT / "configs" / "experiment_config.yaml")
    cfg = yaml.safe_load(Path(config_path).read_text())
    delta = float(cfg["cmdp"]["delta"])
    lambda_arrival = float(cfg["traffic"]["urllc"]["lambda_arrival"])

    run_name = f"ippo{CALIB_TAG}_seed{args.seed}"
    pt_path = ROOT / "results" / "logs" / f"{run_name}.pt"
    eval_dir = ROOT / "results" / "eval_calib"

    if not args.skip_train:
        cmd = [sys.executable, "training/train_baselines.py", "--algo", "ippo",
               "--steps", str(args.steps), "--seed", str(args.seed),
               "--config", config_path, "--tag", CALIB_TAG,
               "--resume", "--ckpt-interval", "25000"]
        print(f"training {run_name}: steps={args.steps} config={config_path} lambda_arrival={lambda_arrival} delta={delta}")
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
    held_out_violation = float(df["sla_violation_pct"].mean()) / 100.0  # fraction, matches delta's units

    print()
    print(f"held-out violation (mean over {len(df)} episodes): {held_out_violation:.4f}  ({held_out_violation*100:.2f}%)")
    print(f"delta: {delta:.4f}  ({delta*100:.2f}%)")
    gap_pp = (delta - held_out_violation) * 100.0
    print(f"gap: {gap_pp:+.2f}pp (delta - violation; target: violation within 5-12pp below delta, i.e. gap in [0, 7]pp roughly -- plan target is violation itself in [5%,12%])")

    if 0.05 <= held_out_violation <= 0.12:
        print("PASS -- held-out violation sits in the target [5%, 12%] band. Freeze lambda_arrival/delta here.")
        return 0

    if held_out_violation < 0.05:
        # violation too low -> constraint isn't binding -> raise load or lower delta
        suggested_lambda = lambda_arrival * (1.0 + (0.08 - held_out_violation))  # heuristic step toward mid-band
        print(f"FAIL -- violation too low (constraint not binding). Try raising lambda_arrival "
              f"{lambda_arrival:.0f} -> ~{suggested_lambda:.0f} (traffic/urllc/lambda_arrival in "
              f"configs/experiment_config.yaml), or lowering delta further, then re-run.")
    else:
        suggested_lambda = lambda_arrival * (1.0 - (held_out_violation - 0.08))
        print(f"FAIL -- violation too high (target overshot). Try lowering lambda_arrival "
              f"{lambda_arrival:.0f} -> ~{suggested_lambda:.0f}, or raising delta, then re-run.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
