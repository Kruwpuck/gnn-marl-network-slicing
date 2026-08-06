"""
Zero-shot topology transfer eval (Fase 1 #4.2, docs/rev2-implementation-plan.md).

Runs existing n_gnb=5 checkpoints (main wave) against n_gnb in {10,20} configs,
no retraining. central-* checkpoints have obs_dim baked in at train time
(n_gnb * 8) -- evaluating at a different n_gnb is a shape mismatch by
construction. That's recorded as CANNOT_RUN, the expected result, not a crash.

Usage:
  python scripts/zeroshot_eval.py
  python scripts/zeroshot_eval.py --n-gnb 20 --episodes 30
"""
from __future__ import annotations
import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.evaluate_checkpoints import evaluate_run
from scripts.run_wave import make_variant_config
from scripts.rliable_report import parse_run_name


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", type=str, default="results/logs/*_seed*.pt")
    p.add_argument("--n-gnb", type=str, default="10,20")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--out-dir", type=str, default="results/eval_zeroshot")
    p.add_argument("--out", type=str, default="results/ZEROSHOT.md")
    args = p.parse_args()

    # main wave only (tag == ''): exclude ablation _floornone / _floorstatic checkpoints
    paths = [Path(pt) for pt in sorted(glob.glob(args.checkpoints)) if "_floor" not in Path(pt).stem]
    n_gnb_list = [int(x) for x in args.n_gnb.split(",")]

    rows = []
    for n_gnb in n_gnb_list:
        config_path = make_variant_config("dynamic", {"env.n_gnb": n_gnb})
        out_dir = Path(args.out_dir) / f"ngnb{n_gnb}"
        for pt_path in paths:
            algo, _, seed = parse_run_name(pt_path.stem)
            try:
                evaluate_run(pt_path, args.episodes, str(config_path), out_dir)
                edf = pd.read_csv(out_dir / f"{pt_path.stem}_eval.csv")
                rows.append((algo, n_gnb, seed, "OK",
                             edf["timely_throughput_mbps"].mean(), edf["embb_p5_mbps"].mean()))
            except RuntimeError as e:
                rows.append((algo, n_gnb, seed, "CANNOT_RUN", None, None))
                print(f"[{pt_path.stem}] n_gnb={n_gnb}: CANNOT_RUN: {e}")

    df = pd.DataFrame(rows, columns=["algo", "n_gnb", "seed", "status", "throughput_mbps", "embb_p5_mbps"])
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(args.out_dir) / "zeroshot_summary.csv", index=False)

    lines = [
        "# Zero-shot topology transfer\n",
        "n_gnb=5 checkpoints (main wave) evaluated at larger n_gnb without retraining. "
        "`central-*` fails with a shape mismatch by construction (obs_dim baked in at "
        "train time) -- CANNOT_RUN there is the expected result, not a bug.\n",
        "| algo | n_gnb | status | mean throughput_mbps | mean embb_p5_mbps | n seeds |",
        "|---|---|---|---|---|---|",
    ]
    for (algo, n_gnb), sub in df.groupby(["algo", "n_gnb"]):
        ok = sub[sub.status == "OK"]
        if ok.empty:
            lines.append(f"| `{algo}` | {n_gnb} | CANNOT_RUN | - | - | 0 |")
        else:
            lines.append(
                f"| `{algo}` | {n_gnb} | OK | {ok.throughput_mbps.mean():.4f} "
                f"| {ok.embb_p5_mbps.mean():.6f} | {len(ok)} |"
            )

    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
