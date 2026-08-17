"""
Zero-shot topology transfer eval (goal1.md §Fallback poin 2).

Runs the n_gnb=5 wave checkpoints against larger topologies with no retraining.
`central-*` has obs_dim = n_gnb * 8 baked in at train time, so evaluating it at a
different n_gnb is a shape mismatch by construction: CANNOT_RUN is the expected
result and is reported as a finding (an architecture that cannot be evaluated
outside its training topology), not as missing data.

Two arms, both reported (goal1.md integritas #3 forbids showing only the
favourable regime). The env docstring states the reason: interferer distance
scales with area_size/sqrt(n_gnb), so raising n_gnb at a fixed area_size
"confounds agent count with coupling strength".

  fixed-area   : area_size stays 500 m      -> more gNBs AND stronger coupling
  const-density: area_size = 500*sqrt(n/5)  -> more gNBs at the training density

Readout: both are produced. P3 (frozen 2026-08-08) makes the stochastic one
primary; greedy is reported alongside and never gates.

Aggregate throughput scales with cell count -- 5 -> 20 gNB roughly quadruples it
whatever the policy does -- so the headline column is per-gNB, with the aggregate
printed next to it.

Usage:
  python scripts/zeroshot_eval.py --episodes 5 --n-gnb 10 --arms const-density   # pilot
  python scripts/zeroshot_eval.py --stochastic
  python scripts/zeroshot_eval.py                                                # greedy pass
"""
from __future__ import annotations
import argparse
import glob
import math
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.mlp_agent import OBS_FEATURES
from scripts.evaluate_checkpoints import evaluate_run
from scripts.rliable_report import parse_run_name, primary_suffix
from scripts.run_wave import BASE_CONFIG, make_variant_config

ARMS = ("fixed-area", "const-density")


def suffix_for(algo: str, suffix: str) -> str:
    """'primary' resolves per family (argmax for DQN, sampled for PPO); else literal."""
    return primary_suffix(algo) if suffix == "primary" else suffix


def readout_label(suffix: str) -> str:
    return {"_eval": "greedy (reported, never gates)",
            "_eval_stoch": "non-greedy — sampled for PPO (P3), epsilon=0.05 for DQN",
            "primary": "primary per family — sampled for PPO (P3), argmax for DQN"}[suffix]


def locked_obs_dim(pt_path: Path) -> int | None:
    """obs_dim of an architecture whose observation width is tied to n_gnb, else None.

    Classified by name the same way scripts/evaluate_checkpoints.py::load_agent does, and
    not by testing whether obs_dim equals n_gnb * OBS_FEATURES: at the training topology
    that product is 5 * 8 = 40, which is exactly mlp-knn-ppo's width, so the "generic" test
    would misclassify the one baseline built to be topology-independent. The number itself
    is read from the checkpoint, not recomputed from the name.
    """
    algo, _, _ = parse_run_name(pt_path.stem)
    if not algo.startswith("central-"):
        return None
    return int(torch.load(pt_path, map_location="cpu", weights_only=False)["obs_dim"])


def base_env() -> tuple[int, float, str]:
    """(n_gnb, area_size, floor_mode) the wave was trained at.

    floor.mode is read, not hard-coded: this script used to evaluate at floor
    `dynamic` while the v4 wave trained at `none`, which silently changes the
    treatment between training and transfer.
    """
    cfg = yaml.safe_load(BASE_CONFIG.read_text())
    return int(cfg["env"]["n_gnb"]), float(cfg["env"]["area_size"]), str(cfg["floor"]["mode"])


def arm_config(arm: str, n_gnb: int, base_n: int, base_area: float, floor_mode: str) -> Path:
    if arm == "fixed-area":
        return make_variant_config(floor_mode, {"env.n_gnb": n_gnb})
    area = round(base_area * math.sqrt(n_gnb / base_n), 1)
    return make_variant_config(floor_mode, {"env.n_gnb": n_gnb, "env.area_size": area})


def summarise(df: pd.DataFrame, n_gnb: int) -> dict:
    return {
        "throughput_per_gnb_mbps": df["timely_throughput_mbps"].mean() / n_gnb,
        "throughput_mbps": df["timely_throughput_mbps"].mean(),
        "sla_satisfaction_pct": df["sla_satisfaction_pct"].mean(),
        "embb_p5_mbps": df["embb_p5_mbps"].mean(),
    }


def reference_rows(paths: list[Path], base_n: int, suffix: str) -> list[dict]:
    """n_gnb=5 baseline, read from the wave's own eval output. Re-running it would
    burn GPU hours to reproduce numbers that already exist and are already reported."""
    rows = []
    for pt_path in paths:
        algo, _, seed = parse_run_name(pt_path.stem)
        csv = Path("results/eval") / f"{pt_path.stem}{suffix_for(algo, suffix)}.csv"
        if not csv.exists():
            continue
        rows.append({"algo": algo, "arm": "reference", "n_gnb": base_n, "seed": seed,
                     "status": "OK", "reason": "", **summarise(pd.read_csv(csv), base_n)})
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", type=str, default="results/logs/*_v4_seed*.pt",
                   help="v4 wave only; the old default also swept the v3 checkpoints")
    p.add_argument("--n-gnb", type=str, default="10,20")
    p.add_argument("--arms", type=str, default=",".join(ARMS))
    p.add_argument("--episodes", type=int, default=150,
                   help="150 matches the wave's held-out protocol; use 5 for a cost pilot")
    p.add_argument("--out-dir", type=str, default="results/eval_zeroshot_v4")
    p.add_argument("--out", type=str, default="results/ZEROSHOT_v4.md")
    p.add_argument("--stochastic", action="store_true",
                   help="P3 primary readout. Without it this writes the greedy pass, "
                        "which is reported alongside but never gates.")
    p.add_argument("--readout", choices=["greedy", "stochastic", "primary"], default=None,
                   help="'primary' is the gating readout per family: sampled for PPO (P3), "
                        "argmax for DQN (determination 2026-08-16). Only meaningful with "
                        "--summary-only, since a run produces one readout at a time.")
    p.add_argument("--summary-only", action="store_true",
                   help="rebuild the report from eval CSVs already on disk, running nothing. "
                        "Needed because checkpoints trained after a grid started (the k-NN "
                        "baseline) would otherwise force a full re-run to appear in one table.")
    args = p.parse_args()

    if args.readout and not args.summary_only:
        raise SystemExit("--readout only applies to --summary-only: a single run writes one "
                         "readout, so mixing them requires reading files already on disk")
    suffix = ({"greedy": "_eval", "stochastic": "_eval_stoch", "primary": "primary"}[args.readout]
              if args.readout else ("_eval_stoch" if args.stochastic else "_eval"))
    base_n, base_area, floor_mode = base_env()
    paths = [Path(pt) for pt in sorted(glob.glob(args.checkpoints)) if "_floor" not in Path(pt).stem]
    n_gnb_list = [int(x) for x in args.n_gnb.split(",")]
    arms = [a for a in args.arms.split(",") if a]
    for arm in arms:
        if arm not in ARMS:
            raise SystemExit(f"unknown arm {arm!r}; pick from {ARMS}")

    rows = reference_rows(paths, base_n, suffix)
    for arm in arms:
        for n_gnb in n_gnb_list:
            out_dir = Path(args.out_dir) / arm / f"ngnb{n_gnb}"
            if args.summary_only:
                print(f"[arm={arm} n_gnb={n_gnb}] summary-only, reading {out_dir}")
            else:
                config_path = arm_config(arm, n_gnb, base_n, base_area, floor_mode)
                print(f"[arm={arm} n_gnb={n_gnb}] config={config_path.name}")
            for pt_path in paths:
                algo, _, seed = parse_run_name(pt_path.stem)
                csv_path = out_dir / f"{pt_path.stem}{suffix_for(algo, suffix)}.csv"
                locked = locked_obs_dim(pt_path)
                if locked is not None and n_gnb != base_n:
                    # Decided from the architecture, never from what is on disk. The
                    # epsilon=1.0 fault produced 150 rows of "results" for central-dqn at
                    # n_gnb=10/20 -- valid-looking output from a network that was never
                    # called (results/quarantine_eps1.0/README.md). Reading the filesystem
                    # to decide this question is what let that pass as data.
                    rows.append({"algo": algo, "arm": arm, "n_gnb": n_gnb, "seed": seed,
                                 "status": "CANNOT_RUN",
                                 "reason": f"obs_dim mismatch: trained {locked}, "
                                           f"required {n_gnb * OBS_FEATURES}"})
                    continue
                if not args.summary_only:
                    try:
                        evaluate_run(pt_path, args.episodes, str(config_path), out_dir,
                                     greedy=not args.stochastic, suffix=suffix)
                    except RuntimeError as e:
                        print(f"[{pt_path.stem}] n_gnb={n_gnb}: CANNOT_RUN: {e}")
                if not csv_path.exists():
                    # No output file means the architecture could not be instantiated at
                    # this topology -- the structural CANNOT_RUN result, not missing data.
                    rows.append({"algo": algo, "arm": arm, "n_gnb": n_gnb, "seed": seed,
                                 "status": "CANNOT_RUN",
                                 "reason": "no eval output produced at this topology"})
                    continue
                try:
                    edf = pd.read_csv(csv_path)
                except pd.errors.EmptyDataError:
                    edf = pd.DataFrame()
                if edf.empty:
                    # A run still in flight (--summary-only against a grid that is mid-write)
                    # is not the same thing as an architecture that cannot run. Kept as its
                    # own status so a half-finished grid is never read as a structural
                    # failure -- that would turn a timing accident into a paper claim.
                    print(f"[{pt_path.stem}] arm={arm} n_gnb={n_gnb}: INCOMPLETE (no rows yet)")
                    rows.append({"algo": algo, "arm": arm, "n_gnb": n_gnb, "seed": seed,
                                 "status": "INCOMPLETE",
                                 "reason": "eval file present but empty — run in flight"})
                    continue
                rows.append({"algo": algo, "arm": arm, "n_gnb": n_gnb, "seed": seed,
                             "status": "OK", "reason": "", **summarise(edf, n_gnb)})

    df = pd.DataFrame(rows)
    out_dir_root = Path(args.out_dir)
    out_dir_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir_root / f"zeroshot_summary{suffix}.csv", index=False)

    readout = readout_label(suffix)
    lines = [
        "# Zero-shot topology transfer — wave v4\n",
        f"Readout: `{readout}`. Episodes per checkpoint: {args.episodes}. "
        f"Trained at n_gnb={base_n}, area_size={base_area:.0f} m, floor.mode=`{floor_mode}` "
        f"(read from the config, not assumed).\n",
        "`central-dqn` / `central-ppo` have `obs_dim = n_gnb * 8` fixed at training time, so "
        "CANNOT_RUN outside n_gnb=5 is a structural property of the architecture and is "
        "reported as a result, not as missing data. Every such row carries its reason, so "
        "**cannot be run** is never confused with **was not attempted**; the status is "
        "decided from the architecture, not from which files happen to exist.\n",
        "Aggregate throughput grows with cell count no matter what the policy does, so "
        "**throughput per gNB** is the column to read; the aggregate is printed beside it. "
        "`retention` is per-gNB throughput relative to the same checkpoints at n_gnb="
        f"{base_n}.\n",
        "Two arms are reported in full (integritas #3). `fixed-area` keeps area_size at "
        f"{base_area:.0f} m so raising n_gnb also raises coupling strength; `const-density` "
        "scales area_size as sqrt(n/5) so density matches training. The env docstring "
        "(envs/network_slicing_env.py) is explicit that the first arm confounds agent count "
        "with coupling strength — it is kept because the v3 report used it.\n",
    ]

    ok = df[df.status == "OK"]
    ref = ok[ok.arm == "reference"].groupby("algo")["throughput_per_gnb_mbps"].mean()
    for arm in ["reference", *arms]:
        sub = df[df.arm == arm]
        if sub.empty:
            continue
        lines += [
            f"\n## {arm}\n",
            "| algo | n_gnb | status | reason | thr/gNB (Mbps) | retention | thr agg (Mbps) "
            "| sla_satisfaction_pct | embb_p5_mbps | n seeds |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for (algo, n_gnb), grp in sub.groupby(["algo", "n_gnb"]):
            good = grp[grp.status == "OK"]
            if good.empty:
                # Report the status that actually occurred: CANNOT_RUN is a finding,
                # INCOMPLETE just means the grid had not finished when this was written.
                status = "/".join(sorted(set(grp.status)))
                why = "; ".join(sorted({r for r in grp.reason if r}))
                lines.append(f"| `{algo}` | {n_gnb} | {status} | {why} "
                             "| - | - | - | - | - | 0 |")
                continue
            per_gnb = good.throughput_per_gnb_mbps.mean()
            base = ref.get(algo)
            ret = f"{per_gnb / base:.3f}" if base else "-"
            lines.append(
                f"| `{algo}` | {n_gnb} | OK | | {per_gnb:.4f} | {ret} "
                f"| {good.throughput_mbps.mean():.4f} | {good.sla_satisfaction_pct.mean():.4f} "
                f"| {good.embb_p5_mbps.mean():.6f} | {len(good)} |"
            )

    lines.append(
        "\nCI and per-family comparison are not computed here. Point "
        "`scripts/rliable_report.py --eval-dir results/eval_zeroshot_v4/<arm>/ngnb<N>` and "
        "`scripts/stability_report.py` at these directories instead — they already do IQM + "
        "stratified bootstrap per budget family (C3) and Wilson collapse rate."
    )

    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
