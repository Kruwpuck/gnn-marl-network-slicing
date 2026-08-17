"""Gate B is frozen at the pre-registered five seeds, and refuses unequal n.

The PPO family was extended to 20 seeds on 2026-08-17 while the DQN family stayed at 5
(goal1.md, "Eksekusi perluasan seed keluarga PPO"). B1-B4 are ranges of means *across* the
8 algorithms, so computing them over unequal n would move a pre-registered range for a
statistical reason rather than because the task changed.

These tests exist because the guard cannot fire against the real eval directory yet -- the
extra seeds have no eval CSVs while training runs -- and a branch that has never executed
is not a safeguard.
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.gate_b_report import GATE_SEEDS, PREREGISTERED

KPI_VALUES = {"timely_throughput_mbps": 60.0, "sla_satisfaction_pct": 80.0,
              "urllc_delay_p99": 8.0, "embb_p5_mbps": 1.0, "jains_fairness": 0.9,
              "sla_violation_pct": 20.0}


def write_eval(eval_dir: Path, algo: str, seed: int, bump: float = 0.0) -> None:
    """One eval CSV in the layout load_eval_dir expects, at the primary readout per family."""
    suffix = "_eval" if "dqn" in algo else "_eval_stoch"
    row = {k: v + bump for k, v in KPI_VALUES.items()}
    pd.DataFrame([row, row]).to_csv(eval_dir / f"{algo}_v4_seed{seed}{suffix}.csv", index=False)


def run_gate_b(eval_dir: Path, out: Path, seeds: str | None = None):
    cmd = [sys.executable, "scripts/gate_b_report.py", "--eval-dir", str(eval_dir),
           "--tag", "_v4", "--readout", "primary", "--out", str(out)]
    if seeds is not None:
        cmd += ["--seeds", seeds]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)


def populate(eval_dir: Path) -> None:
    for algo in PREREGISTERED:
        for seed in GATE_SEEDS:
            write_eval(eval_dir, algo, seed)


def test_default_seeds_are_the_preregistered_five():
    assert GATE_SEEDS == [42, 43, 44, 45, 46]


def test_extra_seeds_are_excluded_not_averaged_in(tmp_path):
    """A 20-seed PPO family must not drag the range: seeds outside the freeze are dropped."""
    populate(tmp_path)
    # An extension seed with a wildly different value: if it were included, B1 would move.
    write_eval(tmp_path, "ippo", 47, bump=500.0)

    out = tmp_path / "gate_b.md"
    res = run_gate_b(tmp_path, out)
    assert res.returncode == 0, res.stderr
    assert "B1=0.00%" in res.stdout, res.stdout          # all algos identical -> zero range
    assert "Seed freeze" in out.read_text(encoding="utf-8")


def test_unequal_seed_counts_are_refused(tmp_path):
    """Explicitly widening the seed set to an unequal one must fail, not quietly compute."""
    populate(tmp_path)
    write_eval(tmp_path, "ippo", 47)   # only ippo has the extra seed

    res = run_gate_b(tmp_path, tmp_path / "gate_b.md", seeds="42,43,44,45,46,47")
    assert res.returncode != 0
    assert "unequal seed counts" in (res.stderr + res.stdout)


def test_posthoc_baseline_stays_out_of_the_gate(tmp_path):
    """Gate B is a pre-registration; a baseline added after seeing results cannot widen it."""
    populate(tmp_path)
    for seed in GATE_SEEDS:
        write_eval(tmp_path, "mlp-knn-ppo", seed, bump=500.0)

    res = run_gate_b(tmp_path, tmp_path / "gate_b.md")
    assert res.returncode == 0, res.stderr
    assert "B1=0.00%" in res.stdout, res.stdout
