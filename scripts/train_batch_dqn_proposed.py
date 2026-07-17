"""Batch 2/3 — proposed GNN-MADQN training jobs (resumable).

  python scripts/train_batch_dqn_proposed.py [--steps 200000] [--seed 42]
"""
from __future__ import annotations
import argparse, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_jobs(jobs):
    stdout_dir = ROOT / "results" / "logs" / "stdout"
    stdout_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    for name, cmd in jobs:
        log = stdout_dir / f"{name}.log"
        print(f"[{time.strftime('%H:%M:%S')}] === {name} ===  -> {log}", flush=True)
        with open(log, "w", encoding="utf-8") as f:
            r = subprocess.run([sys.executable, *cmd], cwd=str(ROOT),
                               stdout=f, stderr=subprocess.STDOUT)
        print(f"[{time.strftime('%H:%M:%S')}] [{name}] exit {r.returncode}", flush=True)
        if r.returncode != 0:
            failures.append(name)
    print(f"\nDone. {'FAILURES: ' + ', '.join(failures) if failures else 'all jobs ok'}")
    return 1 if failures else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--tag", type=str, default="")
    args = p.parse_args()
    common = ["--steps", str(args.steps), "--seed", str(args.seed),
              "--resume", "--ckpt-interval", "25000",
              *(["--config", args.config] if args.config else []),
              *(["--tag", args.tag] if args.tag else [])]
    jobs = [
        ("gnn-madqn_gat",  ["training/train_proposed.py", "--algo", "gnn-madqn", "--backbone", "gat", *common]),
        ("gnn-madqn_sage", ["training/train_proposed.py", "--algo", "gnn-madqn", "--backbone", "sage", *common]),
    ]
    sys.exit(run_jobs(jobs))
