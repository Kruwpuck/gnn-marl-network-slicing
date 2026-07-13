"""Batch 1/3 — all PPO training jobs (resumable).

Runs sequentially, each job in its own process (a crash can't kill the batch).
Parallel-safe vs the DQN batches (distinct log/checkpoint names), so this can run
concurrently with train_batch_dqn_proposed.py / train_batch_dqn_baseline.py.

  python scripts/train_batch_ppo.py [--steps 1000000] [--seed 42]
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
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    common = ["--steps", str(args.steps), "--seed", str(args.seed),
              "--resume", "--ckpt-interval", "25000"]
    jobs = [
        ("gnn-mappo_gat",  ["training/train_proposed.py", "--algo", "gnn-mappo", "--backbone", "gat", *common]),
        ("gnn-mappo_sage", ["training/train_proposed.py", "--algo", "gnn-mappo", "--backbone", "sage", *common]),
        ("ippo",           ["training/train_baselines.py", "--algo", "ippo", *common]),
        ("central-ppo",    ["training/train_baselines.py", "--algo", "central-ppo", *common]),
    ]
    sys.exit(run_jobs(jobs))
