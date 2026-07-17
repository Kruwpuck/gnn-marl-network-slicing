"""
Parallel orchestrator for a training wave: N seeds x 8 algorithms (main wave, floor
dynamic) or N seeds x 4 algorithms (ablation, floor none/static). Every job gets an
*identical* config except floor.mode — generated once into configs/generated/ — so
all algorithms in a wave see the same environment.

Runs up to --max-parallel jobs concurrently via subprocess (GPU is small relative to
model size here; bottleneck is per-step Python overhead, not GPU memory).

Usage:
  python scripts/run_wave.py --seeds 42,43,44,45,46 --floor-mode dynamic --max-parallel 6
  python scripts/run_wave.py --seeds 42,43,44,45,46 --floor-mode none \
      --algos gnn-madqn_sage,gnn-mappo_gat,idqn,central-ppo --max-parallel 6
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import time
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG = ROOT / "configs" / "experiment_config.yaml"
GEN_DIR = ROOT / "configs" / "generated"

DQN_STEPS = 200_000
PPO_STEPS = 1_000_000

# algo -> (script, kind, extra_args_fn(backbone))
PROPOSED_DQN = ["gnn-madqn_gat", "gnn-madqn_sage"]
PROPOSED_PPO = ["gnn-mappo_gat", "gnn-mappo_sage"]
BASELINE_DQN = ["idqn", "central-dqn"]
BASELINE_PPO = ["ippo", "central-ppo"]
ALL_ALGOS = PROPOSED_DQN + PROPOSED_PPO + BASELINE_DQN + BASELINE_PPO


def make_variant_config(floor_mode: str) -> Path:
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(BASE_CONFIG.read_text())
    cfg.setdefault("floor", {})["mode"] = floor_mode
    out_path = GEN_DIR / f"floor_{floor_mode}.yaml"
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out_path


def job_cmd(algo: str, seed: int, config_path: Path, tag: str) -> tuple[str, list[str]]:
    common = ["--seed", str(seed), "--resume", "--ckpt-interval", "25000",
              "--config", str(config_path), "--tag", tag]
    if algo in PROPOSED_DQN:
        backbone = algo.split("_")[1]
        return (f"{algo}{tag}_seed{seed}",
                ["training/train_proposed.py", "--algo", "gnn-madqn", "--backbone", backbone,
                 "--steps", str(DQN_STEPS), *common])
    if algo in PROPOSED_PPO:
        backbone = algo.split("_")[1]
        return (f"{algo}{tag}_seed{seed}",
                ["training/train_proposed.py", "--algo", "gnn-mappo", "--backbone", backbone,
                 "--steps", str(PPO_STEPS), *common])
    if algo in BASELINE_DQN:
        return (f"{algo}{tag}_seed{seed}",
                ["training/train_baselines.py", "--algo", algo, "--steps", str(DQN_STEPS), *common])
    if algo in BASELINE_PPO:
        return (f"{algo}{tag}_seed{seed}",
                ["training/train_baselines.py", "--algo", algo, "--steps", str(PPO_STEPS), *common])
    raise ValueError(f"unknown algo {algo}")


def run_one(name: str, cmd: list[str], stdout_dir: Path) -> tuple[str, int]:
    log = stdout_dir / f"{name}.log"
    with open(log, "w", encoding="utf-8") as f:
        r = subprocess.run([sys.executable, *cmd], cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT)
    return name, r.returncode


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=str, required=True, help="comma-separated, e.g. 42,43,44,45,46")
    p.add_argument("--floor-mode", type=str, default="dynamic", choices=["none", "static", "dynamic"])
    p.add_argument("--algos", type=str, default=None,
                   help="comma-separated subset of algos (default: all 8, for ablation pass 4)")
    p.add_argument("--tag", type=str, default=None,
                   help="override run tag (default: derived from floor-mode, e.g. _floornone; "
                        "'' for the main dynamic wave to keep original filenames)")
    p.add_argument("--max-parallel", type=int, default=6)
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    algos = args.algos.split(",") if args.algos else ALL_ALGOS
    tag = args.tag if args.tag is not None else ("" if args.floor_mode == "dynamic" else f"_floor{args.floor_mode}")

    config_path = make_variant_config(args.floor_mode)
    stdout_dir = ROOT / "results" / "logs" / "stdout"
    stdout_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for seed in seeds:
        for algo in algos:
            name, args_list = job_cmd(algo, seed, config_path, tag)
            jobs.append((name, args_list))

    print(f"Wave: floor={args.floor_mode} tag='{tag}' seeds={seeds} algos={algos} "
          f"-> {len(jobs)} jobs, max_parallel={args.max_parallel}")
    print(f"config: {config_path}")

    t0 = time.time()
    failures = []
    with ThreadPoolExecutor(max_workers=args.max_parallel) as ex:
        futs = {ex.submit(run_one, name, cmd, stdout_dir): name for name, cmd in jobs}
        for fut in as_completed(futs):
            name, rc = fut.result()
            status = "OK" if rc == 0 else f"FAIL(rc={rc})"
            print(f"[{time.strftime('%H:%M:%S')}] {name}: {status}  "
                  f"(elapsed {(time.time()-t0)/3600:.2f}h)", flush=True)
            if rc != 0:
                failures.append(name)

    print(f"\nWave done in {(time.time()-t0)/3600:.2f}h. "
          f"{'FAILURES: ' + ', '.join(failures) if failures else 'all jobs ok'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
