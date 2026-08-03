"""
Treatment-identity test (pre-registration criterion #4, docs/journey/06).

The env must not special-case which algorithm is driving it. This test
feeds two independently constructed NetworkSlicingEnv instances the exact
same seed and the exact same action sequence, then asserts every
constraint-relevant output (floor_applied f_min, lam, delta, per-gNB
violation_rate) is bit-identical step by step. If it isn't, the
CMDP/floor mechanism is leaking some hidden state that could structurally
favor one algorithm over another.

Actions are synthetic (fixed RNG, not from a trained policy) — this test
is about environment determinism/fairness, not about any agent's behavior.

Usage:
  python scripts/test_treatment_identity.py --seed 42 --floor-mode dynamic --steps 200
  python scripts/test_treatment_identity.py --floor-mode none
  python scripts/test_treatment_identity.py --floor-mode static
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.network_slicing_env import NetworkSlicingEnv


def run_episode(seed: int, floor_mode: str, steps: int, config_path: str, action_rng_seed: int):
    env = NetworkSlicingEnv(config_path=config_path, action_type="discrete")
    env.floor_mode = floor_mode
    obs, _ = env.reset(seed=seed)

    act_rng = np.random.default_rng(action_rng_seed)
    trace = []
    for _ in range(steps):
        actions = act_rng.integers(0, env.n_tiers, size=env.n_gnb)
        backlog_bits = np.array(
            [sum(e[1] for e in env._urllc_fifo[i]) for i in range(env.n_gnb)], dtype=np.float64
        )
        f_min_pre = env._compute_floor(backlog_bits)  # floor computed on pre-step backlog, for reference
        obs, reward, terminated, truncated, info = env.step(actions)
        trace.append({
            "f_min": f_min_pre.copy(),
            "lam": info["lam"],
            "delta": env.delta,
            "violation_rate": np.asarray(info["urllc_violation_rate"]).copy(),
            "reward": reward,
        })
        if terminated or truncated:
            break
    return trace


def compare(trace_a, trace_b) -> list[str]:
    mismatches = []
    if len(trace_a) != len(trace_b):
        mismatches.append(f"step count differs: {len(trace_a)} vs {len(trace_b)}")
        return mismatches
    for t, (a, b) in enumerate(zip(trace_a, trace_b)):
        if not np.array_equal(a["f_min"], b["f_min"]):
            mismatches.append(f"step {t}: f_min differs {a['f_min']} vs {b['f_min']}")
        if a["lam"] != b["lam"]:
            mismatches.append(f"step {t}: lam differs {a['lam']} vs {b['lam']}")
        if a["delta"] != b["delta"]:
            mismatches.append(f"step {t}: delta differs {a['delta']} vs {b['delta']}")
        if not np.array_equal(a["violation_rate"], b["violation_rate"]):
            mismatches.append(
                f"step {t}: violation_rate differs {a['violation_rate']} vs {b['violation_rate']}"
            )
        if a["reward"] != b["reward"]:
            mismatches.append(f"step {t}: reward differs {a['reward']} vs {b['reward']}")
    return mismatches


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--floor-mode", type=str, default="dynamic", choices=["none", "static", "dynamic"])
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--config", type=str, default="configs/experiment_config.yaml")
    args = p.parse_args()

    print(f"Treatment-identity test: seed={args.seed} floor_mode={args.floor_mode} steps={args.steps}")

    # "proposed" and "baseline" runs: two independent env instances, identical
    # seed + identical action sequence (same action_rng_seed) simulate two
    # different algorithms picking the exact same actions by coincidence.
    trace_proposed = run_episode(args.seed, args.floor_mode, args.steps, args.config, action_rng_seed=1234)
    trace_baseline = run_episode(args.seed, args.floor_mode, args.steps, args.config, action_rng_seed=1234)

    mismatches = compare(trace_proposed, trace_baseline)
    if mismatches:
        print(f"FAIL: {len(mismatches)} mismatch(es) found — env is NOT identical across identical action sequences")
        for m in mismatches[:20]:
            print(" ", m)
        sys.exit(1)

    print(f"PASS: {len(trace_proposed)} steps, (floor_applied, lam, delta, violation_rate, reward) "
          f"identical bit-for-bit between two independent env instances given identical actions.")
    sys.exit(0)


if __name__ == "__main__":
    main()
