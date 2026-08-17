"""
Why Gate B3 fails: is `urllc_delay_p99` censored by the deadline drop?

B3 asks for >= 2 ms of spread in `urllc_delay_p99` across the 8 algorithms; v4
measured 1.01 ms. This script tests the mechanistic explanation, it does NOT
re-decide the gate -- the 2 ms threshold stands and B3 stays FAILED.

Hypothesis (envs/network_slicing_env.py:333): a packet whose age exceeds
`slices.urllc.max_delay_ms` is popped from the FIFO and counted as a
deadline drop, so it never contributes a delay sample. Delivered delay is
therefore right-censored at the deadline *by construction* (the env docstring
says as much), and additionally quantised: delay = (age in slots) *
`slot_duration_ms`, so with a 10 ms deadline and a 1 ms slot the support is the
11 points {0, 1, ..., 10}. A statistic that can only land on 11 lattice points,
with its upper tail amputated, has little room to separate algorithms -- and
the p99 of any policy whose queue backs up is pinned at the top point.

Measured per algorithm over the 8 pre-registered algorithms (all seeds), under the
per-family primary readout: sampled for PPO (P3), argmax for DQN (determination
2026-08-16). The earlier pass read DQN at epsilon=1.0, which was random actions.
  p99          mean per-episode p99, ms  (matches results/eval/*_eval_stoch.csv)
  mass@dl      % of delivered packets sitting exactly on the deadline bin
  mass>=dl-1   % within one slot of the deadline
  late:ovf     deadline-drop : overflow-drop ratio (Gate A3 context)
  censored     % of resolved packets that never produced a delay sample at all

Usage:
  python scripts/delay_censoring_diag.py --runs "results/logs/*_v4_seed*.pt" \
      --episodes 5 --out results/B3_DELAY_CENSORING.md
"""
from __future__ import annotations
import argparse
import glob
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.network_slicing_env import NetworkSlicingEnv
from scripts.evaluate_checkpoints import EVAL_SEED_BASE, load_agent, select_actions
from scripts.gate_b_report import PREREGISTERED
from scripts.rliable_report import parse_run_name, primary_suffix

READOUTS = ("greedy", "stochastic", "primary")


def is_greedy(algo: str, readout: str) -> bool:
    """Whether this algorithm is read by argmax under the chosen readout.

    'primary' defers to rliable_report.primary_suffix so the family rule lives in one
    place: sampled for PPO (P3), argmax for DQN (human determination 2026-08-16). A second
    copy of `"dqn" in algo` here would be free to drift out of step with the gate numbers.
    """
    return readout == "greedy" or (readout == "primary" and primary_suffix(algo) == "_eval")


def gate_b_value(path: Path, row: str, unit: str) -> str:
    """Pull one gate number out of the generated Gate B report rather than retyping it.

    B2 moved from 8.70 pp to 11.98 pp when the DQN readout was corrected; a hard-coded copy
    in this file's prose would have gone on quoting the old one (docs/HANDOVER.md 11).
    """
    text = path.read_text(encoding="utf-8")
    m = re.search(rf"\|\s*{row}\s*\|.*?\|\s*([\d.]+)\s*{unit}\s*\|", text)
    if not m:
        raise SystemExit(f"could not read {row} from {path} — regenerate it with gate_b_report.py")
    return f"{float(m.group(1)):.2f} {unit}"


def measure(pt_path: Path, episodes: int, config_path: str | None, readout: str) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent, _, kind = load_agent(pt_path, device)
    # from the run name, not ckpt["algo"]: the checkpoint field drops the backbone, which
    # would merge gnn-madqn_gat with gnn-madqn_sage and hide exactly what B3 is about.
    algo, _, _ = parse_run_name(pt_path.stem)
    greedy = is_greedy(algo, readout)
    env = NetworkSlicingEnv(config_path=config_path)
    env.cmdp_enabled = False  # same as evaluate_checkpoints: no lambda drift during readout

    n_bins = int(round(env.urllc_max_delay_ms / env.slot_ms)) + 1  # {0 .. deadline} slots
    hist = np.zeros(n_bins, dtype=np.int64)
    p99_per_ep: list[float] = []
    late = ovf = 0

    for ep in range(episodes):
        torch.manual_seed(EVAL_SEED_BASE + ep)  # same reproducibility fix as the eval script
        obs, info = env.reset(seed=EVAL_SEED_BASE + ep)
        done = False
        ep_delays: list[float] = []
        while not done:
            actions = select_actions(agent, kind, obs, info, env, greedy=greedy)
            obs, _, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            for gnb_delays in info["urllc_delivered_delays"]:
                ep_delays.extend(gnb_delays)
            late += int(np.sum(info["urllc_dropped_late"]))
            ovf += int(np.sum(info["urllc_dropped_overflow"]))
        arr = np.asarray(ep_delays) if ep_delays else np.array([0.0])
        p99_per_ep.append(float(np.percentile(arr, 99)))
        idx = np.clip(np.round(arr / env.slot_ms).astype(int), 0, n_bins - 1)
        hist += np.bincount(idx, minlength=n_bins)

    env.close()
    delivered = int(hist.sum())
    resolved = delivered + late + ovf
    return {
        "algo": algo, "run": pt_path.stem, "greedy": greedy,
        "p99": float(np.mean(p99_per_ep)),
        "hist": hist, "delivered": delivered,
        "late": late, "ovf": ovf, "resolved": resolved,
        "deadline_ms": env.urllc_max_delay_ms, "slot_ms": env.slot_ms, "n_bins": n_bins,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=str, required=True, help="glob for .pt checkpoints")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--readout", choices=READOUTS, default="primary",
                   help="'primary' is the gating readout per family: sampled for PPO (P3), "
                        "argmax for DQN (2026-08-16). The old hard-coded behaviour was "
                        "'stochastic', which for DQN meant epsilon=1.0 random actions.")
    p.add_argument("--gate-b", type=str, default="results/GATE_B_v4_primary.md")
    p.add_argument("--out", type=str, default="results/B3_DELAY_CENSORING.md")
    args = p.parse_args()

    per_algo: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(glob.glob(args.runs)):
        algo, _, _ = parse_run_name(Path(path).stem)
        if algo not in PREREGISTERED:
            # The spread reported here is Gate B3's spread, measured over the same 8
            # pre-registered algorithms. Adding a baseline introduced later (mlp-knn-ppo)
            # would change the range the gate is about.
            continue
        r = measure(Path(path), args.episodes, args.config, args.readout)
        per_algo[r["algo"]].append(r)
        print(f"  {r['run']}: p99={r['p99']:.3f} delivered={r['delivered']} greedy={r['greedy']}")

    if not per_algo:
        raise SystemExit(f"no checkpoints matched {args.runs}")

    any_r = next(iter(per_algo.values()))[0]
    deadline, slot, n_bins = any_r["deadline_ms"], any_r["slot_ms"], any_r["n_bins"]

    gate_b = Path(args.gate_b)
    b2, b3 = gate_b_value(gate_b, "B2", "pp"), gate_b_value(gate_b, "B3", "ms")
    readout_note = {
        "primary": "primary per family — sampled for PPO (P3), argmax for DQN (2026-08-16)",
        "stochastic": "non-greedy — sampled for PPO, epsilon=0.05 for DQN",
        "greedy": "greedy (reported, never gates)",
    }[args.readout]

    lines = [
        "# B3 diagnostic — is `urllc_delay_p99` censored by the deadline drop?\n",
        f"Readout: {readout_note}. {args.episodes} held-out episodes per checkpoint, "
        f"seeds from `EVAL_SEED_BASE={EVAL_SEED_BASE}`.\n",
        "The previous version of this file read the DQN family at epsilon=1.0, i.e. uniform "
        "random actions rather than the policy; those four rows were void and have been "
        "recomputed here (`results/quarantine_eps1.0/README.md`).\n",
        f"**This does not change Gate B3.** The threshold stays >= 2 ms and B3 stays FAILED "
        f"(measured range {b3}, from `{gate_b}`). This file only explains the mechanism.\n",
        f"\nDeadline `slices.urllc.max_delay_ms` = {deadline} ms, slot = {slot} ms, so a delivered "
        f"packet's delay can only take **{n_bins} values** ({{0, {slot:g}, ..., {deadline:g}}} ms) and "
        f"nothing above {deadline:g} ms can ever be observed: `envs/network_slicing_env.py:333` pops "
        "over-age packets before service, so they are counted as drops, not as slow deliveries.\n",
        f"\n| algo | p99 (ms) | mass @ {deadline:g} ms | mass >= {deadline - slot:g} ms | "
        "delivered | censored (drops) | late:ovf |",
        "|---|---|---|---|---|---|---|",
    ]

    for algo in sorted(per_algo):
        runs = per_algo[algo]
        hist = np.sum([r["hist"] for r in runs], axis=0)
        delivered = int(hist.sum())
        late = sum(r["late"] for r in runs)
        ovf = sum(r["ovf"] for r in runs)
        resolved = delivered + late + ovf
        p99 = float(np.mean([r["p99"] for r in runs]))
        mass_dl = 100.0 * hist[-1] / max(delivered, 1)
        mass_near = 100.0 * hist[-2:].sum() / max(delivered, 1)
        censored = 100.0 * (late + ovf) / max(resolved, 1)
        ratio = f"{late / ovf:.1f}:1" if ovf else f"{late}:0"
        lines.append(f"| `{algo}` | {p99:.3f} | {mass_dl:.2f}% | {mass_near:.2f}% | "
                     f"{delivered:,} | {censored:.2f}% | {ratio} |")

    p99s = [float(np.mean([r["p99"] for r in per_algo[a]])) for a in per_algo]
    lines += [
        f"\nSpread of p99 across algorithms: **{max(p99s) - min(p99s):.3f} ms** "
        f"(min {min(p99s):.3f}, max {max(p99s):.3f}). One slot = {slot:g} ms, so B3's 2 ms "
        f"threshold asks for at least **{2.0 / slot:.0f} lattice steps** of separation in a "
        f"statistic whose whole support is {n_bins} steps wide and whose upper end is a hard wall.\n",
        "\nRead the `censored` column together with the p99 column: the packets that would have "
        "formed the discriminating tail are exactly the ones removed from the sample. Two policies "
        "with very different queueing behaviour report near-identical p99 while differing on "
        f"`sla_satisfaction_pct` (which counts the drops) -- that is B2 passing at {b2} while "
        f"B3 fails at {b3}, and it is one phenomenon seen through two metrics.\n",
        "\nMethodological consequence for the paper: at this operating point `urllc_delay_p99` "
        "of delivered packets is not a discriminating KPI, and the honest latency statement lives "
        "in the drop/SLA metrics instead. Reported, not claimed (`handoff/goal1.md`, scoping "
        "decision 2026-08-15).\n",
    ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
