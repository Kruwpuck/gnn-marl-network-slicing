# Quarantine — the ε=1.0 DQN readout (2026-08-17)

**Nothing in this directory may be used in any report, table, or claim.** The files are
kept, not deleted, because the fault they document is itself a methodological finding
(`handoff/paper_structure.md` §2): a wrong readout protocol produces rows that look
perfectly valid.

## What is wrong with these files

Every non-greedy DQN reading in wave v4 was **uniform random action selection**, not the
trained policy.

Root cause: `epsilon` is not part of `state_dict` and `_save_model()` did not store it, so
all 20 DQN v4 checkpoints carry no epsilon. `load_agent` then built a fresh agent sitting at
the YAML start value `epsilon = 1.0`, and `DQNAgent.act` / `MLPDQNAgent.act` return
`np.random.randint(...)` whenever `not greedy and rand < epsilon`. With epsilon at 1.0 that
is every step.

Measured, not inferred: action agreement between the greedy and the "stochastic" reading was
0.097–0.100 for `central-dqn`, `idqn`, `gnn-madqn_gat`, against 1/11 = 0.091 for pure chance.

## How it survived undetected

Aggregate KPIs barely moved — `central-dqn` timely throughput 67.50 greedy / 64.66 at ε=1.0 /
67.49 at ε=0.05 — so nothing in the headline tables looked wrong. The damage was confined to
the cell-edge metric, where it flipped a verdict: `gnn-madqn_gat` seed42 `embb_p5_mbps` is
1.778019 greedy and 1.679319 at ε=0.05, against the 0.000006 that was being reported.

It was found only because `central-dqn` produced 150 rows of data at n_gnb=10 in the
zero-shot grid, where `obs_dim = n_gnb * 8` makes it structurally impossible to run. The
network was never called, so nothing raised.

## Contents

| path | what | why void |
|---|---|---|
| `eval/` | 20 files, `results/eval/*dqn*_v4_seed*_eval_stoch.csv`, n_gnb=5 | ε=1.0 random actions |
| `eval_zeroshot_v4/<arm>/ngnb<N>/` | 20 files, `central-dqn_v4_seed*_eval_stoch.csv` | ε=1.0 random actions **and** an architecture that cannot run at this topology at all |

The zero-shot half is doubly void: those rows exist only because the random-action path
bypassed the network that would have raised a shape error.

## What replaced them

- Readout protocol for the DQN family fixed by human decision 2026-08-16 (`handoff/goal1.md`,
  "Penetapan protokol pembacaan keluarga DQN"), after a pre-registered diagnostic: argmax is
  primary for DQN, ε=0.05 (`epsilon_min`) is the non-greedy reading reported beside it.
- Valid ε=0.05 data: `results/eval_dqn_eps005/` (20 files, 150 episodes).
- Reports recomputed under the per-family primary readout: `results/RLIABLE_v4_primary.md`,
  `results/STABILITY_v4_primary.md`, `results/GATE_B_v4_primary.md`,
  `results/ZEROSHOT_v4_primary.md`.

## Root-cause fixes already in the code

- `scripts/evaluate_checkpoints.py::load_agent` sets epsilon from `agent.dqn.epsilon_min`,
  never 1.0, and takes `--dqn-epsilon` to override it explicitly.
- `_save_model` in both training scripts now stores `epsilon`, so future checkpoints cannot
  repeat this.
- `scripts/zeroshot_eval.py` forces `CANNOT_RUN` for `central-*` at any topology other than
  the training one, whatever files happen to be on disk — so a quarantined file (or a future
  one like it) can never be read as a valid row again.
