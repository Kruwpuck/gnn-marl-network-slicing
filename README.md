# gnn-marl-network-slicing

Graph-neural-network multi-agent reinforcement learning for 5G RAN slicing, evaluated
against centralised and per-agent baselines under a CMDP constraint on URLLC violation.

> **Status: NOT DONE.** `handoff/goal1.md` declares the work finished only when every
> gate passes. Gate B3 fails, C2 is partial, and C4 fails for the DQN family. This is a
> recorded outcome, not a work-in-progress notice -- see `results/GATE_C.md`. Nothing in
> `results/` should be read as a settled claim without reading the gate verdicts first.

## Layout

| Path | What it is |
|---|---|
| `envs/` | the slicing environment, channel model, PettingZoo wrapper |
| `agents/` | DQN and PPO agents, MLP baselines, hyperparameter resolution |
| `gnn/` | interchangeable backbones (GAT, SAGE, GCN) behind one interface |
| `traffic/` | Poisson and MMPP arrival processes |
| `training/` | training loops, replay/rollout buffers, metrics logging |
| `evaluation/` | convergence, network-performance, and zero-shot evaluators |
| `ablation/` | backbone ablation driver |
| `scripts/` | everything run from the command line: waves, evaluation, reports, diagnostics |
| `tests/` | pytest suite |
| `configs/` | `experiment_config.yaml`, the single source of environment and agent settings |
| `results/` | generated reports and per-episode CSVs -- **authoritative for every number** |
| `runs/` | escalation-loop ledger, append-only |
| `docs/` | working rules and history; start at `docs/INDEX.md` |
| `handoff/` | done criteria (`goal1.md`) and paper structure |
| `paper/` | LaTeX sources |

## Setup

```bash
pip install -e .        # dependencies come from requirements.txt
pytest -q               # 96 tests
```

The editable install is what lets `scripts/` import `envs`, `agents`, and the rest from
any working directory. Several scripts still carry a `sys.path.insert` line from before
packaging existed; those are now no-ops, deliberately left in place because
`results/GATE_C.md` cites some of these files by line number as gate evidence.

## Running

```bash
# training wave: N seeds x 8 algorithms
python scripts/run_wave.py --seeds 42,43,44,45,46 --floor-mode dynamic --max-parallel 6

# held-out evaluation, both readouts
python scripts/evaluate_checkpoints.py --runs results/logs/*.pt --episodes 30
python scripts/evaluate_checkpoints.py --runs results/logs/*.pt --episodes 30 --stochastic

# reports
python scripts/rliable_report.py   --eval-dir results/eval --out results/RLIABLE.md
python scripts/stability_report.py
python scripts/gate_b_report.py    --tag _v4 --readout primary

# guards -- both exit non-zero on a problem
python scripts/readout_audit.py     # every reported row traces to a readout file
python scripts/citation_audit.py    # path:line evidence in the gate docs still points where it claims

# Fase 0 diagnostics (docs/revisi/PLAN-01-DIAGNOSTICS.md) -- checkpoint evaluation, no training
python scripts/diag_equivariance.py   # D1 permutation equivariance, D4 parameter counts
python scripts/diag_gnn_reliance.py   # D2a/D2b message ablation, D3 over-smoothing
python scripts/diag_grad_ratio.py     # D2c gradient-norm ratio into the GNN
python scripts/diag_collision.py      # D5 collision-storm hypothesis
python scripts/diag_input_separability.py   # D6 which layer collapses the node representation

# Fase 1 (docs/revisi/PLAN-02) -- per-gNB resilient constraint, wave v5
python scripts/calibrate_fmin.py --sweep --checkpoints "results/logs/*_v4_seed4[2-6].pt"
python scripts/run_wave.py --seeds 42,43 --floor-mode none --resilient learned
```

`--resilient` defaults to `none`, which is bit-identical to v4 (asserted by
`tests/test_resilient.py`), so an unflagged wave is unchanged. `fixed` and `learned` refuse
to start until `resilient.f_min_mbps` is frozen in the config. As of 2026-08-25 it is not:
`calibrate_fmin.py` finds no eligible candidate and says so rather than picking one -- see
`docs/revisi/PREREG-V5.md` §0.

`diag_grad_ratio.py` is the only diagnostic that trains: it resumes a v4 checkpoint for a
few thousand steps to read gradients off the real training loop. It copies each checkpoint
to a scratch name first, so no v4 artifact is written -- verify with `md5sum` over
`results/checkpoints/*_v4_*_last.pt` before and after if you change it.

Training runs on GPU and is started by hand, not by an agent (`docs/HANDOVER.md` §11).

## Two things that are easy to get wrong

**The primary readout differs by family.** Sampled for PPO (pre-registered as P3),
argmax for DQN (human determination of 2026-08-16, after a pre-registered degeneracy
test). Reports state their readout in the header line; comparing a sampled PPO row
against a sampled DQN row is not a valid comparison. Families are never pooled in a
statistical claim (Gate C3).

**`results/quarantine_eps1.0/` is a fault, not a result.** Forty evaluation CSVs were
produced with `epsilon` missing from the checkpoint `state_dict`, making every
non-greedy DQN readout uniform random. They are quarantined rather than deleted because
the fault is itself material for the paper's methodology section. Do not read a number
out of that directory.

## Frozen operating point

Committed before the wave ran and unchanged since (Gate C6). From
`configs/experiment_config.yaml`:

```
delta                  0.085
lambda_arrival         60000.0
buffer.urllc_max_bits  307200
dual_update_every      12500
floor.mode             none
```

## Working here

Read `docs/INDEX.md` first -- it says which document wins when two disagree. The short
version: generated files under `results/` are authoritative for every number, and no
result is ever retyped from memory into prose.
