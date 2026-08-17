# Deleted-file manifest — zero-shot grid, header-only CSVs

**Reconstructed after the fact, not recorded at deletion time.** The deletion is described
in `runs/2026-08-05-run01/ledger.md` 2026-08-16T19:30 in prose only, which is not enough to
audit; this file supplies the paths. How the list was derived, so a reader can redo it: the
complete grid is 9 algorithms x 5 seeds x 4 topology directories x 2 readouts = 360
files. Present on disk: 280. Moved to `results/quarantine_eps1.0/`: 20.
The remainder, 60, are the deleted ones, listed below.

## Why they were deleted

A failed `evaluate_run` used to leave behind a CSV containing only its header row. Read back,
a header-only file is empty rather than absent, so it reported as INCOMPLETE ("the grid had
not finished") instead of CANNOT_RUN ("this architecture cannot be evaluated at this
topology") — a structural finding turned into a blank cell. `evaluate_run` now unlinks its
output on failure, and `scripts/zeroshot_eval.py` decides CANNOT_RUN from the architecture
rather than from which files exist, so neither the fault nor the misreading can recur.

Every deleted file was verified to be header-only and to belong to `central-*` before removal.
No file containing episode rows was deleted.

## Composition

| algorithm | readout | files |
|---|---|---|
| `central-dqn` | `_eval` | 20 |
| `central-ppo` | `_eval` | 20 |
| `central-ppo` | `_eval_stoch` | 20 |
| **total** | | **60** |

`central-dqn` `_eval_stoch` is absent from this table because those 20 files were not
deleted: the epsilon=1.0 fault filled them with random-action rows, so they were quarantined
instead — see `results/quarantine_eps1.0/README.md`.

## Paths

```
results/eval_zeroshot_v4/const-density/ngnb10/central-dqn_v4_seed42_eval.csv
results/eval_zeroshot_v4/const-density/ngnb10/central-dqn_v4_seed43_eval.csv
results/eval_zeroshot_v4/const-density/ngnb10/central-dqn_v4_seed44_eval.csv
results/eval_zeroshot_v4/const-density/ngnb10/central-dqn_v4_seed45_eval.csv
results/eval_zeroshot_v4/const-density/ngnb10/central-dqn_v4_seed46_eval.csv
results/eval_zeroshot_v4/const-density/ngnb10/central-ppo_v4_seed42_eval.csv
results/eval_zeroshot_v4/const-density/ngnb10/central-ppo_v4_seed42_eval_stoch.csv
results/eval_zeroshot_v4/const-density/ngnb10/central-ppo_v4_seed43_eval.csv
results/eval_zeroshot_v4/const-density/ngnb10/central-ppo_v4_seed43_eval_stoch.csv
results/eval_zeroshot_v4/const-density/ngnb10/central-ppo_v4_seed44_eval.csv
results/eval_zeroshot_v4/const-density/ngnb10/central-ppo_v4_seed44_eval_stoch.csv
results/eval_zeroshot_v4/const-density/ngnb10/central-ppo_v4_seed45_eval.csv
results/eval_zeroshot_v4/const-density/ngnb10/central-ppo_v4_seed45_eval_stoch.csv
results/eval_zeroshot_v4/const-density/ngnb10/central-ppo_v4_seed46_eval.csv
results/eval_zeroshot_v4/const-density/ngnb10/central-ppo_v4_seed46_eval_stoch.csv
results/eval_zeroshot_v4/const-density/ngnb20/central-dqn_v4_seed42_eval.csv
results/eval_zeroshot_v4/const-density/ngnb20/central-dqn_v4_seed43_eval.csv
results/eval_zeroshot_v4/const-density/ngnb20/central-dqn_v4_seed44_eval.csv
results/eval_zeroshot_v4/const-density/ngnb20/central-dqn_v4_seed45_eval.csv
results/eval_zeroshot_v4/const-density/ngnb20/central-dqn_v4_seed46_eval.csv
results/eval_zeroshot_v4/const-density/ngnb20/central-ppo_v4_seed42_eval.csv
results/eval_zeroshot_v4/const-density/ngnb20/central-ppo_v4_seed42_eval_stoch.csv
results/eval_zeroshot_v4/const-density/ngnb20/central-ppo_v4_seed43_eval.csv
results/eval_zeroshot_v4/const-density/ngnb20/central-ppo_v4_seed43_eval_stoch.csv
results/eval_zeroshot_v4/const-density/ngnb20/central-ppo_v4_seed44_eval.csv
results/eval_zeroshot_v4/const-density/ngnb20/central-ppo_v4_seed44_eval_stoch.csv
results/eval_zeroshot_v4/const-density/ngnb20/central-ppo_v4_seed45_eval.csv
results/eval_zeroshot_v4/const-density/ngnb20/central-ppo_v4_seed45_eval_stoch.csv
results/eval_zeroshot_v4/const-density/ngnb20/central-ppo_v4_seed46_eval.csv
results/eval_zeroshot_v4/const-density/ngnb20/central-ppo_v4_seed46_eval_stoch.csv
results/eval_zeroshot_v4/fixed-area/ngnb10/central-dqn_v4_seed42_eval.csv
results/eval_zeroshot_v4/fixed-area/ngnb10/central-dqn_v4_seed43_eval.csv
results/eval_zeroshot_v4/fixed-area/ngnb10/central-dqn_v4_seed44_eval.csv
results/eval_zeroshot_v4/fixed-area/ngnb10/central-dqn_v4_seed45_eval.csv
results/eval_zeroshot_v4/fixed-area/ngnb10/central-dqn_v4_seed46_eval.csv
results/eval_zeroshot_v4/fixed-area/ngnb10/central-ppo_v4_seed42_eval.csv
results/eval_zeroshot_v4/fixed-area/ngnb10/central-ppo_v4_seed42_eval_stoch.csv
results/eval_zeroshot_v4/fixed-area/ngnb10/central-ppo_v4_seed43_eval.csv
results/eval_zeroshot_v4/fixed-area/ngnb10/central-ppo_v4_seed43_eval_stoch.csv
results/eval_zeroshot_v4/fixed-area/ngnb10/central-ppo_v4_seed44_eval.csv
results/eval_zeroshot_v4/fixed-area/ngnb10/central-ppo_v4_seed44_eval_stoch.csv
results/eval_zeroshot_v4/fixed-area/ngnb10/central-ppo_v4_seed45_eval.csv
results/eval_zeroshot_v4/fixed-area/ngnb10/central-ppo_v4_seed45_eval_stoch.csv
results/eval_zeroshot_v4/fixed-area/ngnb10/central-ppo_v4_seed46_eval.csv
results/eval_zeroshot_v4/fixed-area/ngnb10/central-ppo_v4_seed46_eval_stoch.csv
results/eval_zeroshot_v4/fixed-area/ngnb20/central-dqn_v4_seed42_eval.csv
results/eval_zeroshot_v4/fixed-area/ngnb20/central-dqn_v4_seed43_eval.csv
results/eval_zeroshot_v4/fixed-area/ngnb20/central-dqn_v4_seed44_eval.csv
results/eval_zeroshot_v4/fixed-area/ngnb20/central-dqn_v4_seed45_eval.csv
results/eval_zeroshot_v4/fixed-area/ngnb20/central-dqn_v4_seed46_eval.csv
results/eval_zeroshot_v4/fixed-area/ngnb20/central-ppo_v4_seed42_eval.csv
results/eval_zeroshot_v4/fixed-area/ngnb20/central-ppo_v4_seed42_eval_stoch.csv
results/eval_zeroshot_v4/fixed-area/ngnb20/central-ppo_v4_seed43_eval.csv
results/eval_zeroshot_v4/fixed-area/ngnb20/central-ppo_v4_seed43_eval_stoch.csv
results/eval_zeroshot_v4/fixed-area/ngnb20/central-ppo_v4_seed44_eval.csv
results/eval_zeroshot_v4/fixed-area/ngnb20/central-ppo_v4_seed44_eval_stoch.csv
results/eval_zeroshot_v4/fixed-area/ngnb20/central-ppo_v4_seed45_eval.csv
results/eval_zeroshot_v4/fixed-area/ngnb20/central-ppo_v4_seed45_eval_stoch.csv
results/eval_zeroshot_v4/fixed-area/ngnb20/central-ppo_v4_seed46_eval.csv
results/eval_zeroshot_v4/fixed-area/ngnb20/central-ppo_v4_seed46_eval_stoch.csv
```
