# f_min calibration -- steps 1-2

Population: trained checkpoints `results/logs/*_v4_seed4[2-6].pt`. 200 steps each, frozen operating point (`delta=0.085`, `floor.mode=none`). The constraint is switched off for this sweep: calibrating `f_min` against a policy already shaped by `f_min` would be circular.

**Not the PRB floor.** `env._compute_floor`'s `f_min` is an allocation *fraction*; `resilient.f_min_mbps` is a *rate* floor in Mbps. Same word, different units.

**Rule, declared before the numbers were read.** Among candidate policies whose mean violation satisfies `delta = 0.085`, take the one with the highest mean eMBB rate, and propose the 25th percentile of its per-gNB rate distribution over **non-collapsed seeds only**. Collapse is the project's existing cell-edge rule, `embb_p5_mbps < 0.01` with the seed as the unit (`scripts/stability_report.py`) -- reused, not redefined. A collapsed seed sits at the floor and would drag the percentile toward zero, which is the *too low to bind* failure this protocol exists to avoid.

**Only the reference family is eligible.** PLAN-02 section 7 step 3 names a non-GNN baseline -- `ippo`, consistent with Gate A -- and that is enforced here rather than left to the ranking rule. Ranking every checkpoint by mean eMBB picks `gnn-mappo`, which would set `f_min` from the proposed method's own reachable distribution and bake its advantage into the constraint. Larangan 1 forbids exactly that. Non-reference rows are shown for context and marked. A candidate also needs at least 3 non-collapsed runs, so no percentile can rest on a single seed.

| policy | eMBB mean (Mbps) | eMBB p5 (Mbps) | violation | <= delta | non-collapsed | eligible |
|---|---|---|---|---|---|---|
| `central-dqn` | 5.167 | 0.7754 | 0.0785 | yes | 2/5 | context only |
| `central-ppo` | 8.954 | 0.3523 | 0.0879 | no | 1/5 | context only |
| `gnn-madqn` | 6.761 | 1.0945 | 0.1444 | no | 6/10 | context only |
| `gnn-mappo` | 10.064 | 0.1907 | 0.0448 | yes | 1/10 | context only |
| `idqn` | 5.526 | 1.0119 | 0.2320 | no | 3/5 | context only |
| `ippo` | 9.136 | 0.0000 | 0.1013 | no | 0/5 | reference |
| `mlp-knn-ppo` | 7.986 | 0.0000 | 0.1133 | no | 0/5 | context only |

## No candidate

Nothing eligible both satisfies `delta = 0.085` and leaves at least 3 non-collapsed runs. That is a finding about the operating point, not a number to soften: report it and do not pick an `f_min` anyway.

For the **static** population this is the expected outcome -- `delta` was calibrated against a trained policy, so re-run with `--checkpoints`.

For the **trained** population it is a real blocker and belongs in front of a human. The reference family cannot currently satisfy the constraint and avoid cell-edge collapse at the same time, which is the same condition that leaves the project at NOT DONE (Gate B3 fails, C4 fails for the DQN family). Calibrating a rate floor against a reference that itself collapses is not possible; either the operating point moves, or the reference family for this calibration is re-declared -- and both are decisions for a human, recorded in the ledger before anything is frozen.

