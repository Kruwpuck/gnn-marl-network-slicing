# D1 permutation equivariance + D4 parameter count

Checkpoints: `results/logs/*_v4_seed42.pt`. Warm-up: 50 policy steps from `env.reset(seed=10000)` before the state is frozen.

**Why the warm-up.** `reset()` zeroes `_last_sinr_embb`, `_prev_alloc`, `_prev_alloc_lag2`, `_viol_ewma`, `_last_backlog_bits` and `_queue_embb`, so 7 of the 8 observation columns are identical across gNB at t=0 and only `ch_gain` varies. Permuting a nearly-uniform state is nearly a no-op: every model would pass D1 for a reason unrelated to equivariance. The state is taken after the policy has driven the environment for the stated number of steps.

**Readout.** D1 tests a mapping, not a KPI, so P3 does not apply -- the primary pass is argmax. The sampled pass is a second check under a fixed per-permutation seed, not a KPI reading. Do not compare these numbers against anything in `results/RLIABLE_*.md`.

**Reading the two columns.** `action_var_max` is the largest per-gNB variance of the un-permuted action across all 120 relabellings. `score_dev_max` is the largest absolute deviation of the un-permuted pre-argmax logits/Q-values from the identity permutation, with `score_scale` for context. Float reduction order changes when nodes are permuted, so an argmax over near-tied scores can flip on the last bit: a non-zero `action_var_max` with a `score_dev_max` orders of magnitude below `score_scale` is a tie being broken differently, not a model reading node identity.

**`central-dqn` / `central-ppo` rows are not comparable to the rest.** Those two consume the flattened global observation and broadcast one action to every gNB (`scripts/evaluate_checkpoints.py` `select_actions`), so the un-permuted action vector is constant by construction and would read as perfectly equivariant. Their rows report the single pre-broadcast decision instead.

## D1

| algo | kind | argmax action_var_max | argmax distinct action vectors | argmax score_dev_max | score_scale | sampled action_var_max |
|---|---|---|---|---|---|---|
| `central-dqn` *(pre-broadcast scalar)* | mlp-dqn-central | 8.798e+00 | 2 | 1.624e+02 | 470.1287 | 8.798e+00 |
| `central-ppo` *(pre-broadcast scalar)* | mlp-ppo-central | 1.808e+00 | 3 | 3.737e-01 | 1.2017 | 0.000e+00 |
| `gnn-madqn_gat` | gnn-dqn | 0.000e+00 | 1 | 1.831e-04 | 247.8780 | 0.000e+00 |
| `gnn-madqn_sage` | gnn-dqn | 0.000e+00 | 1 | 4.272e-04 | 313.1984 | 0.000e+00 |
| `gnn-mappo_gat` | gnn-ppo | 0.000e+00 | 1 | 2.384e-07 | 0.9061 | 8.240e+00 |
| `gnn-mappo_sage` | gnn-ppo | 0.000e+00 | 1 | 1.192e-07 | 1.9708 | 1.096e+01 |
| `idqn` | mlp-dqn | 0.000e+00 | 1 | 0.000e+00 | 142.4671 | 0.000e+00 |
| `ippo` | mlp-ppo | 0.000e+00 | 1 | 0.000e+00 | 1.0791 | 8.240e+00 |
| `mlp-knn-ppo` | mlp-knn-ppo | 0.000e+00 | 1 | 0.000e+00 | 0.6169 | 1.024e+01 |

## D4 trainable parameters

PLAN-01 D4 requires this table in the paper whichever way it falls: if the counts are lopsided, any performance difference can be attributed to capacity rather than to graph inductive bias, in either direction.

| algo | total | backbone (GNN) | head |
|---|---|---|---|
| `central-dqn` | 39,820 | -- | 39,820 |
| `central-ppo` | 12,044 | -- | 12,044 |
| `gnn-madqn_gat` | 37,580 | 19,392 | 18,188 |
| `gnn-madqn_sage` | 27,532 | 9,344 | 18,188 |
| `gnn-mappo_gat` | 37,580 | 19,392 | 18,188 |
| `gnn-mappo_sage` | 27,532 | 9,344 | 18,188 |
| `idqn` | 35,724 | -- | 35,724 |
| `ippo` | 3,852 | -- | 3,852 |
| `mlp-knn-ppo` | 12,044 | -- | 12,044 |
