# Stability Report — collapse rate over embb_p5_mbps

Readout: `primary per family — sampled for PPO (P3), argmax for DQN (2026-08-16)`.

Tag filter: `_v4`. Collapse threshold: 0.01 Mbps. Unit of collapse is the seed (mean over its held-out episodes), not the episode.

| algo | seeds collapsed | rate | 95% Wilson CI | worst seed mean | CVaR@20% |
|---|---|---|---|---|---|
| `central-dqn` | 0/5 | 0.00 | [0.00, 0.43] | 0.278037 | 0.000038 |
| `central-ppo` | 0/5 | 0.00 | [0.00, 0.43] | 0.059709 | 0.000040 |
| `gnn-madqn_gat` | 2/5 | 0.40 | [0.12, 0.77] | 0.000038 | 0.000037 |
| `gnn-madqn_sage` | 0/5 | 0.00 | [0.00, 0.43] | 0.738033 | 0.247377 |
| `gnn-mappo_gat` | 4/5 | 0.80 | [0.38, 0.96] | 0.000004 | 0.000002 |
| `gnn-mappo_sage` | 5/5 | 1.00 | [0.57, 1.00] | 0.000003 | 0.000001 |
| `idqn` | 2/5 | 0.40 | [0.12, 0.77] | 0.000004 | 0.000003 |
| `ippo` | 5/5 | 1.00 | [0.57, 1.00] | 0.000003 | 0.000001 |
| `mlp-knn-ppo` | 5/5 | 1.00 | [0.57, 1.00] | 0.000003 | 0.000002 |

## CVaR@20% — tail risk beside the collapse rate

Collapse rate answers *how often* a seed lands in the bad mode; CVaR answers *how bad the tail is* when it does. Two units are reported because they are not the same statistic:

- **episode CVaR** — mean of the worst episodes pooled across seeds, i.e. the within-run tail. 95% CI by stratified bootstrap (seeds resampled first, then episodes inside them).
- **seed CVaR** — mean of the worst seeds. At n=5 seeds and alpha=0.2 this takes the worst 1 seed, so it **degenerates to the worst-seed mean** and carries no more information than the column above. Printed anyway rather than quietly dropped: the degeneracy is a consequence of C4 failing at 5 seeds.

| algo | KPI | episode CVaR | 95% CI | seed CVaR | mean |
|---|---|---|---|---|---|
| `central-dqn` | `embb_p5_mbps` | 0.000038 | [0.000037, 0.014848] | 0.278037 | 0.825904 |
| `central-dqn` | `timely_throughput_mbps` | 44.273433 | [39.485847, 48.549612] | 60.813180 | 64.338880 |
| `central-dqn` | `sla_satisfaction_pct` | 60.719125 | [54.860956, 65.924849] | 81.056282 | 85.164454 |
| `central-ppo` | `embb_p5_mbps` | 0.000040 | [0.000039, 0.000040] | 0.059709 | 0.433748 |
| `central-ppo` | `timely_throughput_mbps` | 45.878046 | [43.979522, 47.953451] | 64.077334 | 64.269636 |
| `central-ppo` | `sla_satisfaction_pct` | 62.679139 | [60.368944, 65.283391] | 84.909821 | 85.134451 |
| `gnn-madqn_gat` | `embb_p5_mbps` | 0.000037 | [0.000036, 0.000038] | 0.000038 | 0.639863 |
| `gnn-madqn_gat` | `timely_throughput_mbps` | 43.668431 | [38.354386, 48.306929] | 57.485815 | 63.280999 |
| `gnn-madqn_gat` | `sla_satisfaction_pct` | 60.023989 | [53.568384, 65.706900] | 77.140401 | 83.955673 |
| `gnn-madqn_sage` | `embb_p5_mbps` | 0.247377 | [0.000032, 1.176492] | 0.738033 | 1.558226 |
| `gnn-madqn_sage` | `timely_throughput_mbps` | 46.257318 | [44.081835, 48.525212] | 63.912421 | 64.565896 |
| `gnn-madqn_sage` | `sla_satisfaction_pct` | 63.133103 | [60.466767, 65.946031] | 84.640675 | 85.441659 |
| `gnn-mappo_gat` | `embb_p5_mbps` | 0.000002 | [0.000001, 0.000003] | 0.000004 | 0.280535 |
| `gnn-mappo_gat` | `timely_throughput_mbps` | 53.301368 | [50.878627, 55.881587] | 67.317053 | 67.967021 |
| `gnn-mappo_gat` | `sla_satisfaction_pct` | 72.522336 | [69.416503, 75.768958] | 89.113014 | 90.081270 |
| `gnn-mappo_sage` | `embb_p5_mbps` | 0.000001 | [0.000001, 0.000002] | 0.000003 | 0.000004 |
| `gnn-mappo_sage` | `timely_throughput_mbps` | 55.363440 | [52.172456, 58.703018] | 67.720148 | 68.940134 |
| `gnn-mappo_sage` | `sla_satisfaction_pct` | 74.916741 | [71.049721, 79.053321] | 89.596588 | 91.119216 |
| `idqn` | `embb_p5_mbps` | 0.000003 | [0.000001, 1.131902] | 0.000004 | 1.061016 |
| `idqn` | `timely_throughput_mbps` | 32.252114 | [25.111935, 49.660544] | 35.110009 | 59.264628 |
| `idqn` | `sla_satisfaction_pct` | 46.357062 | [37.209258, 67.520602] | 50.160299 | 79.137452 |
| `ippo` | `embb_p5_mbps` | 0.000001 | [0.000001, 0.000001] | 0.000003 | 0.000003 |
| `ippo` | `timely_throughput_mbps` | 53.667065 | [51.672883, 55.592570] | 68.218064 | 68.612085 |
| `ippo` | `sla_satisfaction_pct` | 72.752107 | [70.310144, 75.061849] | 90.142867 | 90.641535 |
| `mlp-knn-ppo` | `embb_p5_mbps` | 0.000002 | [0.000001, 0.000002] | 0.000003 | 0.000004 |
| `mlp-knn-ppo` | `timely_throughput_mbps` | 53.533499 | [51.484298, 55.649610] | 67.521732 | 68.193699 |
| `mlp-knn-ppo` | `sla_satisfaction_pct` | 72.741462 | [70.190475, 75.327748] | 89.367647 | 90.261533 |

On `embb_p5_mbps` the episode-level tail spans 0.000001 (`ippo`) to 0.247377 (`gnn-madqn_sage`). Not every algorithm's tail falls below the collapse threshold, so the two statistics are separating different things here — read them together.