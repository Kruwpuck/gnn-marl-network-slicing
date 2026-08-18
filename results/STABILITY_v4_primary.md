# Stability Report — collapse rate over embb_p5_mbps

Readout: `primary per family — sampled for PPO (P3), argmax for DQN (2026-08-16)`. Source directory: `results/eval`.

Tag filter: `_v4`. Collapse threshold: 0.01 Mbps. Unit of collapse is the seed (mean over its held-out episodes), not the episode.

| algo | seeds collapsed | rate | 95% Wilson CI | worst seed mean | CVaR@20% |
|---|---|---|---|---|---|
| `central-dqn` | 0/5 | 0.00 | [0.00, 0.43] | 0.278037 | 0.000038 |
| `central-ppo` | 3/20 | 0.15 | [0.05, 0.36] | 0.000040 | 0.000039 |
| `gnn-madqn_gat` | 2/5 | 0.40 | [0.12, 0.77] | 0.000038 | 0.000037 |
| `gnn-madqn_sage` | 0/5 | 0.00 | [0.00, 0.43] | 0.738033 | 0.247377 |
| `gnn-mappo_gat` | 14/20 | 0.70 | [0.48, 0.85] | 0.000002 | 0.000002 |
| `gnn-mappo_sage` | 19/20 | 0.95 | [0.76, 0.99] | 0.000003 | 0.000001 |
| `idqn` | 2/5 | 0.40 | [0.12, 0.77] | 0.000004 | 0.000003 |
| `ippo` | 20/20 | 1.00 | [0.84, 1.00] | 0.000002 | 0.000001 |
| `mlp-knn-ppo` | 5/5 | 1.00 | [0.57, 1.00] | 0.000003 | 0.000002 |

## CVaR@20% — tail risk beside the collapse rate

Collapse rate answers *how often* a seed lands in the bad mode; CVaR answers *how bad the tail is* when it does. Two units are reported because they are not the same statistic:

- **episode CVaR** — mean of the worst episodes pooled across seeds, i.e. the within-run tail. 95% CI by stratified bootstrap (seeds resampled first, then episodes inside them).
- **seed CVaR** — mean of the worst seeds. At n=20 seeds and alpha=0.2 this takes the worst 4 seeds, so it **degenerates to the worst-seed mean** and carries no more information than the column above. Printed anyway rather than quietly dropped: the degeneracy is a consequence of C4 failing at 5 seeds.

| algo | KPI | episode CVaR | 95% CI | seed CVaR | mean |
|---|---|---|---|---|---|
| `central-dqn` | `embb_p5_mbps` | 0.000038 | [0.000037, 0.014848] | 0.278037 | 0.825904 |
| `central-dqn` | `timely_throughput_mbps` | 44.273433 | [39.485847, 48.549612] | 60.813180 | 64.338880 |
| `central-dqn` | `sla_satisfaction_pct` | 60.719125 | [54.860956, 65.924849] | 81.056282 | 85.164454 |
| `central-ppo` | `embb_p5_mbps` | 0.000039 | [0.000039, 0.000040] | 0.003203 | 0.488663 |
| `central-ppo` | `timely_throughput_mbps` | 45.844485 | [44.888831, 46.883644] | 64.027967 | 64.219675 |
| `central-ppo` | `sla_satisfaction_pct` | 62.633508 | [61.492079, 63.923563] | 84.851654 | 85.075223 |
| `gnn-madqn_gat` | `embb_p5_mbps` | 0.000037 | [0.000036, 0.000038] | 0.000038 | 0.639863 |
| `gnn-madqn_gat` | `timely_throughput_mbps` | 43.668431 | [38.354386, 48.306929] | 57.485815 | 63.280999 |
| `gnn-madqn_gat` | `sla_satisfaction_pct` | 60.023989 | [53.568384, 65.706900] | 77.140401 | 83.955673 |
| `gnn-madqn_sage` | `embb_p5_mbps` | 0.247377 | [0.000032, 1.176492] | 0.738033 | 1.558226 |
| `gnn-madqn_sage` | `timely_throughput_mbps` | 46.257318 | [44.081835, 48.525212] | 63.912421 | 64.565896 |
| `gnn-madqn_sage` | `sla_satisfaction_pct` | 63.133103 | [60.466767, 65.946031] | 84.640675 | 85.441659 |
| `gnn-mappo_gat` | `embb_p5_mbps` | 0.000002 | [0.000002, 0.000003] | 0.000003 | 0.278056 |
| `gnn-mappo_gat` | `timely_throughput_mbps` | 53.307478 | [51.975963, 54.763905] | 66.472559 | 68.071588 |
| `gnn-mappo_gat` | `sla_satisfaction_pct` | 72.426240 | [70.713981, 74.227825] | 88.074353 | 90.095149 |
| `gnn-mappo_sage` | `embb_p5_mbps` | 0.000001 | [0.000001, 0.000002] | 0.000003 | 0.043084 |
| `gnn-mappo_sage` | `timely_throughput_mbps` | 54.315430 | [52.738032, 56.029118] | 67.464372 | 68.599434 |
| `gnn-mappo_sage` | `sla_satisfaction_pct` | 73.614165 | [71.686025, 75.753204] | 89.320188 | 90.735429 |
| `idqn` | `embb_p5_mbps` | 0.000003 | [0.000001, 1.131902] | 0.000004 | 1.061016 |
| `idqn` | `timely_throughput_mbps` | 32.252114 | [25.111935, 49.660544] | 35.110009 | 59.264628 |
| `idqn` | `sla_satisfaction_pct` | 46.357062 | [37.209258, 67.520602] | 50.160299 | 79.137452 |
| `ippo` | `embb_p5_mbps` | 0.000001 | [0.000001, 0.000001] | 0.000003 | 0.000003 |
| `ippo` | `timely_throughput_mbps` | 54.216783 | [53.247399, 55.214361] | 68.321837 | 68.869782 |
| `ippo` | `sla_satisfaction_pct` | 73.455476 | [72.278159, 74.659210] | 90.265085 | 90.975734 |
| `mlp-knn-ppo` | `embb_p5_mbps` | 0.000002 | [0.000001, 0.000002] | 0.000003 | 0.000004 |
| `mlp-knn-ppo` | `timely_throughput_mbps` | 53.533499 | [51.484298, 55.649610] | 67.521732 | 68.193699 |
| `mlp-knn-ppo` | `sla_satisfaction_pct` | 72.741462 | [70.190475, 75.327748] | 89.367647 | 90.261533 |

On `embb_p5_mbps` the episode-level tail spans 0.000001 (`ippo`) to 0.247377 (`gnn-madqn_sage`). Not every algorithm's tail falls below the collapse threshold, so the two statistics are separating different things here — read them together.

## Readout provenance — which file each row came from

| algo | family | readout | source |
|---|---|---|---|
| `central-dqn` | DQN | argmax (greedy) | `results/eval/central-dqn_*_eval.csv` |
| `central-ppo` | PPO | sampled (P3) | `results/eval/central-ppo_*_eval_stoch.csv` |
| `gnn-madqn_gat` | DQN | argmax (greedy) | `results/eval/gnn-madqn_gat_*_eval.csv` |
| `gnn-madqn_sage` | DQN | argmax (greedy) | `results/eval/gnn-madqn_sage_*_eval.csv` |
| `gnn-mappo_gat` | PPO | sampled (P3) | `results/eval/gnn-mappo_gat_*_eval_stoch.csv` |
| `gnn-mappo_sage` | PPO | sampled (P3) | `results/eval/gnn-mappo_sage_*_eval_stoch.csv` |
| `idqn` | DQN | argmax (greedy) | `results/eval/idqn_*_eval.csv` |
| `ippo` | PPO | sampled (P3) | `results/eval/ippo_*_eval_stoch.csv` |
| `mlp-knn-ppo` | PPO | sampled (P3) | `results/eval/mlp-knn-ppo_*_eval_stoch.csv` |

For the DQN family the **primary** column is argmax, not the non-greedy one (determination 2026-08-16, after a pre-registered degeneracy test). Its valid non-greedy reading is epsilon=0.05 and lives in a separate file, `results/STABILITY_v4_dqn_eps005.md`. The epsilon=1.0 files that used to fill that column were uniform random actions and are quarantined (`results/quarantine_eps1.0/README.md`).