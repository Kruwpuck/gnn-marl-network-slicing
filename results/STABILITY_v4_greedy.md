# Stability Report — collapse rate over embb_p5_mbps

Readout: `greedy (report-only, not gate)`. Source directory: `results/eval`.

Tag filter: `_v4`. Collapse threshold: 0.01 Mbps. Unit of collapse is the seed (mean over its held-out episodes), not the episode.

| algo | seeds collapsed | rate | 95% Wilson CI | worst seed mean | CVaR@20% |
|---|---|---|---|---|---|
| `central-dqn` | 0/5 | 0.00 | [0.00, 0.43] | 0.278037 | 0.000038 |
| `central-ppo` | 0/5 | 0.00 | [0.00, 0.43] | 1.736439 | 0.990243 |
| `gnn-madqn_gat` | 2/5 | 0.40 | [0.12, 0.77] | 0.000038 | 0.000037 |
| `gnn-madqn_sage` | 0/5 | 0.00 | [0.00, 0.43] | 0.738033 | 0.247377 |
| `gnn-mappo_gat` | 0/5 | 0.00 | [0.00, 0.43] | 0.939668 | 0.302137 |
| `gnn-mappo_sage` | 1/5 | 0.20 | [0.04, 0.62] | 0.000036 | 0.000032 |
| `idqn` | 2/5 | 0.40 | [0.12, 0.77] | 0.000004 | 0.000003 |
| `ippo` | 5/5 | 1.00 | [0.57, 1.00] | 0.000002 | 0.000001 |
| `mlp-knn-ppo` | 2/5 | 0.40 | [0.12, 0.77] | 0.000028 | 0.000003 |

## CVaR@20% — tail risk beside the collapse rate

Collapse rate answers *how often* a seed lands in the bad mode; CVaR answers *how bad the tail is* when it does. Two units are reported because they are not the same statistic:

- **episode CVaR** — mean of the worst episodes pooled across seeds, i.e. the within-run tail. 95% CI by stratified bootstrap (seeds resampled first, then episodes inside them).
- **seed CVaR** — mean of the worst seeds. At n=5 seeds and alpha=0.2 this takes the worst 1 seed, so it **degenerates to the worst-seed mean** and carries no more information than the column above. Printed anyway rather than quietly dropped: the degeneracy is a consequence of C4 failing at 5 seeds.

| algo | KPI | episode CVaR | 95% CI | seed CVaR | mean |
|---|---|---|---|---|---|
| `central-dqn` | `embb_p5_mbps` | 0.000038 | [0.000037, 0.014848] | 0.278037 | 0.825904 |
| `central-dqn` | `timely_throughput_mbps` | 44.273433 | [39.485847, 48.549612] | 60.813180 | 64.338880 |
| `central-dqn` | `sla_satisfaction_pct` | 60.719125 | [54.860956, 65.924849] | 81.056282 | 85.164454 |
| `central-ppo` | `embb_p5_mbps` | 0.990243 | [0.859963, 1.145583] | 1.736439 | 1.749188 |
| `central-ppo` | `timely_throughput_mbps` | 47.434579 | [45.402821, 49.570118] | 64.544173 | 65.281090 |
| `central-ppo` | `sla_satisfaction_pct` | 64.608136 | [62.107555, 67.261058] | 85.437111 | 86.312925 |
| `gnn-madqn_gat` | `embb_p5_mbps` | 0.000037 | [0.000036, 0.000038] | 0.000038 | 0.639863 |
| `gnn-madqn_gat` | `timely_throughput_mbps` | 43.668431 | [38.354386, 48.306929] | 57.485815 | 63.280999 |
| `gnn-madqn_gat` | `sla_satisfaction_pct` | 60.023989 | [53.568384, 65.706900] | 77.140401 | 83.955673 |
| `gnn-madqn_sage` | `embb_p5_mbps` | 0.247377 | [0.000032, 1.176492] | 0.738033 | 1.558226 |
| `gnn-madqn_sage` | `timely_throughput_mbps` | 46.257318 | [44.081835, 48.525212] | 63.912421 | 64.565896 |
| `gnn-madqn_sage` | `sla_satisfaction_pct` | 63.133103 | [60.466767, 65.946031] | 84.640675 | 85.441659 |
| `gnn-mappo_gat` | `embb_p5_mbps` | 0.302137 | [0.000039, 1.171448] | 0.939668 | 1.570221 |
| `gnn-mappo_gat` | `timely_throughput_mbps` | 0.000198 | [0.000196, 7.047991] | 0.000202 | 22.694694 |
| `gnn-mappo_gat` | `sla_satisfaction_pct` | 5.446219 | [5.396815, 14.367937] | 5.516624 | 33.800937 |
| `gnn-mappo_sage` | `embb_p5_mbps` | 0.000032 | [0.000025, 1.087323] | 0.000036 | 1.376508 |
| `gnn-mappo_sage` | `timely_throughput_mbps` | 0.000202 | [0.000197, 37.115222] | 0.000202 | 40.168926 |
| `gnn-mappo_sage` | `sla_satisfaction_pct` | 5.515410 | [5.415834, 52.227647] | 5.516624 | 55.608942 |
| `idqn` | `embb_p5_mbps` | 0.000003 | [0.000001, 1.131902] | 0.000004 | 1.061016 |
| `idqn` | `timely_throughput_mbps` | 32.252114 | [25.111935, 49.660544] | 35.110009 | 59.264628 |
| `idqn` | `sla_satisfaction_pct` | 46.357062 | [37.209258, 67.520602] | 50.160299 | 79.137452 |
| `ippo` | `embb_p5_mbps` | 0.000001 | [0.000000, 0.000016] | 0.000002 | 0.000020 |
| `ippo` | `timely_throughput_mbps` | 49.950339 | [46.988198, 54.159959] | 65.296274 | 68.151875 |
| `ippo` | `sla_satisfaction_pct` | 67.796929 | [64.109280, 72.922607] | 86.325955 | 89.848357 |
| `mlp-knn-ppo` | `embb_p5_mbps` | 0.000003 | [0.000001, 0.000033] | 0.000028 | 0.685794 |
| `mlp-knn-ppo` | `timely_throughput_mbps` | 3.247581 | [1.260833, 43.466965] | 8.837067 | 42.112346 |
| `mlp-knn-ppo` | `sla_satisfaction_pct` | 9.639458 | [7.155366, 59.817329] | 16.757866 | 58.039794 |

On `embb_p5_mbps` the episode-level tail spans 0.000001 (`ippo`) to 0.990243 (`central-ppo`). Not every algorithm's tail falls below the collapse threshold, so the two statistics are separating different things here — read them together.

## Readout provenance — which file each row came from

| algo | family | readout | source |
|---|---|---|---|
| `central-dqn` | DQN | argmax (greedy) | `results/eval/central-dqn_*_eval.csv` |
| `central-ppo` | PPO | argmax (greedy) | `results/eval/central-ppo_*_eval.csv` |
| `gnn-madqn_gat` | DQN | argmax (greedy) | `results/eval/gnn-madqn_gat_*_eval.csv` |
| `gnn-madqn_sage` | DQN | argmax (greedy) | `results/eval/gnn-madqn_sage_*_eval.csv` |
| `gnn-mappo_gat` | PPO | argmax (greedy) | `results/eval/gnn-mappo_gat_*_eval.csv` |
| `gnn-mappo_sage` | PPO | argmax (greedy) | `results/eval/gnn-mappo_sage_*_eval.csv` |
| `idqn` | DQN | argmax (greedy) | `results/eval/idqn_*_eval.csv` |
| `ippo` | PPO | argmax (greedy) | `results/eval/ippo_*_eval.csv` |
| `mlp-knn-ppo` | PPO | argmax (greedy) | `results/eval/mlp-knn-ppo_*_eval.csv` |

For the DQN family the **primary** column is argmax, not the non-greedy one (determination 2026-08-16, after a pre-registered degeneracy test). Its valid non-greedy reading is epsilon=0.05 and lives in a separate file, `results/STABILITY_v4_dqn_eps005.md`. The epsilon=1.0 files that used to fill that column were uniform random actions and are quarantined (`results/quarantine_eps1.0/README.md`).