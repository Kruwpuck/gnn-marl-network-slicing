# Stability Report — collapse rate over embb_p5_mbps

> **VOID FOR THE DQN FAMILY — do not cite this file.** The `central-dqn`, `idqn`, `gnn-madqn_gat` and `gnn-madqn_sage` rows were produced by the
> epsilon=1.0 readout, i.e. uniform random actions rather than the trained policy
> (`results/quarantine_eps1.0/README.md`). The source CSVs have been quarantined, so this
> file cannot be regenerated as-is and is kept only as a record of what was reported before
> the fault was found. Valid replacement: `results/STABILITY_v4_primary.md`.

Readout: `stochastic (P3 primary)`.

Tag filter: `_v4`. Collapse threshold: 0.01 Mbps. Unit of collapse is the seed (mean over its held-out episodes), not the episode.

| algo | seeds collapsed | rate | 95% Wilson CI | worst seed mean | CVaR@20% |
|---|---|---|---|---|---|
| `central-dqn` | 1/5 | 0.20 | [0.04, 0.62] | 0.007117 | 0.000039 |
| `central-ppo` | 0/5 | 0.00 | [0.00, 0.43] | 0.059709 | 0.000040 |
| `gnn-madqn_gat` | 5/5 | 1.00 | [0.57, 1.00] | 0.000005 | 0.000002 |
| `gnn-madqn_sage` | 5/5 | 1.00 | [0.57, 1.00] | 0.000005 | 0.000002 |
| `gnn-mappo_gat` | 4/5 | 0.80 | [0.38, 0.96] | 0.000004 | 0.000002 |
| `gnn-mappo_sage` | 5/5 | 1.00 | [0.57, 1.00] | 0.000003 | 0.000001 |
| `idqn` | 5/5 | 1.00 | [0.57, 1.00] | 0.000005 | 0.000002 |
| `ippo` | 5/5 | 1.00 | [0.57, 1.00] | 0.000003 | 0.000001 |

## CVaR@20% — tail risk beside the collapse rate

Collapse rate answers *how often* a seed lands in the bad mode; CVaR answers *how bad the tail is* when it does. Two units are reported because they are not the same statistic:

- **episode CVaR** — mean of the worst episodes pooled across seeds, i.e. the within-run tail. 95% CI by stratified bootstrap (seeds resampled first, then episodes inside them).
- **seed CVaR** — mean of the worst seeds. At n=5 seeds and alpha=0.2 this takes the worst 1 seed, so it **degenerates to the worst-seed mean** and carries no more information than the column above. Printed anyway rather than quietly dropped: the degeneracy is a consequence of C4 failing at 5 seeds.

| algo | KPI | episode CVaR | 95% CI | seed CVaR | mean |
|---|---|---|---|---|---|
| `central-dqn` | `embb_p5_mbps` | 0.000039 | [0.000039, 0.000039] | 0.007117 | 0.026477 |
| `central-dqn` | `timely_throughput_mbps` | 42.856969 | [40.879186, 44.866838] | 61.808078 | 61.870991 |
| `central-dqn` | `sla_satisfaction_pct` | 59.160600 | [56.675724, 61.664087] | 82.363886 | 82.423736 |
| `central-ppo` | `embb_p5_mbps` | 0.000040 | [0.000039, 0.000040] | 0.059709 | 0.433748 |
| `central-ppo` | `timely_throughput_mbps` | 45.878046 | [43.979522, 47.953451] | 64.077334 | 64.269636 |
| `central-ppo` | `sla_satisfaction_pct` | 62.679139 | [60.368944, 65.283391] | 84.909821 | 85.134451 |
| `gnn-madqn_gat` | `embb_p5_mbps` | 0.000002 | [0.000002, 0.000003] | 0.000005 | 0.000006 |
| `gnn-madqn_gat` | `timely_throughput_mbps` | 51.114784 | [49.145910, 53.094987] | 67.072211 | 67.090338 |
| `gnn-madqn_gat` | `sla_satisfaction_pct` | 69.626730 | [67.210327, 72.054131] | 88.798164 | 88.830211 |
| `gnn-madqn_sage` | `embb_p5_mbps` | 0.000002 | [0.000002, 0.000003] | 0.000005 | 0.000005 |
| `gnn-madqn_sage` | `timely_throughput_mbps` | 51.110662 | [49.215603, 53.061060] | 67.012954 | 67.079935 |
| `gnn-madqn_sage` | `sla_satisfaction_pct` | 69.612041 | [67.273998, 71.963389] | 88.737499 | 88.798638 |
| `gnn-mappo_gat` | `embb_p5_mbps` | 0.000002 | [0.000001, 0.000003] | 0.000004 | 0.280535 |
| `gnn-mappo_gat` | `timely_throughput_mbps` | 53.301368 | [50.878627, 55.881587] | 67.317053 | 67.967021 |
| `gnn-mappo_gat` | `sla_satisfaction_pct` | 72.522336 | [69.416503, 75.768958] | 89.113014 | 90.081270 |
| `gnn-mappo_sage` | `embb_p5_mbps` | 0.000001 | [0.000001, 0.000002] | 0.000003 | 0.000004 |
| `gnn-mappo_sage` | `timely_throughput_mbps` | 55.363440 | [52.172456, 58.703018] | 67.720148 | 68.940134 |
| `gnn-mappo_sage` | `sla_satisfaction_pct` | 74.916741 | [71.049721, 79.053321] | 89.596588 | 91.119216 |
| `idqn` | `embb_p5_mbps` | 0.000002 | [0.000002, 0.000003] | 0.000005 | 0.000005 |
| `idqn` | `timely_throughput_mbps` | 51.277524 | [49.314415, 53.213865] | 66.978920 | 67.063316 |
| `idqn` | `sla_satisfaction_pct` | 69.803616 | [67.373401, 72.198727] | 88.686177 | 88.789659 |
| `ippo` | `embb_p5_mbps` | 0.000001 | [0.000001, 0.000001] | 0.000003 | 0.000003 |
| `ippo` | `timely_throughput_mbps` | 53.667065 | [51.672883, 55.592570] | 68.218064 | 68.612085 |
| `ippo` | `sla_satisfaction_pct` | 72.752107 | [70.310144, 75.061849] | 90.142867 | 90.641535 |

On `embb_p5_mbps` the episode-level tail spans 0.000001 (`ippo`) to 0.000040 (`central-ppo`), and **every** algorithm's tail sits below the 0.01 Mbps collapse threshold. In the worst 20% of episodes cell-edge service is absent for all of them; what differs between architectures is how often a whole seed lands in that mode, which is what the collapse rate above measures. CVaR is reported as the complement it is, not as a second version of the same finding.