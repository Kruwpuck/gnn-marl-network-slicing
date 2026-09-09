# Stability Report — collapse rate over embb_p5_mbps

Readout: `primary per family — sampled for PPO (P3), argmax for DQN (2026-08-16)`. Source directory: `results/eval`.

Tag filter: `_v6`. Collapse threshold: 0.01 Mbps. Unit of collapse is the seed (mean over its held-out episodes), not the episode.

| algo | seeds collapsed | rate | 95% Wilson CI | worst seed mean | CVaR@20% |
|---|---|---|---|---|---|
| `gnn-madqn_gatedge` | 0/5 | 0.00 | [0.00, 0.43] | 0.140445 | 0.000037 |
| `gnn-madqn_gatres` | 0/5 | 0.00 | [0.00, 0.43] | 0.012889 | 0.000017 |
| `gnn-madqn_gatres-edge` | 2/5 | 0.40 | [0.12, 0.77] | 0.000032 | 0.000019 |
| `gnn-mappo_gatedge` | 14/20 | 0.70 | [0.48, 0.85] | 0.000002 | 0.000001 |
| `gnn-mappo_gatres` | 10/20 | 0.50 | [0.30, 0.70] | 0.000003 | 0.000002 |
| `gnn-mappo_gatres-edge` | 14/20 | 0.70 | [0.48, 0.85] | 0.000003 | 0.000002 |

## CVaR@20% — tail risk beside the collapse rate

Collapse rate answers *how often* a seed lands in the bad mode; CVaR answers *how bad the tail is* when it does. Two units are reported because they are not the same statistic:

- **episode CVaR** — mean of the worst episodes pooled across seeds, i.e. the within-run tail. 95% CI by stratified bootstrap (seeds resampled first, then episodes inside them).
- **seed CVaR** — mean of the worst seeds. At n=20 seeds and alpha=0.2 this takes the worst 4 seeds, so it **degenerates to the worst-seed mean** and carries no more information than the column above. Printed anyway rather than quietly dropped: the degeneracy is a consequence of C4 failing at 5 seeds.

| algo | KPI | episode CVaR | 95% CI | seed CVaR | mean |
|---|---|---|---|---|---|
| `gnn-madqn_gatedge` | `embb_p5_mbps` | 0.000037 | [0.000037, 0.000038] | 0.140445 | 0.633587 |
| `gnn-madqn_gatedge` | `timely_throughput_mbps` | 41.247312 | [34.839233, 46.428682] | 61.057193 | 63.009195 |
| `gnn-madqn_gatedge` | `sla_satisfaction_pct` | 57.035486 | [49.153618, 63.401488] | 81.479018 | 83.679776 |
| `gnn-madqn_gatres` | `embb_p5_mbps` | 0.000017 | [0.000007, 0.037448] | 0.012889 | 0.760406 |
| `gnn-madqn_gatres` | `timely_throughput_mbps` | 44.872134 | [41.521251, 48.665177] | 59.615993 | 63.945456 |
| `gnn-madqn_gatres` | `sla_satisfaction_pct` | 61.496514 | [57.438406, 66.196391] | 79.896851 | 84.960682 |
| `gnn-madqn_gatres-edge` | `embb_p5_mbps` | 0.000019 | [0.000009, 0.000031] | 0.000032 | 0.558908 |
| `gnn-madqn_gatres-edge` | `timely_throughput_mbps` | 45.603527 | [42.021132, 49.043805] | 62.109460 | 64.303546 |
| `gnn-madqn_gatres-edge` | `sla_satisfaction_pct` | 62.367749 | [57.852482, 66.619302] | 82.794180 | 85.237300 |
| `gnn-mappo_gatedge` | `embb_p5_mbps` | 0.000001 | [0.000001, 0.000002] | 0.000003 | 0.087356 |
| `gnn-mappo_gatedge` | `timely_throughput_mbps` | 53.487582 | [51.981771, 55.229455] | 66.681638 | 68.227863 |
| `gnn-mappo_gatedge` | `sla_satisfaction_pct` | 72.611348 | [70.758509, 74.749836] | 88.415520 | 90.320295 |
| `gnn-mappo_gatres` | `embb_p5_mbps` | 0.000002 | [0.000001, 0.000003] | 0.000003 | 0.389718 |
| `gnn-mappo_gatres` | `timely_throughput_mbps` | 52.996906 | [51.631610, 54.503855] | 66.343083 | 67.841179 |
| `gnn-mappo_gatres` | `sla_satisfaction_pct` | 72.046033 | [70.341969, 73.895871] | 87.994195 | 89.856531 |
| `gnn-mappo_gatres-edge` | `embb_p5_mbps` | 0.000002 | [0.000002, 0.000003] | 0.000003 | 0.275739 |
| `gnn-mappo_gatres-edge` | `timely_throughput_mbps` | 52.976523 | [51.417649, 54.565593] | 66.087830 | 67.923901 |
| `gnn-mappo_gatres-edge` | `sla_satisfaction_pct` | 71.970260 | [70.028430, 73.919747] | 87.627735 | 89.922986 |

On `embb_p5_mbps` the episode-level tail spans 0.000001 (`gnn-mappo_gatedge`) to 0.000037 (`gnn-madqn_gatedge`), and **every** algorithm's tail sits below the 0.01 Mbps collapse threshold. In the worst 20% of episodes cell-edge service is absent for all of them; what differs between architectures is how often a whole seed lands in that mode, which is what the collapse rate above measures. CVaR is reported as the complement it is, not as a second version of the same finding.

## Readout provenance — which file each row came from

| algo | family | readout | source |
|---|---|---|---|
| `gnn-madqn_gatedge` | DQN | argmax (greedy) | `results/eval/gnn-madqn_gatedge_*_eval.csv` |
| `gnn-madqn_gatres` | DQN | argmax (greedy) | `results/eval/gnn-madqn_gatres_*_eval.csv` |
| `gnn-madqn_gatres-edge` | DQN | argmax (greedy) | `results/eval/gnn-madqn_gatres-edge_*_eval.csv` |
| `gnn-mappo_gatedge` | PPO | sampled (P3) | `results/eval/gnn-mappo_gatedge_*_eval_stoch.csv` |
| `gnn-mappo_gatres` | PPO | sampled (P3) | `results/eval/gnn-mappo_gatres_*_eval_stoch.csv` |
| `gnn-mappo_gatres-edge` | PPO | sampled (P3) | `results/eval/gnn-mappo_gatres-edge_*_eval_stoch.csv` |

For the DQN family the **primary** column is argmax, not the non-greedy one (determination 2026-08-16, after a pre-registered degeneracy test). Its valid non-greedy reading is epsilon=0.05 and lives in a separate file, `results/STABILITY_v4_dqn_eps005.md`. The epsilon=1.0 files that used to fill that column were uniform random actions and are quarantined (`results/quarantine_eps1.0/README.md`).