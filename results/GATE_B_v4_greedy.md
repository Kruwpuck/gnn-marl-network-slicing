# Gate B — discrimination, recomputed

Readout: `greedy`. Tag `_v4`. 8 pre-registered algorithms; later additions such as `mlp-knn-ppo` are excluded because Gate B is a pre-registration and widening its algorithm set after seeing results would change the range it measures.

| # | KPI | min | max | range | threshold | verdict |
|---|---|---|---|---|---|---|
| B1 | `timely_throughput_mbps` | 22.6947 (`gnn-mappo_gat`) | 68.1519 (`ippo`) | 200.2987 % | >= 5% | **LOLOS** |
| B2 | `sla_satisfaction_pct` | 33.8009 (`gnn-mappo_gat`) | 89.8484 (`ippo`) | 56.0474 pp | >= 5 pp | **LOLOS** |
| B3 | `urllc_delay_p99` | 4.7280 (`gnn-mappo_gat`) | 8.3131 (`gnn-madqn_gat`) | 3.5851 ms | >= 2 ms | **LOLOS** |
| B4 | saturated KPIs | - | - | 0 of 5 | <= 1 of 5 | **LOLOS** |

## Per-algorithm means

| algo | `timely_throughput_mbps` | `sla_satisfaction_pct` | `urllc_delay_p99` | `embb_p5_mbps` | `jains_fairness` |
|---|---|---|---|---|---|
| `central-dqn` | 64.3389 | 85.1645 | 8.1253 | 0.8259 | 0.7515 |
| `central-ppo` | 65.2811 | 86.3129 | 7.5560 | 1.7492 | 0.6468 |
| `gnn-madqn_gat` | 63.2810 | 83.9557 | 8.3131 | 0.6399 | 0.7332 |
| `gnn-madqn_sage` | 64.5659 | 85.4417 | 8.2520 | 1.5582 | 0.6186 |
| `gnn-mappo_gat` | 22.6947 | 33.8009 | 4.7280 | 1.5702 | 0.6165 |
| `gnn-mappo_sage` | 40.1689 | 55.6089 | 6.6707 | 1.3765 | 0.6625 |
| `idqn` | 59.2646 | 79.1375 | 8.2880 | 1.0610 | 0.6573 |
| `ippo` | 68.1519 | 89.8484 | 6.7853 | 0.0000 | 0.7358 |