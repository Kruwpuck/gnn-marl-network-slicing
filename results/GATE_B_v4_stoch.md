# Gate B — discrimination, recomputed

> **VOID FOR THE DQN FAMILY — do not cite this file.** Every gate number here spans all 8 algorithms, half of which were produced by the
> epsilon=1.0 readout, i.e. uniform random actions rather than the trained policy
> (`results/quarantine_eps1.0/README.md`). The source CSVs have been quarantined, so this
> file cannot be regenerated as-is and is kept only as a record of what was reported before
> the fault was found. Valid replacement: `results/GATE_B_v4_primary.md`.

Readout: `stochastic`. Tag `_v4`. 8 pre-registered algorithms; later additions such as `mlp-knn-ppo` are excluded because Gate B is a pre-registration and widening its algorithm set after seeing results would change the range it measures.

| # | KPI | min | max | range | threshold | verdict |
|---|---|---|---|---|---|---|
| B1 | `timely_throughput_mbps` | 61.8710 (`central-dqn`) | 68.9401 (`gnn-mappo_sage`) | 11.4256 % | >= 5% | **LOLOS** |
| B2 | `sla_satisfaction_pct` | 82.4237 (`central-dqn`) | 91.1192 (`gnn-mappo_sage`) | 8.6955 pp | >= 5 pp | **LOLOS** |
| B3 | `urllc_delay_p99` | 7.9827 (`central-ppo`) | 8.9967 (`gnn-mappo_gat`) | 1.0140 ms | >= 2 ms | **GAGAL** |
| B4 | saturated KPIs | - | - | 1 of 5 (`embb_p5_mbps`) | <= 1 of 5 | **LOLOS** |

## Per-algorithm means

| algo | `timely_throughput_mbps` | `sla_satisfaction_pct` | `urllc_delay_p99` | `embb_p5_mbps` | `jains_fairness` |
|---|---|---|---|---|---|
| `central-dqn` | 61.8710 | 82.4237 | 8.9773 | 0.0265 | 0.6517 |
| `central-ppo` | 64.2696 | 85.1345 | 7.9827 | 0.4337 | 0.6636 |
| `gnn-madqn_gat` | 67.0903 | 88.8302 | 8.7133 | 0.0000 | 0.5389 |
| `gnn-madqn_sage` | 67.0799 | 88.7986 | 8.7600 | 0.0000 | 0.5387 |
| `gnn-mappo_gat` | 67.9670 | 90.0813 | 8.9967 | 0.2805 | 0.5343 |
| `gnn-mappo_sage` | 68.9401 | 91.1192 | 8.5525 | 0.0000 | 0.5025 |
| `idqn` | 67.0633 | 88.7897 | 8.7359 | 0.0000 | 0.5390 |
| `ippo` | 68.6121 | 90.6415 | 8.2520 | 0.0000 | 0.4874 |