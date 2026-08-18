# Gate B — discrimination, recomputed

Readout: `primary`. Tag `_v4`. 8 pre-registered algorithms, seeds `42,43,44,45,46`; later additions such as `mlp-knn-ppo` are excluded because Gate B is a pre-registration and widening its algorithm set after seeing results would change the range it measures.

**Seed freeze.** 9000 eval rows from seeds outside the pre-registered set were excluded. The PPO family was extended to 20 seeds on 2026-08-17 while the DQN family stayed at 5; B1-B4 are ranges *across* the 8 algorithms, so unequal n would shift a pre-registered range for a statistical reason rather than a change in the task. The extra seeds are used for the C4 characterisation of the PPO family instead (`results/STABILITY_v4_primary.md`).

| # | KPI | min | max | range | threshold | verdict |
|---|---|---|---|---|---|---|
| B1 | `timely_throughput_mbps` | 59.2646 (`idqn`) | 68.9401 (`gnn-mappo_sage`) | 16.3259 % | >= 5% | **LOLOS** |
| B2 | `sla_satisfaction_pct` | 79.1375 (`idqn`) | 91.1192 (`gnn-mappo_sage`) | 11.9818 pp | >= 5 pp | **LOLOS** |
| B3 | `urllc_delay_p99` | 7.9827 (`central-ppo`) | 8.9967 (`gnn-mappo_gat`) | 1.0140 ms | >= 2 ms | **GAGAL** |
| B4 | saturated KPIs | - | - | 0 of 5 | <= 1 of 5 | **LOLOS** |

## C3 supplement — the same ranges within each budget family

Not a re-run of the gate. The pre-registered verdict above stands as computed across all 8 algorithms; this split only answers whether any verdict depends on mixing the 200K-step and 1M-step families.

| set | B1 relative range | B2 range | B3 range |
|---|---|---|---|
| all 8 (pre-registered, **this is the gate**) | 16.33% | 11.98 pp | 1.01 ms |
| DQN family (4, 200K steps) | 8.95% | 6.30 pp | 0.19 ms |
| PPO family (4, 1M steps) | 7.27% | 5.98 pp | 1.01 ms |

## Per-algorithm means

| algo | `timely_throughput_mbps` | `sla_satisfaction_pct` | `urllc_delay_p99` | `embb_p5_mbps` | `jains_fairness` |
|---|---|---|---|---|---|
| `central-dqn` | 64.3389 | 85.1645 | 8.1253 | 0.8259 | 0.7515 |
| `central-ppo` | 64.2696 | 85.1345 | 7.9827 | 0.4337 | 0.6636 |
| `gnn-madqn_gat` | 63.2810 | 83.9557 | 8.3131 | 0.6399 | 0.7332 |
| `gnn-madqn_sage` | 64.5659 | 85.4417 | 8.2520 | 1.5582 | 0.6186 |
| `gnn-mappo_gat` | 67.9670 | 90.0813 | 8.9967 | 0.2805 | 0.5343 |
| `gnn-mappo_sage` | 68.9401 | 91.1192 | 8.5525 | 0.0000 | 0.5025 |
| `idqn` | 59.2646 | 79.1375 | 8.2880 | 1.0610 | 0.6573 |
| `ippo` | 68.6121 | 90.6415 | 8.2520 | 0.0000 | 0.4874 |