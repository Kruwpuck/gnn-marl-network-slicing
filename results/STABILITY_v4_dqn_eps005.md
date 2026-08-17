# Stability Report — collapse rate over embb_p5_mbps

Readout: `non-greedy (P3 primary for PPO; for DQN it depends on the epsilon in --eval-dir, see provenance below)`. Source directory: `results/eval_dqn_eps005`.

Tag filter: `_v4`. Collapse threshold: 0.01 Mbps. Unit of collapse is the seed (mean over its held-out episodes), not the episode.

| algo | seeds collapsed | rate | 95% Wilson CI | worst seed mean | CVaR@20% |
|---|---|---|---|---|---|
| `central-dqn` | 0/5 | 0.00 | [0.00, 0.43] | 0.165086 | 0.000038 |
| `gnn-madqn_gat` | 3/5 | 0.60 | [0.23, 0.88] | 0.000038 | 0.000037 |
| `gnn-madqn_sage` | 0/5 | 0.00 | [0.00, 0.43] | 0.649138 | 0.206940 |
| `idqn` | 2/5 | 0.40 | [0.12, 0.77] | 0.000003 | 0.000002 |

## CVaR@20% — tail risk beside the collapse rate

Collapse rate answers *how often* a seed lands in the bad mode; CVaR answers *how bad the tail is* when it does. Two units are reported because they are not the same statistic:

- **episode CVaR** — mean of the worst episodes pooled across seeds, i.e. the within-run tail. 95% CI by stratified bootstrap (seeds resampled first, then episodes inside them).
- **seed CVaR** — mean of the worst seeds. At n=5 seeds and alpha=0.2 this takes the worst 1 seed, so it **degenerates to the worst-seed mean** and carries no more information than the column above. Printed anyway rather than quietly dropped: the degeneracy is a consequence of C4 failing at 5 seeds.

| algo | KPI | episode CVaR | 95% CI | seed CVaR | mean |
|---|---|---|---|---|---|
| `central-dqn` | `embb_p5_mbps` | 0.000038 | [0.000037, 0.003604] | 0.165086 | 0.781979 |
| `central-dqn` | `timely_throughput_mbps` | 44.339488 | [39.849105, 48.393499] | 60.876178 | 64.305343 |
| `central-dqn` | `sla_satisfaction_pct` | 60.809684 | [55.308597, 65.774141] | 81.173113 | 85.139672 |
| `gnn-madqn_gat` | `embb_p5_mbps` | 0.000037 | [0.000036, 0.000038] | 0.000038 | 0.634932 |
| `gnn-madqn_gat` | `timely_throughput_mbps` | 44.031799 | [38.853332, 48.577546] | 57.677358 | 63.454124 |
| `gnn-madqn_gat` | `sla_satisfaction_pct` | 60.531678 | [54.279032, 66.092119] | 77.472935 | 84.206670 |
| `gnn-madqn_sage` | `embb_p5_mbps` | 0.206940 | [0.000031, 1.176080] | 0.649138 | 1.538500 |
| `gnn-madqn_sage` | `timely_throughput_mbps` | 46.620870 | [44.483714, 48.823873] | 64.191355 | 64.804780 |
| `gnn-madqn_sage` | `sla_satisfaction_pct` | 63.610182 | [60.970063, 66.346598] | 84.997084 | 85.755814 |
| `idqn` | `embb_p5_mbps` | 0.000002 | [0.000001, 1.039226] | 0.000003 | 1.046292 |
| `idqn` | `timely_throughput_mbps` | 36.694962 | [29.094258, 51.331603] | 40.618556 | 60.811437 |
| `idqn` | `sla_satisfaction_pct` | 51.902945 | [42.274356, 69.583949] | 57.035395 | 81.076636 |

On `embb_p5_mbps` the episode-level tail spans 0.000002 (`idqn`) to 0.206940 (`gnn-madqn_sage`). Not every algorithm's tail falls below the collapse threshold, so the two statistics are separating different things here — read them together.

## Readout provenance — which file each row came from

| algo | family | readout | source |
|---|---|---|---|
| `central-dqn` | DQN | non-greedy — sampled for PPO, epsilon from the checkpoint for DQN | `results/eval_dqn_eps005/central-dqn_*_eval_stoch.csv` |
| `gnn-madqn_gat` | DQN | non-greedy — sampled for PPO, epsilon from the checkpoint for DQN | `results/eval_dqn_eps005/gnn-madqn_gat_*_eval_stoch.csv` |
| `gnn-madqn_sage` | DQN | non-greedy — sampled for PPO, epsilon from the checkpoint for DQN | `results/eval_dqn_eps005/gnn-madqn_sage_*_eval_stoch.csv` |
| `idqn` | DQN | non-greedy — sampled for PPO, epsilon from the checkpoint for DQN | `results/eval_dqn_eps005/idqn_*_eval_stoch.csv` |

For the DQN family the **primary** column is argmax, not the non-greedy one (determination 2026-08-16, after a pre-registered degeneracy test). Its valid non-greedy reading is epsilon=0.05 and lives in a separate file, `results/STABILITY_v4_dqn_eps005.md`. The epsilon=1.0 files that used to fill that column were uniform random actions and are quarantined (`results/quarantine_eps1.0/README.md`).