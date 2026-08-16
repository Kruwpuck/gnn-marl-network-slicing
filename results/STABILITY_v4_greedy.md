# Stability Report — collapse rate over embb_p5_mbps

Readout: `greedy (report-only, not gate)`.

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