# Stability Report — collapse rate over embb_p5_mbps

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