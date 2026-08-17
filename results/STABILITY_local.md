# Stability Report — collapse rate over embb_p5_mbps

> **Readout: `greedy` (wave v3). Superseded — cite `results/STABILITY.md` instead.** Label
> added 2026-08-17 by the provenance audit (`scripts/readout_audit.py`). This file is the
> bare generated table for the same v3 data; `STABILITY.md` carries identical numbers plus
> the seed-count power analysis. Kept so earlier references do not dangle.

Tag filter: `(main wave)`. Collapse threshold: 0.01 Mbps. Unit of collapse is the seed (mean over its held-out episodes), not the episode.

| algo | seeds collapsed | rate | 95% Wilson CI | worst seed mean | CVaR@20% |
|---|---|---|---|---|---|
| `central-dqn` | 0/5 | 0.00 | [0.00, 0.43] | 0.254387 | 0.000038 |
| `central-ppo` | 0/5 | 0.00 | [0.00, 0.43] | 1.773975 | 1.147688 |
| `gnn-madqn_gat` | 1/5 | 0.20 | [0.04, 0.62] | 0.000038 | 0.000037 |
| `gnn-madqn_sage` | 0/5 | 0.00 | [0.00, 0.43] | 1.772099 | 1.175351 |
| `gnn-mappo_gat` | 1/5 | 0.20 | [0.04, 0.62] | 0.000038 | 0.000038 |
| `gnn-mappo_sage` | 0/5 | 0.00 | [0.00, 0.43] | 1.772099 | 1.181356 |
| `idqn` | 1/5 | 0.20 | [0.04, 0.62] | 0.000004 | 0.000004 |
| `ippo` | 5/5 | 1.00 | [0.57, 1.00] | 0.000002 | 0.000001 |