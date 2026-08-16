# Zero-shot topology transfer

n_gnb=5 checkpoints (main wave) evaluated at larger n_gnb without retraining. `central-*` fails with a shape mismatch by construction (obs_dim baked in at train time) -- CANNOT_RUN there is the expected result, not a bug.

| algo | n_gnb | status | mean throughput_mbps | mean embb_p5_mbps | n seeds |
|---|---|---|---|---|---|
| `central-dqn` | 20 | CANNOT_RUN | - | - | 0 |
| `central-ppo` | 20 | CANNOT_RUN | - | - | 0 |
| `gnn-madqn_gat` | 20 | OK | 86.5283 | 0.713021 | 1 |
| `gnn-madqn_sage` | 20 | OK | 88.2422 | 0.711678 | 1 |
| `gnn-mappo_gat` | 20 | OK | 93.0375 | 0.000038 | 1 |
| `gnn-mappo_sage` | 20 | OK | 86.4896 | 0.656100 | 1 |
| `idqn` | 20 | OK | 86.1764 | 0.774527 | 1 |
| `ippo` | 20 | OK | 93.0375 | 0.000038 | 1 |