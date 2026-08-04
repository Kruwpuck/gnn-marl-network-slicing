# Zero-shot topology transfer

n_gnb=5 checkpoints (main wave) evaluated at larger n_gnb without retraining. `central-*` fails with a shape mismatch by construction (obs_dim baked in at train time) -- CANNOT_RUN there is the expected result, not a bug.

| algo | n_gnb | status | mean throughput_mbps | mean embb_p5_mbps | n seeds |
|---|---|---|---|---|---|
| `central-dqn` | 10 | CANNOT_RUN | - | - | 0 |
| `central-dqn` | 20 | CANNOT_RUN | - | - | 0 |
| `central-ppo` | 10 | CANNOT_RUN | - | - | 0 |
| `central-ppo` | 20 | CANNOT_RUN | - | - | 0 |
| `gnn-madqn_gat` | 10 | OK | 53.6283 | 0.772859 | 5 |
| `gnn-madqn_gat` | 20 | OK | 98.2603 | 0.559591 | 5 |
| `gnn-madqn_sage` | 10 | OK | 52.9201 | 1.322236 | 5 |
| `gnn-madqn_sage` | 20 | OK | 96.7307 | 0.804922 | 5 |
| `gnn-mappo_gat` | 10 | OK | 52.5869 | 1.075150 | 5 |
| `gnn-mappo_gat` | 20 | OK | 96.2575 | 0.807309 | 5 |
| `gnn-mappo_sage` | 10 | OK | 53.3520 | 1.304158 | 5 |
| `gnn-mappo_sage` | 20 | OK | 97.5888 | 0.821591 | 5 |
| `idqn` | 10 | OK | 52.9210 | 1.036578 | 5 |
| `idqn` | 20 | OK | 96.6237 | 0.679456 | 5 |
| `ippo` | 10 | OK | 53.5427 | 0.000015 | 5 |
| `ippo` | 20 | OK | 98.1832 | 0.000015 | 5 |