# Zero-shot topology transfer — wave v4

Readout: `greedy (reported, never gates)`. Episodes per checkpoint: 150. Trained at n_gnb=5, area_size=500 m, floor.mode=`none` (read from the config, not assumed).

`central-dqn` / `central-ppo` have `obs_dim = n_gnb * 8` fixed at training time, so CANNOT_RUN outside n_gnb=5 is a structural property of the architecture and is reported as a result, not as missing data.

Aggregate throughput grows with cell count no matter what the policy does, so **throughput per gNB** is the column to read; the aggregate is printed beside it. `retention` is per-gNB throughput relative to the same checkpoints at n_gnb=5.

Two arms are reported in full (integritas #3). `fixed-area` keeps area_size at 500 m so raising n_gnb also raises coupling strength; `const-density` scales area_size as sqrt(n/5) so density matches training. The env docstring (envs/network_slicing_env.py) is explicit that the first arm confounds agent count with coupling strength — it is kept because the v3 report used it.


## reference

| algo | n_gnb | status | thr/gNB (Mbps) | retention | thr agg (Mbps) | sla_satisfaction_pct | embb_p5_mbps | n seeds |
|---|---|---|---|---|---|---|---|---|
| `central-dqn` | 5 | OK | 12.8678 | 1.000 | 64.3389 | 85.1645 | 0.825904 | 5 |
| `central-ppo` | 5 | OK | 13.0562 | 1.000 | 65.2811 | 86.3129 | 1.749188 | 5 |
| `gnn-madqn_gat` | 5 | OK | 12.6562 | 1.000 | 63.2810 | 83.9557 | 0.639863 | 5 |
| `gnn-madqn_sage` | 5 | OK | 12.9132 | 1.000 | 64.5659 | 85.4417 | 1.558226 | 5 |
| `gnn-mappo_gat` | 5 | OK | 4.5389 | 1.000 | 22.6947 | 33.8009 | 1.570221 | 5 |
| `gnn-mappo_sage` | 5 | OK | 8.0338 | 1.000 | 40.1689 | 55.6089 | 1.376508 | 5 |
| `idqn` | 5 | OK | 11.8529 | 1.000 | 59.2646 | 79.1375 | 1.061016 | 5 |
| `ippo` | 5 | OK | 13.6304 | 1.000 | 68.1519 | 89.8484 | 0.000020 | 5 |
| `mlp-knn-ppo` | 5 | OK | 8.4225 | 1.000 | 42.1123 | 58.0398 | 0.685794 | 5 |

## fixed-area

| algo | n_gnb | status | thr/gNB (Mbps) | retention | thr agg (Mbps) | sla_satisfaction_pct | embb_p5_mbps | n seeds |
|---|---|---|---|---|---|---|---|---|
| `central-dqn` | 10 | INCOMPLETE | - | - | - | - | - | 0 |
| `central-dqn` | 20 | INCOMPLETE | - | - | - | - | - | 0 |
| `central-ppo` | 10 | INCOMPLETE | - | - | - | - | - | 0 |
| `central-ppo` | 20 | INCOMPLETE | - | - | - | - | - | 0 |
| `gnn-madqn_gat` | 10 | OK | 10.1260 | 0.800 | 101.2599 | 68.6419 | 0.497462 | 5 |
| `gnn-madqn_gat` | 20 | OK | 7.0764 | 0.559 | 141.5279 | 49.9047 | 0.326373 | 5 |
| `gnn-madqn_sage` | 10 | OK | 10.5121 | 0.814 | 105.1206 | 70.9016 | 1.147682 | 5 |
| `gnn-madqn_sage` | 20 | OK | 7.6284 | 0.591 | 152.5671 | 53.2548 | 0.615622 | 5 |
| `gnn-mappo_gat` | 10 | OK | 3.0200 | 0.665 | 30.1996 | 24.4342 | 1.425798 | 5 |
| `gnn-mappo_gat` | 20 | OK | 1.7611 | 0.388 | 35.2224 | 16.5572 | 0.966510 | 5 |
| `gnn-mappo_sage` | 10 | OK | 5.4726 | 0.681 | 54.7260 | 39.8774 | 1.056559 | 5 |
| `gnn-mappo_sage` | 20 | OK | 4.1426 | 0.516 | 82.8520 | 31.5696 | 0.605992 | 5 |
| `idqn` | 10 | OK | 9.6234 | 0.812 | 96.2341 | 65.5314 | 0.850105 | 5 |
| `idqn` | 20 | OK | 7.0711 | 0.597 | 141.4220 | 49.8281 | 0.534295 | 5 |
| `ippo` | 10 | OK | 11.3670 | 0.834 | 113.6702 | 76.2702 | 0.000017 | 5 |
| `ippo` | 20 | OK | 8.2506 | 0.605 | 165.0113 | 57.2080 | 0.000018 | 5 |
| `mlp-knn-ppo` | 10 | OK | 7.0568 | 0.838 | 70.5675 | 49.7828 | 0.510145 | 5 |
| `mlp-knn-ppo` | 20 | OK | 5.2975 | 0.629 | 105.9498 | 38.9647 | 0.205656 | 5 |

## const-density

| algo | n_gnb | status | thr/gNB (Mbps) | retention | thr agg (Mbps) | sla_satisfaction_pct | embb_p5_mbps | n seeds |
|---|---|---|---|---|---|---|---|---|
| `central-dqn` | 10 | INCOMPLETE | - | - | - | - | - | 0 |
| `central-dqn` | 20 | INCOMPLETE | - | - | - | - | - | 0 |
| `central-ppo` | 10 | INCOMPLETE | - | - | - | - | - | 0 |
| `central-ppo` | 20 | INCOMPLETE | - | - | - | - | - | 0 |
| `gnn-madqn_gat` | 10 | OK | 12.1812 | 0.962 | 121.8118 | 81.1397 | 0.633715 | 5 |
| `gnn-madqn_gat` | 20 | OK | 12.0378 | 0.951 | 240.7560 | 80.2892 | 0.676256 | 5 |
| `gnn-madqn_sage` | 10 | OK | 12.4559 | 0.965 | 124.5589 | 82.7005 | 1.476060 | 5 |
| `gnn-madqn_sage` | 20 | OK | 12.3296 | 0.955 | 246.5912 | 81.9583 | 1.470770 | 5 |
| `gnn-mappo_gat` | 10 | OK | 3.6327 | 0.800 | 36.3270 | 28.2327 | 1.595069 | 5 |
| `gnn-mappo_gat` | 20 | OK | 3.0583 | 0.674 | 61.1654 | 24.6693 | 1.654456 | 5 |
| `gnn-mappo_sage` | 10 | OK | 7.1681 | 0.892 | 71.6808 | 50.4228 | 1.269669 | 5 |
| `gnn-mappo_sage` | 20 | OK | 6.7233 | 0.837 | 134.4663 | 47.7438 | 1.103605 | 5 |
| `idqn` | 10 | OK | 11.4659 | 0.967 | 114.6588 | 76.8233 | 0.998794 | 5 |
| `idqn` | 20 | OK | 11.2876 | 0.952 | 225.7525 | 75.8011 | 1.029479 | 5 |
| `ippo` | 10 | OK | 13.2436 | 0.972 | 132.4360 | 87.5610 | 0.000018 | 5 |
| `ippo` | 20 | OK | 13.1510 | 0.965 | 263.0193 | 87.0122 | 0.000017 | 5 |
| `mlp-knn-ppo` | 10 | OK | 8.2115 | 0.975 | 82.1151 | 56.7669 | 0.643306 | 5 |
| `mlp-knn-ppo` | 20 | OK | 8.1192 | 0.964 | 162.3836 | 56.2347 | 0.565265 | 5 |

CI and per-family comparison are not computed here. Point `scripts/rliable_report.py --eval-dir results/eval_zeroshot_v4/<arm>/ngnb<N>` and `scripts/stability_report.py` at these directories instead — they already do IQM + stratified bootstrap per budget family (C3) and Wilson collapse rate.