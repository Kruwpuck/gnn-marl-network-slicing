# Zero-shot topology transfer — wave v4

Readout: `primary per family — sampled for PPO (P3), argmax for DQN`. Episodes per checkpoint: 150. Trained at n_gnb=5, area_size=500 m, floor.mode=`none` (read from the config, not assumed).

`central-dqn` / `central-ppo` have `obs_dim = n_gnb * 8` fixed at training time, so CANNOT_RUN outside n_gnb=5 is a structural property of the architecture and is reported as a result, not as missing data. Every such row carries its reason, so **cannot be run** is never confused with **was not attempted**; the status is decided from the architecture, not from which files happen to exist.

Aggregate throughput grows with cell count no matter what the policy does, so **throughput per gNB** is the column to read; the aggregate is printed beside it. `retention` is per-gNB throughput relative to the same checkpoints at n_gnb=5.

Two arms are reported in full (integritas #3). `fixed-area` keeps area_size at 500 m so raising n_gnb also raises coupling strength; `const-density` scales area_size as sqrt(n/5) so density matches training. The env docstring (envs/network_slicing_env.py) is explicit that the first arm confounds agent count with coupling strength — it is kept because the v3 report used it.


## reference

| algo | n_gnb | status | reason | thr/gNB (Mbps) | retention | thr agg (Mbps) | sla_satisfaction_pct | embb_p5_mbps | n seeds |
|---|---|---|---|---|---|---|---|---|---|
| `central-dqn` | 5 | OK | | 12.8678 | 1.000 | 64.3389 | 85.1645 | 0.825904 | 5 |
| `central-ppo` | 5 | OK | | 12.8539 | 1.000 | 64.2696 | 85.1345 | 0.433748 | 5 |
| `gnn-madqn_gat` | 5 | OK | | 12.6562 | 1.000 | 63.2810 | 83.9557 | 0.639863 | 5 |
| `gnn-madqn_sage` | 5 | OK | | 12.9132 | 1.000 | 64.5659 | 85.4417 | 1.558226 | 5 |
| `gnn-mappo_gat` | 5 | OK | | 13.5934 | 1.000 | 67.9670 | 90.0813 | 0.280535 | 5 |
| `gnn-mappo_sage` | 5 | OK | | 13.7880 | 1.000 | 68.9401 | 91.1192 | 0.000004 | 5 |
| `idqn` | 5 | OK | | 11.8529 | 1.000 | 59.2646 | 79.1375 | 1.061016 | 5 |
| `ippo` | 5 | OK | | 13.7224 | 1.000 | 68.6121 | 90.6415 | 0.000003 | 5 |
| `mlp-knn-ppo` | 5 | OK | | 13.6387 | 1.000 | 68.1937 | 90.2615 | 0.000004 | 5 |

## fixed-area

| algo | n_gnb | status | reason | thr/gNB (Mbps) | retention | thr agg (Mbps) | sla_satisfaction_pct | embb_p5_mbps | n seeds |
|---|---|---|---|---|---|---|---|---|---|
| `central-dqn` | 10 | CANNOT_RUN | obs_dim mismatch: trained 40, required 80 | - | - | - | - | - | 0 |
| `central-dqn` | 20 | CANNOT_RUN | obs_dim mismatch: trained 40, required 160 | - | - | - | - | - | 0 |
| `central-ppo` | 10 | CANNOT_RUN | obs_dim mismatch: trained 40, required 80 | - | - | - | - | - | 0 |
| `central-ppo` | 20 | CANNOT_RUN | obs_dim mismatch: trained 40, required 160 | - | - | - | - | - | 0 |
| `gnn-madqn_gat` | 10 | OK | | 10.1260 | 0.800 | 101.2599 | 68.6419 | 0.497462 | 5 |
| `gnn-madqn_gat` | 20 | OK | | 7.0764 | 0.559 | 141.5279 | 49.9047 | 0.326373 | 5 |
| `gnn-madqn_sage` | 10 | OK | | 10.5121 | 0.814 | 105.1206 | 70.9016 | 1.147682 | 5 |
| `gnn-madqn_sage` | 20 | OK | | 7.6284 | 0.591 | 152.5671 | 53.2548 | 0.615622 | 5 |
| `gnn-mappo_gat` | 10 | OK | | 11.0263 | 0.811 | 110.2631 | 74.6510 | 0.165752 | 5 |
| `gnn-mappo_gat` | 20 | OK | | 7.8366 | 0.576 | 156.7319 | 55.0128 | 0.063375 | 5 |
| `gnn-mappo_sage` | 10 | OK | | 11.4192 | 0.828 | 114.1925 | 76.9632 | 0.000002 | 5 |
| `gnn-mappo_sage` | 20 | OK | | 8.2209 | 0.596 | 164.4181 | 57.3712 | 0.000001 | 5 |
| `idqn` | 10 | OK | | 9.6234 | 0.812 | 96.2341 | 65.5314 | 0.850105 | 5 |
| `idqn` | 20 | OK | | 7.0711 | 0.597 | 141.4220 | 49.8281 | 0.534295 | 5 |
| `ippo` | 10 | OK | | 11.3386 | 0.826 | 113.3859 | 76.3181 | 0.000001 | 5 |
| `ippo` | 20 | OK | | 8.2706 | 0.603 | 165.4117 | 57.5307 | 0.000001 | 5 |
| `mlp-knn-ppo` | 10 | OK | | 11.1920 | 0.821 | 111.9200 | 75.5544 | 0.000002 | 5 |
| `mlp-knn-ppo` | 20 | OK | | 8.0688 | 0.592 | 161.3753 | 56.3788 | 0.000001 | 5 |

## const-density

| algo | n_gnb | status | reason | thr/gNB (Mbps) | retention | thr agg (Mbps) | sla_satisfaction_pct | embb_p5_mbps | n seeds |
|---|---|---|---|---|---|---|---|---|---|
| `central-dqn` | 10 | CANNOT_RUN | obs_dim mismatch: trained 40, required 80 | - | - | - | - | - | 0 |
| `central-dqn` | 20 | CANNOT_RUN | obs_dim mismatch: trained 40, required 160 | - | - | - | - | - | 0 |
| `central-ppo` | 10 | CANNOT_RUN | obs_dim mismatch: trained 40, required 80 | - | - | - | - | - | 0 |
| `central-ppo` | 20 | CANNOT_RUN | obs_dim mismatch: trained 40, required 160 | - | - | - | - | - | 0 |
| `gnn-madqn_gat` | 10 | OK | | 12.1812 | 0.962 | 121.8118 | 81.1397 | 0.633715 | 5 |
| `gnn-madqn_gat` | 20 | OK | | 12.0378 | 0.951 | 240.7560 | 80.2892 | 0.676256 | 5 |
| `gnn-madqn_sage` | 10 | OK | | 12.4559 | 0.965 | 124.5589 | 82.7005 | 1.476060 | 5 |
| `gnn-madqn_sage` | 20 | OK | | 12.3296 | 0.955 | 246.5912 | 81.9583 | 1.470770 | 5 |
| `gnn-mappo_gat` | 10 | OK | | 13.0208 | 0.958 | 130.2084 | 86.6814 | 0.251634 | 5 |
| `gnn-mappo_gat` | 20 | OK | | 12.8109 | 0.942 | 256.2184 | 85.4597 | 0.209675 | 5 |
| `gnn-mappo_sage` | 10 | OK | | 13.2680 | 0.962 | 132.6798 | 88.0692 | 0.000003 | 5 |
| `gnn-mappo_sage` | 20 | OK | | 13.0961 | 0.950 | 261.9223 | 87.0505 | 0.000003 | 5 |
| `idqn` | 10 | OK | | 11.4659 | 0.967 | 114.6588 | 76.8233 | 0.998794 | 5 |
| `idqn` | 20 | OK | | 11.2876 | 0.952 | 225.7525 | 75.8011 | 1.029479 | 5 |
| `ippo` | 10 | OK | | 13.2260 | 0.964 | 132.2604 | 87.7054 | 0.000002 | 5 |
| `ippo` | 20 | OK | | 13.0910 | 0.954 | 261.8203 | 86.8944 | 0.000002 | 5 |
| `mlp-knn-ppo` | 10 | OK | | 13.1166 | 0.962 | 131.1660 | 87.1623 | 0.000004 | 5 |
| `mlp-knn-ppo` | 20 | OK | | 12.9431 | 0.949 | 258.8614 | 86.1432 | 0.000003 | 5 |

CI and per-family comparison are not computed here. Point `scripts/rliable_report.py --eval-dir results/eval_zeroshot_v4/<arm>/ngnb<N>` and `scripts/stability_report.py` at these directories instead — they already do IQM + stratified bootstrap per budget family (C3) and Wilson collapse rate.