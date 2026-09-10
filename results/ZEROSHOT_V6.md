# Zero-shot topology transfer — wave v6

Readout: `primary per family — sampled for PPO (P3), argmax for DQN`. Episodes per checkpoint: 150. Trained at n_gnb=5, area_size=500 m, floor.mode=`none` (read from the config, not assumed).

`central-dqn` / `central-ppo` have `obs_dim = n_gnb * 8` fixed at training time, so CANNOT_RUN outside n_gnb=5 is a structural property of the architecture and is reported as a result, not as missing data. Every such row carries its reason, so **cannot be run** is never confused with **was not attempted**; the status is decided from the architecture, not from which files happen to exist.

Aggregate throughput grows with cell count no matter what the policy does, so **throughput per gNB** is the column to read; the aggregate is printed beside it. `retention` is per-gNB throughput relative to the same checkpoints at n_gnb=5.

Two arms are reported in full (integritas #3). `fixed-area` keeps area_size at 500 m so raising n_gnb also raises coupling strength; `const-density` scales area_size as sqrt(n/5) so density matches training. The env docstring (envs/network_slicing_env.py) is explicit that the first arm confounds agent count with coupling strength — it is kept because the v3 report used it.


## reference

| algo | n_gnb | status | reason | thr/gNB (Mbps) | retention | thr agg (Mbps) | sla_satisfaction_pct | embb_p5_mbps | n seeds |
|---|---|---|---|---|---|---|---|---|---|
| `gnn-madqn_gatedge` | 5 | OK | | 12.6018 | 1.000 | 63.0092 | 83.6798 | 0.633587 | 5 |
| `gnn-madqn_gatres` | 5 | OK | | 12.7891 | 1.000 | 63.9455 | 84.9607 | 0.760406 | 5 |
| `gnn-madqn_gatres-edge` | 5 | OK | | 12.8607 | 1.000 | 64.3035 | 85.2373 | 0.558908 | 5 |
| `gnn-mappo_gatedge` | 5 | OK | | 13.6456 | 1.000 | 68.2279 | 90.3203 | 0.087356 | 20 |
| `gnn-mappo_gatres` | 5 | OK | | 13.5682 | 1.000 | 67.8412 | 89.8565 | 0.389718 | 20 |
| `gnn-mappo_gatres-edge` | 5 | OK | | 13.5848 | 1.000 | 67.9239 | 89.9230 | 0.275739 | 20 |

## fixed-area

| algo | n_gnb | status | reason | thr/gNB (Mbps) | retention | thr agg (Mbps) | sla_satisfaction_pct | embb_p5_mbps | n seeds |
|---|---|---|---|---|---|---|---|---|---|
| `gnn-madqn_gatedge` | 10 | OK | | 9.7589 | 0.774 | 97.5888 | 66.3756 | 0.542217 | 5 |
| `gnn-madqn_gatedge` | 20 | OK | | 6.2981 | 0.500 | 125.9619 | 45.0248 | 0.320613 | 5 |
| `gnn-madqn_gatres` | 10 | OK | | 10.3057 | 0.806 | 103.0571 | 69.7881 | 0.675526 | 5 |
| `gnn-madqn_gatres` | 20 | OK | | 7.2680 | 0.568 | 145.3600 | 51.1026 | 0.488252 | 5 |
| `gnn-madqn_gatres-edge` | 10 | OK | | 10.3333 | 0.803 | 103.3329 | 69.9097 | 0.496475 | 5 |
| `gnn-madqn_gatres-edge` | 20 | OK | | 7.2658 | 0.565 | 145.3161 | 51.0199 | 0.174050 | 5 |
| `gnn-mappo_gatedge` | 10 | OK | | 11.1934 | 0.820 | 111.9341 | 75.5842 | 0.042909 | 20 |
| `gnn-mappo_gatedge` | 20 | OK | | 8.0636 | 0.591 | 161.2725 | 56.3437 | 0.030714 | 20 |
| `gnn-mappo_gatres` | 10 | OK | | 11.0784 | 0.816 | 110.7842 | 74.8925 | 0.223398 | 20 |
| `gnn-mappo_gatres` | 20 | OK | | 7.9497 | 0.586 | 158.9946 | 55.6521 | 0.090475 | 20 |
| `gnn-mappo_gatres-edge` | 10 | OK | | 11.1277 | 0.819 | 111.2775 | 75.1577 | 0.171292 | 20 |
| `gnn-mappo_gatres-edge` | 20 | OK | | 8.0208 | 0.590 | 160.4155 | 56.0615 | 0.076201 | 20 |

## const-density

| algo | n_gnb | status | reason | thr/gNB (Mbps) | retention | thr agg (Mbps) | sla_satisfaction_pct | embb_p5_mbps | n seeds |
|---|---|---|---|---|---|---|---|---|---|
| `gnn-madqn_gatedge` | 10 | OK | | 12.0428 | 0.956 | 120.4276 | 80.2848 | 0.594169 | 5 |
| `gnn-madqn_gatedge` | 20 | OK | | 11.8137 | 0.937 | 236.2735 | 78.9524 | 0.685787 | 5 |
| `gnn-madqn_gatres` | 10 | OK | | 12.3052 | 0.962 | 123.0523 | 81.9624 | 0.717637 | 5 |
| `gnn-madqn_gatres` | 20 | OK | | 12.1858 | 0.953 | 243.7151 | 81.2308 | 0.705019 | 5 |
| `gnn-madqn_gatres-edge` | 10 | OK | | 12.3571 | 0.961 | 123.5711 | 82.2012 | 0.661613 | 5 |
| `gnn-madqn_gatres-edge` | 20 | OK | | 12.1771 | 0.947 | 243.5429 | 81.0959 | 0.700558 | 5 |
| `gnn-mappo_gatedge` | 10 | OK | | 13.1151 | 0.961 | 131.1505 | 87.1738 | 0.061165 | 20 |
| `gnn-mappo_gatedge` | 20 | OK | | 12.9303 | 0.948 | 258.6052 | 86.0919 | 0.041274 | 20 |
| `gnn-mappo_gatres` | 10 | OK | | 13.0197 | 0.960 | 130.1974 | 86.6038 | 0.333103 | 20 |
| `gnn-mappo_gatres` | 20 | OK | | 12.8305 | 0.946 | 256.6091 | 85.5033 | 0.275800 | 20 |
| `gnn-mappo_gatres-edge` | 10 | OK | | 13.0538 | 0.961 | 130.5384 | 86.7798 | 0.238549 | 20 |
| `gnn-mappo_gatres-edge` | 20 | OK | | 12.8716 | 0.948 | 257.4329 | 85.7107 | 0.201843 | 20 |

CI and per-family comparison are not computed here. Point `scripts/rliable_report.py --eval-dir results/eval_zeroshot_v6/<arm>/ngnb<N>` and `scripts/stability_report.py` at these directories instead — they already do IQM + stratified bootstrap per budget family (C3) and Wilson collapse rate.