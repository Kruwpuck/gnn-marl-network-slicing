# Zero-shot topology transfer — wave v6

Readout: `greedy (reported, never gates)`. Episodes per checkpoint: 150. Trained at n_gnb=5, area_size=500 m, floor.mode=`none` (read from the config, not assumed).

`central-dqn` / `central-ppo` have `obs_dim = n_gnb * 8` fixed at training time, so CANNOT_RUN outside n_gnb=5 is a structural property of the architecture and is reported as a result, not as missing data. Every such row carries its reason, so **cannot be run** is never confused with **was not attempted**; the status is decided from the architecture, not from which files happen to exist.

Aggregate throughput grows with cell count no matter what the policy does, so **throughput per gNB** is the column to read; the aggregate is printed beside it. `retention` is per-gNB throughput relative to the same checkpoints at n_gnb=5.

Two arms are reported in full (integritas #3). `fixed-area` keeps area_size at 500 m so raising n_gnb also raises coupling strength; `const-density` scales area_size as sqrt(n/5) so density matches training. The env docstring (envs/network_slicing_env.py) is explicit that the first arm confounds agent count with coupling strength — it is kept because the v3 report used it.


## reference

| algo | n_gnb | status | reason | thr/gNB (Mbps) | retention | thr agg (Mbps) | sla_satisfaction_pct | embb_p5_mbps | n seeds |
|---|---|---|---|---|---|---|---|---|---|
| `gnn-madqn_gatedge` | 5 | OK | | 12.6018 | 1.000 | 63.0092 | 83.6798 | 0.633587 | 5 |
| `gnn-madqn_gatres` | 5 | OK | | 12.7891 | 1.000 | 63.9455 | 84.9607 | 0.760406 | 5 |
| `gnn-madqn_gatres-edge` | 5 | OK | | 12.8607 | 1.000 | 64.3035 | 85.2373 | 0.558908 | 5 |
| `gnn-mappo_gatedge` | 5 | OK | | 9.5519 | 1.000 | 47.7595 | 64.8710 | 1.057997 | 20 |
| `gnn-mappo_gatres` | 5 | OK | | 8.1978 | 1.000 | 40.9892 | 56.6035 | 1.744941 | 20 |
| `gnn-mappo_gatres-edge` | 5 | OK | | 7.4579 | 1.000 | 37.2895 | 52.0200 | 1.585852 | 20 |

## fixed-area

| algo | n_gnb | status | reason | thr/gNB (Mbps) | retention | thr agg (Mbps) | sla_satisfaction_pct | embb_p5_mbps | n seeds |
|---|---|---|---|---|---|---|---|---|---|
| `gnn-madqn_gatedge` | 10 | OK | | 9.7589 | 0.774 | 97.5888 | 66.3756 | 0.542217 | 5 |
| `gnn-madqn_gatedge` | 20 | OK | | 6.2981 | 0.500 | 125.9619 | 45.0248 | 0.320613 | 5 |
| `gnn-madqn_gatres` | 10 | OK | | 10.3057 | 0.806 | 103.0571 | 69.7881 | 0.675526 | 5 |
| `gnn-madqn_gatres` | 20 | OK | | 7.2680 | 0.568 | 145.3600 | 51.1026 | 0.488252 | 5 |
| `gnn-madqn_gatres-edge` | 10 | OK | | 10.3333 | 0.803 | 103.3329 | 69.9097 | 0.496475 | 5 |
| `gnn-madqn_gatres-edge` | 20 | OK | | 7.2658 | 0.565 | 145.3161 | 51.0199 | 0.174050 | 5 |
| `gnn-mappo_gatedge` | 10 | OK | | 7.7825 | 0.815 | 77.8250 | 54.0444 | 0.807203 | 20 |
| `gnn-mappo_gatedge` | 20 | OK | | 5.5993 | 0.586 | 111.9857 | 40.6069 | 0.591550 | 20 |
| `gnn-mappo_gatres` | 10 | OK | | 6.4620 | 0.788 | 64.6201 | 45.8918 | 1.364428 | 20 |
| `gnn-mappo_gatres` | 20 | OK | | 4.5677 | 0.557 | 91.3536 | 34.1549 | 0.889635 | 20 |
| `gnn-mappo_gatres-edge` | 10 | OK | | 5.9401 | 0.796 | 59.4006 | 42.6170 | 1.264354 | 20 |
| `gnn-mappo_gatres-edge` | 20 | OK | | 4.3949 | 0.589 | 87.8980 | 33.0097 | 0.814008 | 20 |

## const-density

| algo | n_gnb | status | reason | thr/gNB (Mbps) | retention | thr agg (Mbps) | sla_satisfaction_pct | embb_p5_mbps | n seeds |
|---|---|---|---|---|---|---|---|---|---|
| `gnn-madqn_gatedge` | 10 | OK | | 12.0428 | 0.956 | 120.4276 | 80.2848 | 0.594169 | 5 |
| `gnn-madqn_gatedge` | 20 | OK | | 11.8137 | 0.937 | 236.2735 | 78.9524 | 0.685787 | 5 |
| `gnn-madqn_gatres` | 10 | OK | | 12.3052 | 0.962 | 123.0523 | 81.9624 | 0.717637 | 5 |
| `gnn-madqn_gatres` | 20 | OK | | 12.1858 | 0.953 | 243.7151 | 81.2308 | 0.705019 | 5 |
| `gnn-madqn_gatres-edge` | 10 | OK | | 12.3571 | 0.961 | 123.5711 | 82.2012 | 0.661613 | 5 |
| `gnn-madqn_gatres-edge` | 20 | OK | | 12.1771 | 0.947 | 243.5429 | 81.0959 | 0.700558 | 5 |
| `gnn-mappo_gatedge` | 10 | OK | | 9.2894 | 0.973 | 92.8945 | 63.2719 | 0.943577 | 20 |
| `gnn-mappo_gatedge` | 20 | OK | | 9.2017 | 0.963 | 184.0345 | 62.7363 | 0.932800 | 20 |
| `gnn-mappo_gatres` | 10 | OK | | 7.8631 | 0.959 | 78.6311 | 54.5394 | 1.637705 | 20 |
| `gnn-mappo_gatres` | 20 | OK | | 7.7423 | 0.944 | 154.8456 | 53.7923 | 1.660967 | 20 |
| `gnn-mappo_gatres-edge` | 10 | OK | | 7.0819 | 0.950 | 70.8193 | 49.7130 | 1.489795 | 20 |
| `gnn-mappo_gatres-edge` | 20 | OK | | 6.8770 | 0.922 | 137.5402 | 48.4395 | 1.492164 | 20 |

CI and per-family comparison are not computed here. Point `scripts/rliable_report.py --eval-dir results/eval_zeroshot_v6/<arm>/ngnb<N>` and `scripts/stability_report.py` at these directories instead — they already do IQM + stratified bootstrap per budget family (C3) and Wilson collapse rate.