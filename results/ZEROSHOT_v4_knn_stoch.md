# Zero-shot topology transfer — wave v4

Readout: `stochastic (P3 primary)`. Episodes per checkpoint: 150. Trained at n_gnb=5, area_size=500 m, floor.mode=`none` (read from the config, not assumed).

`central-dqn` / `central-ppo` have `obs_dim = n_gnb * 8` fixed at training time, so CANNOT_RUN outside n_gnb=5 is a structural property of the architecture and is reported as a result, not as missing data.

Aggregate throughput grows with cell count no matter what the policy does, so **throughput per gNB** is the column to read; the aggregate is printed beside it. `retention` is per-gNB throughput relative to the same checkpoints at n_gnb=5.

Two arms are reported in full (integritas #3). `fixed-area` keeps area_size at 500 m so raising n_gnb also raises coupling strength; `const-density` scales area_size as sqrt(n/5) so density matches training. The env docstring (envs/network_slicing_env.py) is explicit that the first arm confounds agent count with coupling strength — it is kept because the v3 report used it.


## reference

| algo | n_gnb | status | thr/gNB (Mbps) | retention | thr agg (Mbps) | sla_satisfaction_pct | embb_p5_mbps | n seeds |
|---|---|---|---|---|---|---|---|---|
| `mlp-knn-ppo` | 5 | OK | 13.6387 | 1.000 | 68.1937 | 90.2615 | 0.000004 | 5 |

## fixed-area

| algo | n_gnb | status | thr/gNB (Mbps) | retention | thr agg (Mbps) | sla_satisfaction_pct | embb_p5_mbps | n seeds |
|---|---|---|---|---|---|---|---|---|
| `mlp-knn-ppo` | 10 | OK | 11.1920 | 0.821 | 111.9200 | 75.5544 | 0.000002 | 5 |
| `mlp-knn-ppo` | 20 | OK | 8.0688 | 0.592 | 161.3753 | 56.3788 | 0.000001 | 5 |

## const-density

| algo | n_gnb | status | thr/gNB (Mbps) | retention | thr agg (Mbps) | sla_satisfaction_pct | embb_p5_mbps | n seeds |
|---|---|---|---|---|---|---|---|---|
| `mlp-knn-ppo` | 10 | OK | 13.1166 | 0.962 | 131.1660 | 87.1623 | 0.000004 | 5 |
| `mlp-knn-ppo` | 20 | OK | 12.9431 | 0.949 | 258.8614 | 86.1432 | 0.000003 | 5 |

CI and per-family comparison are not computed here. Point `scripts/rliable_report.py --eval-dir results/eval_zeroshot_v4/<arm>/ngnb<N>` and `scripts/stability_report.py` at these directories instead — they already do IQM + stratified bootstrap per budget family (C3) and Wilson collapse rate.