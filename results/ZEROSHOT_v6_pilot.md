# Zero-shot topology transfer — wave v4

Readout: `greedy (reported, never gates)`. Episodes per checkpoint: 5. Trained at n_gnb=5, area_size=500 m, floor.mode=`none` (read from the config, not assumed).

`central-dqn` / `central-ppo` have `obs_dim = n_gnb * 8` fixed at training time, so CANNOT_RUN outside n_gnb=5 is a structural property of the architecture and is reported as a result, not as missing data. Every such row carries its reason, so **cannot be run** is never confused with **was not attempted**; the status is decided from the architecture, not from which files happen to exist.

Aggregate throughput grows with cell count no matter what the policy does, so **throughput per gNB** is the column to read; the aggregate is printed beside it. `retention` is per-gNB throughput relative to the same checkpoints at n_gnb=5.

Two arms are reported in full (integritas #3). `fixed-area` keeps area_size at 500 m so raising n_gnb also raises coupling strength; `const-density` scales area_size as sqrt(n/5) so density matches training. The env docstring (envs/network_slicing_env.py) is explicit that the first arm confounds agent count with coupling strength — it is kept because the v3 report used it.


## reference

| algo | n_gnb | status | reason | thr/gNB (Mbps) | retention | thr agg (Mbps) | sla_satisfaction_pct | embb_p5_mbps | n seeds |
|---|---|---|---|---|---|---|---|---|---|
| `gnn-madqn_gatres` | 5 | OK | | 11.9232 | 1.000 | 59.6160 | 79.8969 | 1.731195 | 1 |

## fixed-area

| algo | n_gnb | status | reason | thr/gNB (Mbps) | retention | thr agg (Mbps) | sla_satisfaction_pct | embb_p5_mbps | n seeds |
|---|---|---|---|---|---|---|---|---|---|
| `gnn-madqn_gatres` | 10 | OK | | 8.2055 | 0.688 | 82.0547 | 56.8076 | 1.281166 | 1 |
| `gnn-madqn_gatres` | 20 | OK | | 6.1575 | 0.516 | 123.1492 | 44.3747 | 1.056436 | 1 |

## const-density

| algo | n_gnb | status | reason | thr/gNB (Mbps) | retention | thr agg (Mbps) | sla_satisfaction_pct | embb_p5_mbps | n seeds |
|---|---|---|---|---|---|---|---|---|---|
| `gnn-madqn_gatres` | 10 | OK | | 10.4659 | 0.878 | 104.6587 | 70.7881 | 1.680837 | 1 |
| `gnn-madqn_gatres` | 20 | OK | | 11.3121 | 0.949 | 226.2420 | 76.4445 | 1.898182 | 1 |

CI and per-family comparison are not computed here. Point `scripts/rliable_report.py --eval-dir results/eval_zeroshot_v4/<arm>/ngnb<N>` and `scripts/stability_report.py` at these directories instead — they already do IQM + stratified bootstrap per budget family (C3) and Wilson collapse rate.