# Zero-shot topology transfer — wave v4

Readout: `greedy (reported, never gates)`. Episodes per checkpoint: 150. Trained at n_gnb=5, area_size=500 m, floor.mode=`none` (read from the config, not assumed).

`central-dqn` / `central-ppo` have `obs_dim = n_gnb * 8` fixed at training time, so CANNOT_RUN outside n_gnb=5 is a structural property of the architecture and is reported as a result, not as missing data.

Aggregate throughput grows with cell count no matter what the policy does, so **throughput per gNB** is the column to read; the aggregate is printed beside it. `retention` is per-gNB throughput relative to the same checkpoints at n_gnb=5.

Two arms are reported in full (integritas #3). `fixed-area` keeps area_size at 500 m so raising n_gnb also raises coupling strength; `const-density` scales area_size as sqrt(n/5) so density matches training. The env docstring (envs/network_slicing_env.py) is explicit that the first arm confounds agent count with coupling strength — it is kept because the v3 report used it.


## reference

| algo | n_gnb | status | thr/gNB (Mbps) | retention | thr agg (Mbps) | sla_satisfaction_pct | embb_p5_mbps | n seeds |
|---|---|---|---|---|---|---|---|---|
| `mlp-knn-ppo` | 5 | OK | 8.4225 | 1.000 | 42.1123 | 58.0398 | 0.685794 | 5 |

## fixed-area

| algo | n_gnb | status | thr/gNB (Mbps) | retention | thr agg (Mbps) | sla_satisfaction_pct | embb_p5_mbps | n seeds |
|---|---|---|---|---|---|---|---|---|
| `mlp-knn-ppo` | 10 | OK | 7.0568 | 0.838 | 70.5675 | 49.7828 | 0.510145 | 5 |
| `mlp-knn-ppo` | 20 | OK | 5.2975 | 0.629 | 105.9498 | 38.9647 | 0.205656 | 5 |

## const-density

| algo | n_gnb | status | thr/gNB (Mbps) | retention | thr agg (Mbps) | sla_satisfaction_pct | embb_p5_mbps | n seeds |
|---|---|---|---|---|---|---|---|---|
| `mlp-knn-ppo` | 10 | OK | 8.2115 | 0.975 | 82.1151 | 56.7669 | 0.643306 | 5 |
| `mlp-knn-ppo` | 20 | OK | 8.1192 | 0.964 | 162.3836 | 56.2347 | 0.565265 | 5 |

CI and per-family comparison are not computed here. Point `scripts/rliable_report.py --eval-dir results/eval_zeroshot_v4/<arm>/ngnb<N>` and `scripts/stability_report.py` at these directories instead — they already do IQM + stratified bootstrap per budget family (C3) and Wilson collapse rate.