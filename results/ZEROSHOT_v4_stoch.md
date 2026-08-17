# Zero-shot topology transfer — wave v4

> **VOID FOR THE DQN FAMILY — do not cite this file.** The `central-dqn`, `idqn`, `gnn-madqn_gat` and `gnn-madqn_sage` rows were produced by the
> epsilon=1.0 readout, i.e. uniform random actions rather than the trained policy
> (`results/quarantine_eps1.0/README.md`). The source CSVs have been quarantined, so this
> file cannot be regenerated as-is and is kept only as a record of what was reported before
> the fault was found. Valid replacement: `results/ZEROSHOT_v4_primary.md`.

Readout: `stochastic (P3 primary)`. Episodes per checkpoint: 150. Trained at n_gnb=5, area_size=500 m, floor.mode=`none` (read from the config, not assumed).

`central-dqn` / `central-ppo` have `obs_dim = n_gnb * 8` fixed at training time, so CANNOT_RUN outside n_gnb=5 is a structural property of the architecture and is reported as a result, not as missing data.

Aggregate throughput grows with cell count no matter what the policy does, so **throughput per gNB** is the column to read; the aggregate is printed beside it. `retention` is per-gNB throughput relative to the same checkpoints at n_gnb=5.

Two arms are reported in full (integritas #3). `fixed-area` keeps area_size at 500 m so raising n_gnb also raises coupling strength; `const-density` scales area_size as sqrt(n/5) so density matches training. The env docstring (envs/network_slicing_env.py) is explicit that the first arm confounds agent count with coupling strength — it is kept because the v3 report used it.


## reference

| algo | n_gnb | status | thr/gNB (Mbps) | retention | thr agg (Mbps) | sla_satisfaction_pct | embb_p5_mbps | n seeds |
|---|---|---|---|---|---|---|---|---|
| `central-dqn` | 5 | OK | 12.3742 | 1.000 | 61.8710 | 82.4237 | 0.026477 | 5 |
| `central-ppo` | 5 | OK | 12.8539 | 1.000 | 64.2696 | 85.1345 | 0.433748 | 5 |
| `gnn-madqn_gat` | 5 | OK | 13.4181 | 1.000 | 67.0903 | 88.8302 | 0.000006 | 5 |
| `gnn-madqn_sage` | 5 | OK | 13.4160 | 1.000 | 67.0799 | 88.7986 | 0.000005 | 5 |
| `gnn-mappo_gat` | 5 | OK | 13.5934 | 1.000 | 67.9670 | 90.0813 | 0.280535 | 5 |
| `gnn-mappo_sage` | 5 | OK | 13.7880 | 1.000 | 68.9401 | 91.1192 | 0.000004 | 5 |
| `idqn` | 5 | OK | 13.4127 | 1.000 | 67.0633 | 88.7897 | 0.000005 | 5 |
| `ippo` | 5 | OK | 13.7224 | 1.000 | 68.6121 | 90.6415 | 0.000003 | 5 |

## fixed-area

| algo | n_gnb | status | thr/gNB (Mbps) | retention | thr agg (Mbps) | sla_satisfaction_pct | embb_p5_mbps | n seeds |
|---|---|---|---|---|---|---|---|---|
| `central-dqn` | 10 | OK | 9.7410 | 0.787 | 97.4098 | 66.4433 | 0.034171 | 5 |
| `central-dqn` | 20 | OK | 6.7817 | 0.548 | 135.6336 | 48.1701 | 0.016562 | 5 |
| `central-ppo` | 10 | CANNOT_RUN | - | - | - | - | - | 0 |
| `central-ppo` | 20 | CANNOT_RUN | - | - | - | - | - | 0 |
| `gnn-madqn_gat` | 10 | OK | 10.9729 | 0.818 | 109.7292 | 74.1272 | 0.000003 | 5 |
| `gnn-madqn_gat` | 20 | OK | 7.9224 | 0.590 | 158.4472 | 55.3861 | 0.000002 | 5 |
| `gnn-madqn_sage` | 10 | OK | 10.9716 | 0.818 | 109.7162 | 74.1109 | 0.000003 | 5 |
| `gnn-madqn_sage` | 20 | OK | 7.9200 | 0.590 | 158.4005 | 55.3741 | 0.000002 | 5 |
| `gnn-mappo_gat` | 10 | OK | 11.0263 | 0.811 | 110.2631 | 74.6510 | 0.165752 | 5 |
| `gnn-mappo_gat` | 20 | OK | 7.8366 | 0.576 | 156.7319 | 55.0128 | 0.063375 | 5 |
| `gnn-mappo_sage` | 10 | OK | 11.4192 | 0.828 | 114.1925 | 76.9632 | 0.000002 | 5 |
| `gnn-mappo_sage` | 20 | OK | 8.2209 | 0.596 | 164.4181 | 57.3712 | 0.000001 | 5 |
| `idqn` | 10 | OK | 10.9687 | 0.818 | 109.6868 | 74.0985 | 0.000003 | 5 |
| `idqn` | 20 | OK | 7.9218 | 0.591 | 158.4356 | 55.3856 | 0.000002 | 5 |
| `ippo` | 10 | OK | 11.3386 | 0.826 | 113.3859 | 76.3181 | 0.000001 | 5 |
| `ippo` | 20 | OK | 8.2706 | 0.603 | 165.4117 | 57.5307 | 0.000001 | 5 |

## const-density

| algo | n_gnb | status | thr/gNB (Mbps) | retention | thr agg (Mbps) | sla_satisfaction_pct | embb_p5_mbps | n seeds |
|---|---|---|---|---|---|---|---|---|
| `central-dqn` | 10 | OK | 11.8597 | 0.958 | 118.5972 | 79.3270 | 0.019069 | 5 |
| `central-dqn` | 20 | OK | 11.6762 | 0.944 | 233.5249 | 78.2652 | 0.023334 | 5 |
| `central-ppo` | 10 | CANNOT_RUN | - | - | - | - | - | 0 |
| `central-ppo` | 20 | CANNOT_RUN | - | - | - | - | - | 0 |
| `gnn-madqn_gat` | 10 | OK | 12.9051 | 0.962 | 129.0508 | 85.7712 | 0.000005 | 5 |
| `gnn-madqn_gat` | 20 | OK | 12.7388 | 0.949 | 254.7760 | 84.8170 | 0.000004 | 5 |
| `gnn-madqn_sage` | 10 | OK | 12.9067 | 0.962 | 129.0668 | 85.7808 | 0.000005 | 5 |
| `gnn-madqn_sage` | 20 | OK | 12.7395 | 0.950 | 254.7897 | 84.8219 | 0.000004 | 5 |
| `gnn-mappo_gat` | 10 | OK | 13.0208 | 0.958 | 130.2084 | 86.6814 | 0.251634 | 5 |
| `gnn-mappo_gat` | 20 | OK | 12.8109 | 0.942 | 256.2184 | 85.4597 | 0.209675 | 5 |
| `gnn-mappo_sage` | 10 | OK | 13.2680 | 0.962 | 132.6798 | 88.0692 | 0.000003 | 5 |
| `gnn-mappo_sage` | 20 | OK | 13.0961 | 0.950 | 261.9223 | 87.0505 | 0.000003 | 5 |
| `idqn` | 10 | OK | 12.9093 | 0.962 | 129.0933 | 85.8034 | 0.000005 | 5 |
| `idqn` | 20 | OK | 12.7412 | 0.950 | 254.8248 | 84.8264 | 0.000004 | 5 |
| `ippo` | 10 | OK | 13.2260 | 0.964 | 132.2604 | 87.7054 | 0.000002 | 5 |
| `ippo` | 20 | OK | 13.0910 | 0.954 | 261.8203 | 86.8944 | 0.000002 | 5 |

CI and per-family comparison are not computed here. Point `scripts/rliable_report.py --eval-dir results/eval_zeroshot_v4/<arm>/ngnb<N>` and `scripts/stability_report.py` at these directories instead — they already do IQM + stratified bootstrap per budget family (C3) and Wilson collapse rate.