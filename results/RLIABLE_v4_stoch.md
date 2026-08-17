# rliable Report — IQM + Stratified Bootstrap 95% CI

> **VOID FOR THE DQN FAMILY — do not cite this file.** The `central-dqn`, `idqn`, `gnn-madqn_gat` and `gnn-madqn_sage` rows were produced by the
> epsilon=1.0 readout, i.e. uniform random actions rather than the trained policy
> (`results/quarantine_eps1.0/README.md`). The source CSVs have been quarantined, so this
> file cannot be regenerated as-is and is kept only as a record of what was reported before
> the fault was found. Valid replacement: `results/RLIABLE_v4_primary.md`.

Readout: `stochastic (P3 primary)`.

Tag filter: `_v4`. DQN (200K steps) and PPO (1M steps) families are never pooled. A CI overlap means 'comparable', not 'proposed wins' — report accordingly.


## gnn-madqn_gat vs idqn, central-dqn (DQN family)

- **timely_throughput_mbps** (higher is better):
  - `gnn-madqn_gat` (proposed): IQM=67.0816  95% CI=[67.0724, 67.1221]
  - `idqn`: IQM=67.0774  95% CI=[67.0028, 67.1032]
  - `central-dqn`: IQM=61.8649  95% CI=[61.8159, 61.9354]
    -> vs `idqn`: COMPARABLE (CIs overlap)
    -> vs `central-dqn`: proposed BETTER (CIs disjoint)
    -> P(`gnn-madqn_gat` > `idqn`) = 0.560  95% CI=[0.160, 0.921]
    -> P(`gnn-madqn_gat` > `central-dqn`) = 1.000  95% CI=[1.000, 1.000]
    -> performance profile @ tau=median(idqn)=67.0831: P(`gnn-madqn_gat` at least as good as tau)=0.400  P(`idqn` at least as good as tau)=0.400
    -> performance profile @ tau=median(central-dqn)=61.8611: P(`gnn-madqn_gat` at least as good as tau)=1.000  P(`central-dqn` at least as good as tau)=0.400
- **sla_satisfaction_pct** (higher is better):
  - `gnn-madqn_gat` (proposed): IQM=88.8265  95% CI=[88.8062, 88.8603]
  - `idqn`: IQM=88.8123  95% CI=[88.7217, 88.8248]
  - `central-dqn`: IQM=82.4129  95% CI=[82.3714, 82.4937]
    -> vs `idqn`: COMPARABLE (CIs overlap)
    -> vs `central-dqn`: proposed BETTER (CIs disjoint)
    -> P(`gnn-madqn_gat` > `idqn`) = 0.720  95% CI=[0.320, 1.000]
    -> P(`gnn-madqn_gat` > `central-dqn`) = 1.000  95% CI=[1.000, 1.000]
    -> performance profile @ tau=median(idqn)=88.8202: P(`gnn-madqn_gat` at least as good as tau)=0.800  P(`idqn` at least as good as tau)=0.400
    -> performance profile @ tau=median(central-dqn)=82.4036: P(`gnn-madqn_gat` at least as good as tau)=1.000  P(`central-dqn` at least as good as tau)=0.400
- **embb_p5_mbps** (higher is better):
  - `gnn-madqn_gat` (proposed): IQM=0.0000  95% CI=[0.0000, 0.0000]
  - `idqn`: IQM=0.0000  95% CI=[0.0000, 0.0000]
  - `central-dqn`: IQM=0.0246  95% CI=[0.0084, 0.0460]
    -> vs `idqn`: COMPARABLE (CIs overlap)
    -> vs `central-dqn`: proposed WORSE (CIs disjoint)
    -> P(`gnn-madqn_gat` > `idqn`) = 0.800  95% CI=[0.440, 1.000]
    -> P(`gnn-madqn_gat` > `central-dqn`) = 0.000  95% CI=[0.000, 0.000]
    -> performance profile @ tau=median(idqn)=0.0000: P(`gnn-madqn_gat` at least as good as tau)=0.800  P(`idqn` at least as good as tau)=0.400
    -> performance profile @ tau=median(central-dqn)=0.0280: P(`gnn-madqn_gat` at least as good as tau)=0.000  P(`central-dqn` at least as good as tau)=0.400
- **jains_fairness** (higher is better):
  - `gnn-madqn_gat` (proposed): IQM=0.5391  95% CI=[0.5374, 0.5403]
  - `idqn`: IQM=0.5390  95% CI=[0.5387, 0.5393]
  - `central-dqn`: IQM=0.6517  95% CI=[0.6510, 0.6524]
    -> vs `idqn`: COMPARABLE (CIs overlap)
    -> vs `central-dqn`: proposed WORSE (CIs disjoint)
    -> P(`gnn-madqn_gat` > `idqn`) = 0.520  95% CI=[0.120, 0.920]
    -> P(`gnn-madqn_gat` > `central-dqn`) = 0.000  95% CI=[0.000, 0.000]
    -> performance profile @ tau=median(idqn)=0.5389: P(`gnn-madqn_gat` at least as good as tau)=0.600  P(`idqn` at least as good as tau)=0.400
    -> performance profile @ tau=median(central-dqn)=0.6515: P(`gnn-madqn_gat` at least as good as tau)=0.000  P(`central-dqn` at least as good as tau)=0.400
- **urllc_delay_p99** (lower is better):
  - `gnn-madqn_gat` (proposed): IQM=8.7333  95% CI=[8.6578, 8.7400]
  - `idqn`: IQM=8.7598  95% CI=[8.6267, 8.8089]
  - `central-dqn`: IQM=8.9800  95% CI=[8.9511, 9.0000]
    -> vs `idqn`: COMPARABLE (CIs overlap)
    -> vs `central-dqn`: proposed BETTER (CIs disjoint)
    -> P(`gnn-madqn_gat` > `idqn`) = 0.640  95% CI=[0.200, 1.000]
    -> P(`gnn-madqn_gat` > `central-dqn`) = 1.000  95% CI=[1.000, 1.000]
    -> performance profile @ tau=median(idqn)=8.7729: P(`gnn-madqn_gat` at least as good as tau)=1.000  P(`idqn` at least as good as tau)=0.400
    -> performance profile @ tau=median(central-dqn)=8.9800: P(`gnn-madqn_gat` at least as good as tau)=1.000  P(`central-dqn` at least as good as tau)=0.400

## gnn-madqn_sage vs idqn, central-dqn (DQN family)

- **timely_throughput_mbps** (higher is better):
  - `gnn-madqn_sage` (proposed): IQM=67.0865  95% CI=[67.0283, 67.1199]
  - `idqn`: IQM=67.0774  95% CI=[67.0028, 67.1032]
  - `central-dqn`: IQM=61.8649  95% CI=[61.8159, 61.9354]
    -> vs `idqn`: COMPARABLE (CIs overlap)
    -> vs `central-dqn`: proposed BETTER (CIs disjoint)
    -> P(`gnn-madqn_sage` > `idqn`) = 0.640  95% CI=[0.240, 0.960]
    -> P(`gnn-madqn_sage` > `central-dqn`) = 1.000  95% CI=[1.000, 1.000]
    -> performance profile @ tau=median(idqn)=67.0831: P(`gnn-madqn_sage` at least as good as tau)=0.600  P(`idqn` at least as good as tau)=0.400
    -> performance profile @ tau=median(central-dqn)=61.8611: P(`gnn-madqn_sage` at least as good as tau)=1.000  P(`central-dqn` at least as good as tau)=0.400
- **sla_satisfaction_pct** (higher is better):
  - `gnn-madqn_sage` (proposed): IQM=88.7984  95% CI=[88.7446, 88.8546]
  - `idqn`: IQM=88.8123  95% CI=[88.7217, 88.8248]
  - `central-dqn`: IQM=82.4129  95% CI=[82.3714, 82.4937]
    -> vs `idqn`: COMPARABLE (CIs overlap)
    -> vs `central-dqn`: proposed BETTER (CIs disjoint)
    -> P(`gnn-madqn_sage` > `idqn`) = 0.560  95% CI=[0.200, 0.880]
    -> P(`gnn-madqn_sage` > `central-dqn`) = 1.000  95% CI=[1.000, 1.000]
    -> performance profile @ tau=median(idqn)=88.8202: P(`gnn-madqn_sage` at least as good as tau)=0.400  P(`idqn` at least as good as tau)=0.400
    -> performance profile @ tau=median(central-dqn)=82.4036: P(`gnn-madqn_sage` at least as good as tau)=1.000  P(`central-dqn` at least as good as tau)=0.400
- **embb_p5_mbps** (higher is better):
  - `gnn-madqn_sage` (proposed): IQM=0.0000  95% CI=[0.0000, 0.0000]
  - `idqn`: IQM=0.0000  95% CI=[0.0000, 0.0000]
  - `central-dqn`: IQM=0.0246  95% CI=[0.0084, 0.0460]
    -> vs `idqn`: COMPARABLE (CIs overlap)
    -> vs `central-dqn`: proposed WORSE (CIs disjoint)
    -> P(`gnn-madqn_sage` > `idqn`) = 0.320  95% CI=[0.000, 0.720]
    -> P(`gnn-madqn_sage` > `central-dqn`) = 0.000  95% CI=[0.000, 0.000]
    -> performance profile @ tau=median(idqn)=0.0000: P(`gnn-madqn_sage` at least as good as tau)=0.200  P(`idqn` at least as good as tau)=0.400
    -> performance profile @ tau=median(central-dqn)=0.0280: P(`gnn-madqn_sage` at least as good as tau)=0.000  P(`central-dqn` at least as good as tau)=0.400
- **jains_fairness** (higher is better):
  - `gnn-madqn_sage` (proposed): IQM=0.5388  95% CI=[0.5380, 0.5395]
  - `idqn`: IQM=0.5390  95% CI=[0.5387, 0.5393]
  - `central-dqn`: IQM=0.6517  95% CI=[0.6510, 0.6524]
    -> vs `idqn`: COMPARABLE (CIs overlap)
    -> vs `central-dqn`: proposed WORSE (CIs disjoint)
    -> P(`gnn-madqn_sage` > `idqn`) = 0.320  95% CI=[0.000, 0.720]
    -> P(`gnn-madqn_sage` > `central-dqn`) = 0.000  95% CI=[0.000, 0.000]
    -> performance profile @ tau=median(idqn)=0.5389: P(`gnn-madqn_sage` at least as good as tau)=0.400  P(`idqn` at least as good as tau)=0.400
    -> performance profile @ tau=median(central-dqn)=0.6515: P(`gnn-madqn_sage` at least as good as tau)=0.000  P(`central-dqn` at least as good as tau)=0.400
- **urllc_delay_p99** (lower is better):
  - `gnn-madqn_sage` (proposed): IQM=8.7666  95% CI=[8.6933, 8.8222]
  - `idqn`: IQM=8.7598  95% CI=[8.6267, 8.8089]
  - `central-dqn`: IQM=8.9800  95% CI=[8.9511, 9.0000]
    -> vs `idqn`: COMPARABLE (CIs overlap)
    -> vs `central-dqn`: proposed BETTER (CIs disjoint)
    -> P(`gnn-madqn_sage` > `idqn`) = 0.460  95% CI=[0.120, 0.840]
    -> P(`gnn-madqn_sage` > `central-dqn`) = 1.000  95% CI=[1.000, 1.000]
    -> performance profile @ tau=median(idqn)=8.7729: P(`gnn-madqn_sage` at least as good as tau)=0.600  P(`idqn` at least as good as tau)=0.400
    -> performance profile @ tau=median(central-dqn)=8.9800: P(`gnn-madqn_sage` at least as good as tau)=1.000  P(`central-dqn` at least as good as tau)=0.400

## gnn-mappo_gat vs ippo, central-ppo (PPO family)

- **timely_throughput_mbps** (higher is better):
  - `gnn-mappo_gat` (proposed): IQM=67.9627  95% CI=[67.3482, 68.5825]
  - `ippo`: IQM=68.5563  95% CI=[68.2290, 69.0997]
  - `central-ppo`: IQM=64.2827  95% CI=[64.1337, 64.3955]
    -> vs `ippo`: COMPARABLE (CIs overlap)
    -> vs `central-ppo`: proposed BETTER (CIs disjoint)
    -> P(`gnn-mappo_gat` > `ippo`) = 0.240  95% CI=[0.000, 0.640]
    -> P(`gnn-mappo_gat` > `central-ppo`) = 1.000  95% CI=[1.000, 1.000]
    -> performance profile @ tau=median(ippo)=68.4658: P(`gnn-mappo_gat` at least as good as tau)=0.400  P(`ippo` at least as good as tau)=0.400
    -> performance profile @ tau=median(central-ppo)=64.2606: P(`gnn-mappo_gat` at least as good as tau)=1.000  P(`central-ppo` at least as good as tau)=0.400
- **sla_satisfaction_pct** (higher is better):
  - `gnn-mappo_gat` (proposed): IQM=90.0586  95% CI=[89.1603, 90.9524]
  - `ippo`: IQM=90.5660  95% CI=[90.1481, 91.2718]
  - `central-ppo`: IQM=85.1478  95% CI=[84.9784, 85.2832]
    -> vs `ippo`: COMPARABLE (CIs overlap)
    -> vs `central-ppo`: proposed BETTER (CIs disjoint)
    -> P(`gnn-mappo_gat` > `ippo`) = 0.360  95% CI=[0.040, 0.760]
    -> P(`gnn-mappo_gat` > `central-ppo`) = 1.000  95% CI=[1.000, 1.000]
    -> performance profile @ tau=median(ippo)=90.4575: P(`gnn-mappo_gat` at least as good as tau)=0.400  P(`ippo` at least as good as tau)=0.400
    -> performance profile @ tau=median(central-ppo)=85.1160: P(`gnn-mappo_gat` at least as good as tau)=1.000  P(`central-ppo` at least as good as tau)=0.400
- **embb_p5_mbps** (higher is better):
  - `gnn-mappo_gat` (proposed): IQM=0.0000  95% CI=[0.0000, 0.9351]
  - `ippo`: IQM=0.0000  95% CI=[0.0000, 0.0000]
  - `central-ppo`: IQM=0.3470  95% CI=[0.0716, 0.8771]
    -> vs `ippo`: proposed BETTER (CIs disjoint)
    -> vs `central-ppo`: COMPARABLE (CIs overlap)
    -> P(`gnn-mappo_gat` > `ippo`) = 1.000  95% CI=[1.000, 1.000]
    -> P(`gnn-mappo_gat` > `central-ppo`) = 0.200  95% CI=[0.000, 0.600]
    -> performance profile @ tau=median(ippo)=0.0000: P(`gnn-mappo_gat` at least as good as tau)=1.000  P(`ippo` at least as good as tau)=0.400
    -> performance profile @ tau=median(central-ppo)=0.4505: P(`gnn-mappo_gat` at least as good as tau)=0.200  P(`central-ppo` at least as good as tau)=0.400
- **jains_fairness** (higher is better):
  - `gnn-mappo_gat` (proposed): IQM=0.5313  95% CI=[0.5182, 0.5543]
  - `ippo`: IQM=0.4876  95% CI=[0.4800, 0.4936]
  - `central-ppo`: IQM=0.6673  95% CI=[0.6530, 0.6689]
    -> vs `ippo`: proposed BETTER (CIs disjoint)
    -> vs `central-ppo`: proposed WORSE (CIs disjoint)
    -> P(`gnn-mappo_gat` > `ippo`) = 1.000  95% CI=[1.000, 1.000]
    -> P(`gnn-mappo_gat` > `central-ppo`) = 0.000  95% CI=[0.000, 0.000]
    -> performance profile @ tau=median(ippo)=0.4910: P(`gnn-mappo_gat` at least as good as tau)=1.000  P(`ippo` at least as good as tau)=0.400
    -> performance profile @ tau=median(central-ppo)=0.6682: P(`gnn-mappo_gat` at least as good as tau)=0.000  P(`central-ppo` at least as good as tau)=0.400
- **urllc_delay_p99** (lower is better):
  - `gnn-mappo_gat` (proposed): IQM=8.9522  95% CI=[8.6022, 9.5333]
  - `ippo`: IQM=8.2533  95% CI=[8.2089, 8.2933]
  - `central-ppo`: IQM=7.9622  95% CI=[7.9067, 8.0889]
    -> vs `ippo`: proposed WORSE (CIs disjoint)
    -> vs `central-ppo`: proposed WORSE (CIs disjoint)
    -> P(`gnn-mappo_gat` > `ippo`) = 0.000  95% CI=[0.000, 0.000]
    -> P(`gnn-mappo_gat` > `central-ppo`) = 0.000  95% CI=[0.000, 0.000]
    -> performance profile @ tau=median(ippo)=8.2533: P(`gnn-mappo_gat` at least as good as tau)=0.000  P(`ippo` at least as good as tau)=0.400
    -> performance profile @ tau=median(central-ppo)=7.9533: P(`gnn-mappo_gat` at least as good as tau)=0.000  P(`central-ppo` at least as good as tau)=0.400

## gnn-mappo_sage vs ippo, central-ppo (PPO family)

- **timely_throughput_mbps** (higher is better):
  - `gnn-mappo_sage` (proposed): IQM=68.8968  95% CI=[67.7281, 70.1377]
  - `ippo`: IQM=68.5563  95% CI=[68.2290, 69.0997]
  - `central-ppo`: IQM=64.2827  95% CI=[64.1337, 64.3955]
    -> vs `ippo`: COMPARABLE (CIs overlap)
    -> vs `central-ppo`: proposed BETTER (CIs disjoint)
    -> P(`gnn-mappo_sage` > `ippo`) = 0.560  95% CI=[0.160, 1.000]
    -> P(`gnn-mappo_sage` > `central-ppo`) = 1.000  95% CI=[1.000, 1.000]
    -> performance profile @ tau=median(ippo)=68.4658: P(`gnn-mappo_sage` at least as good as tau)=0.600  P(`ippo` at least as good as tau)=0.400
    -> performance profile @ tau=median(central-ppo)=64.2606: P(`gnn-mappo_sage` at least as good as tau)=1.000  P(`central-ppo` at least as good as tau)=0.400
- **sla_satisfaction_pct** (higher is better):
  - `gnn-mappo_sage` (proposed): IQM=91.0831  95% CI=[89.6116, 92.5848]
  - `ippo`: IQM=90.5660  95% CI=[90.1481, 91.2718]
  - `central-ppo`: IQM=85.1478  95% CI=[84.9784, 85.2832]
    -> vs `ippo`: COMPARABLE (CIs overlap)
    -> vs `central-ppo`: proposed BETTER (CIs disjoint)
    -> P(`gnn-mappo_sage` > `ippo`) = 0.560  95% CI=[0.160, 1.000]
    -> P(`gnn-mappo_sage` > `central-ppo`) = 1.000  95% CI=[1.000, 1.000]
    -> performance profile @ tau=median(ippo)=90.4575: P(`gnn-mappo_sage` at least as good as tau)=0.600  P(`ippo` at least as good as tau)=0.400
    -> performance profile @ tau=median(central-ppo)=85.1160: P(`gnn-mappo_sage` at least as good as tau)=1.000  P(`central-ppo` at least as good as tau)=0.400
- **embb_p5_mbps** (higher is better):
  - `gnn-mappo_sage` (proposed): IQM=0.0000  95% CI=[0.0000, 0.0000]
  - `ippo`: IQM=0.0000  95% CI=[0.0000, 0.0000]
  - `central-ppo`: IQM=0.3470  95% CI=[0.0716, 0.8771]
    -> vs `ippo`: COMPARABLE (CIs overlap)
    -> vs `central-ppo`: proposed WORSE (CIs disjoint)
    -> P(`gnn-mappo_sage` > `ippo`) = 0.800  95% CI=[0.400, 1.000]
    -> P(`gnn-mappo_sage` > `central-ppo`) = 0.000  95% CI=[0.000, 0.000]
    -> performance profile @ tau=median(ippo)=0.0000: P(`gnn-mappo_sage` at least as good as tau)=0.800  P(`ippo` at least as good as tau)=0.400
    -> performance profile @ tau=median(central-ppo)=0.4505: P(`gnn-mappo_sage` at least as good as tau)=0.000  P(`central-ppo` at least as good as tau)=0.400
- **jains_fairness** (higher is better):
  - `gnn-mappo_sage` (proposed): IQM=0.5105  95% CI=[0.4651, 0.5285]
  - `ippo`: IQM=0.4876  95% CI=[0.4800, 0.4936]
  - `central-ppo`: IQM=0.6673  95% CI=[0.6530, 0.6689]
    -> vs `ippo`: COMPARABLE (CIs overlap)
    -> vs `central-ppo`: proposed WORSE (CIs disjoint)
    -> P(`gnn-mappo_sage` > `ippo`) = 0.800  95% CI=[0.400, 1.000]
    -> P(`gnn-mappo_sage` > `central-ppo`) = 0.000  95% CI=[0.000, 0.000]
    -> performance profile @ tau=median(ippo)=0.4910: P(`gnn-mappo_sage` at least as good as tau)=0.800  P(`ippo` at least as good as tau)=0.400
    -> performance profile @ tau=median(central-ppo)=0.6682: P(`gnn-mappo_sage` at least as good as tau)=0.000  P(`central-ppo` at least as good as tau)=0.400
- **urllc_delay_p99** (lower is better):
  - `gnn-mappo_sage` (proposed): IQM=8.5867  95% CI=[8.3645, 8.7082]
  - `ippo`: IQM=8.2533  95% CI=[8.2089, 8.2933]
  - `central-ppo`: IQM=7.9622  95% CI=[7.9067, 8.0889]
    -> vs `ippo`: proposed WORSE (CIs disjoint)
    -> vs `central-ppo`: proposed WORSE (CIs disjoint)
    -> P(`gnn-mappo_sage` > `ippo`) = 0.040  95% CI=[0.000, 0.240]
    -> P(`gnn-mappo_sage` > `central-ppo`) = 0.000  95% CI=[0.000, 0.000]
    -> performance profile @ tau=median(ippo)=8.2533: P(`gnn-mappo_sage` at least as good as tau)=0.000  P(`ippo` at least as good as tau)=0.400
    -> performance profile @ tau=median(central-ppo)=7.9533: P(`gnn-mappo_sage` at least as good as tau)=0.000  P(`central-ppo` at least as good as tau)=0.400