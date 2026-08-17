# rliable Report — IQM + Stratified Bootstrap 95% CI

Readout: `primary per family — sampled for PPO (P3), argmax for DQN (2026-08-16)`.

Tag filter: `_v4`. DQN (200K steps) and PPO (1M steps) families are never pooled. A CI overlap means 'comparable', not 'proposed wins' — report accordingly.


## gnn-madqn_gat vs idqn, central-dqn (DQN family)

- **timely_throughput_mbps** (higher is better):
  - `gnn-madqn_gat` (proposed): IQM=149.7642  95% CI=[112.3098, 160.6364]
  - `idqn`: IQM=155.5314  95% CI=[93.6940, 171.0865]
    -> vs `idqn`: COMPARABLE (CIs overlap)
    -> P(`gnn-madqn_gat` > `idqn`) = 0.360  95% CI=[0.000, 0.760]
    -> performance profile @ tau=median(idqn)=153.2841: P(`gnn-madqn_gat` at least as good as tau)=0.400  P(`idqn` at least as good as tau)=0.400
- **sla_satisfaction_pct** (higher is better):
  - `gnn-madqn_gat` (proposed): IQM=52.3964  95% CI=[40.9692, 55.7714]
  - `idqn`: IQM=54.1715  95% CI=[34.8386, 59.2470]
    -> vs `idqn`: COMPARABLE (CIs overlap)
    -> P(`gnn-madqn_gat` > `idqn`) = 0.360  95% CI=[0.000, 0.760]
    -> performance profile @ tau=median(idqn)=53.5095: P(`gnn-madqn_gat` at least as good as tau)=0.400  P(`idqn` at least as good as tau)=0.400
- **embb_p5_mbps** (higher is better):
  - `gnn-madqn_gat` (proposed): IQM=0.2177  95% CI=[0.0000, 0.8692]
  - `idqn`: IQM=0.5684  95% CI=[0.0000, 0.9501]
    -> vs `idqn`: COMPARABLE (CIs overlap)
    -> P(`gnn-madqn_gat` > `idqn`) = 0.520  95% CI=[0.200, 0.880]
    -> performance profile @ tau=median(idqn)=0.7875: P(`gnn-madqn_gat` at least as good as tau)=0.200  P(`idqn` at least as good as tau)=0.400
- **jains_fairness** (higher is better):
  - `gnn-madqn_gat` (proposed): IQM=0.6819  95% CI=[0.5437, 0.8410]
  - `idqn`: IQM=0.5057  95% CI=[0.2821, 0.8448]
    -> vs `idqn`: COMPARABLE (CIs overlap)
    -> P(`gnn-madqn_gat` > `idqn`) = 0.760  95% CI=[0.360, 1.000]
    -> performance profile @ tau=median(idqn)=0.5128: P(`gnn-madqn_gat` at least as good as tau)=1.000  P(`idqn` at least as good as tau)=0.400
- **urllc_delay_p99** (lower is better):
  - `gnn-madqn_gat` (proposed): IQM=10.0000  95% CI=[10.0000, 10.0000]
  - `idqn`: IQM=10.0000  95% CI=[10.0000, 10.0000]
    -> vs `idqn`: COMPARABLE (CIs overlap)
    -> P(`gnn-madqn_gat` > `idqn`) = 0.500  95% CI=[0.500, 0.500]
    -> performance profile @ tau=median(idqn)=10.0000: P(`gnn-madqn_gat` at least as good as tau)=0.000  P(`idqn` at least as good as tau)=0.000

## gnn-madqn_sage vs idqn, central-dqn (DQN family)

- **timely_throughput_mbps** (higher is better):
  - `gnn-madqn_sage` (proposed): IQM=152.7799  95% CI=[145.8508, 158.8534]
  - `idqn`: IQM=155.5314  95% CI=[93.6940, 171.0865]
    -> vs `idqn`: COMPARABLE (CIs overlap)
    -> P(`gnn-madqn_sage` > `idqn`) = 0.440  95% CI=[0.040, 0.840]
    -> performance profile @ tau=median(idqn)=153.2841: P(`gnn-madqn_sage` at least as good as tau)=0.400  P(`idqn` at least as good as tau)=0.400
- **sla_satisfaction_pct** (higher is better):
  - `gnn-madqn_sage` (proposed): IQM=53.3221  95% CI=[51.1699, 55.1942]
  - `idqn`: IQM=54.1715  95% CI=[34.8386, 59.2470]
    -> vs `idqn`: COMPARABLE (CIs overlap)
    -> P(`gnn-madqn_sage` > `idqn`) = 0.440  95% CI=[0.080, 0.800]
    -> performance profile @ tau=median(idqn)=53.5095: P(`gnn-madqn_sage` at least as good as tau)=0.400  P(`idqn` at least as good as tau)=0.400
- **embb_p5_mbps** (higher is better):
  - `gnn-madqn_sage` (proposed): IQM=0.7058  95% CI=[0.2058, 0.9405]
  - `idqn`: IQM=0.5684  95% CI=[0.0000, 0.9501]
    -> vs `idqn`: COMPARABLE (CIs overlap)
    -> P(`gnn-madqn_sage` > `idqn`) = 0.560  95% CI=[0.160, 0.920]
    -> performance profile @ tau=median(idqn)=0.7875: P(`gnn-madqn_sage` at least as good as tau)=0.400  P(`idqn` at least as good as tau)=0.400
- **jains_fairness** (higher is better):
  - `gnn-madqn_sage` (proposed): IQM=0.5482  95% CI=[0.5179, 0.6105]
  - `idqn`: IQM=0.5057  95% CI=[0.2821, 0.8448]
    -> vs `idqn`: COMPARABLE (CIs overlap)
    -> P(`gnn-madqn_sage` > `idqn`) = 0.680  95% CI=[0.280, 1.000]
    -> performance profile @ tau=median(idqn)=0.5128: P(`gnn-madqn_sage` at least as good as tau)=0.800  P(`idqn` at least as good as tau)=0.400
- **urllc_delay_p99** (lower is better):
  - `gnn-madqn_sage` (proposed): IQM=10.0000  95% CI=[10.0000, 10.0000]
  - `idqn`: IQM=10.0000  95% CI=[10.0000, 10.0000]
    -> vs `idqn`: COMPARABLE (CIs overlap)
    -> P(`gnn-madqn_sage` > `idqn`) = 0.500  95% CI=[0.500, 0.500]
    -> performance profile @ tau=median(idqn)=10.0000: P(`gnn-madqn_sage` at least as good as tau)=0.000  P(`idqn` at least as good as tau)=0.000

## gnn-mappo_gat vs ippo, central-ppo (PPO family)

- **timely_throughput_mbps** (higher is better):
  - `gnn-mappo_gat` (proposed): IQM=156.1094  95% CI=[153.4092, 160.6974]
  - `ippo`: IQM=165.4195  95% CI=[164.7180, 166.1332]
    -> vs `ippo`: proposed WORSE (CIs disjoint)
    -> P(`gnn-mappo_gat` > `ippo`) = 0.000  95% CI=[0.000, 0.000]
    -> performance profile @ tau=median(ippo)=165.3046: P(`gnn-mappo_gat` at least as good as tau)=0.000  P(`ippo` at least as good as tau)=0.400
- **sla_satisfaction_pct** (higher is better):
  - `gnn-mappo_gat` (proposed): IQM=54.7729  95% CI=[54.0141, 56.3075]
  - `ippo`: IQM=57.5278  95% CI=[57.2983, 57.7900]
    -> vs `ippo`: proposed WORSE (CIs disjoint)
    -> P(`gnn-mappo_gat` > `ippo`) = 0.000  95% CI=[0.000, 0.000]
    -> performance profile @ tau=median(ippo)=57.4587: P(`gnn-mappo_gat` at least as good as tau)=0.000  P(`ippo` at least as good as tau)=0.400
- **embb_p5_mbps** (higher is better):
  - `gnn-mappo_gat` (proposed): IQM=0.0000  95% CI=[0.0000, 0.2112]
  - `ippo`: IQM=0.0000  95% CI=[0.0000, 0.0000]
    -> vs `ippo`: proposed BETTER (CIs disjoint)
    -> P(`gnn-mappo_gat` > `ippo`) = 1.000  95% CI=[1.000, 1.000]
    -> performance profile @ tau=median(ippo)=0.0000: P(`gnn-mappo_gat` at least as good as tau)=1.000  P(`ippo` at least as good as tau)=0.400
- **jains_fairness** (higher is better):
  - `gnn-mappo_gat` (proposed): IQM=0.4072  95% CI=[0.3913, 0.4280]
  - `ippo`: IQM=0.3408  95% CI=[0.3357, 0.3482]
    -> vs `ippo`: proposed BETTER (CIs disjoint)
    -> P(`gnn-mappo_gat` > `ippo`) = 1.000  95% CI=[1.000, 1.000]
    -> performance profile @ tau=median(ippo)=0.3414: P(`gnn-mappo_gat` at least as good as tau)=1.000  P(`ippo` at least as good as tau)=0.400
- **urllc_delay_p99** (lower is better):
  - `gnn-mappo_gat` (proposed): IQM=10.0000  95% CI=[10.0000, 10.0000]
  - `ippo`: IQM=10.0000  95% CI=[10.0000, 10.0000]
    -> vs `ippo`: COMPARABLE (CIs overlap)
    -> P(`gnn-mappo_gat` > `ippo`) = 0.500  95% CI=[0.500, 0.500]
    -> performance profile @ tau=median(ippo)=10.0000: P(`gnn-mappo_gat` at least as good as tau)=0.000  P(`ippo` at least as good as tau)=0.000

## gnn-mappo_sage vs ippo, central-ppo (PPO family)

- **timely_throughput_mbps** (higher is better):
  - `gnn-mappo_sage` (proposed): IQM=163.5297  95% CI=[157.7164, 172.6823]
  - `ippo`: IQM=165.4195  95% CI=[164.7180, 166.1332]
    -> vs `ippo`: COMPARABLE (CIs overlap)
    -> P(`gnn-mappo_sage` > `ippo`) = 0.400  95% CI=[0.000, 0.800]
    -> performance profile @ tau=median(ippo)=165.3046: P(`gnn-mappo_sage` at least as good as tau)=0.400  P(`ippo` at least as good as tau)=0.400
- **sla_satisfaction_pct** (higher is better):
  - `gnn-mappo_sage` (proposed): IQM=57.0780  95% CI=[55.1592, 60.0973]
  - `ippo`: IQM=57.5278  95% CI=[57.2983, 57.7900]
    -> vs `ippo`: COMPARABLE (CIs overlap)
    -> P(`gnn-mappo_sage` > `ippo`) = 0.400  95% CI=[0.000, 0.800]
    -> performance profile @ tau=median(ippo)=57.4587: P(`gnn-mappo_sage` at least as good as tau)=0.400  P(`ippo` at least as good as tau)=0.400
- **embb_p5_mbps** (higher is better):
  - `gnn-mappo_sage` (proposed): IQM=0.0000  95% CI=[0.0000, 0.0000]
  - `ippo`: IQM=0.0000  95% CI=[0.0000, 0.0000]
    -> vs `ippo`: COMPARABLE (CIs overlap)
    -> P(`gnn-mappo_sage` > `ippo`) = 0.600  95% CI=[0.200, 1.000]
    -> performance profile @ tau=median(ippo)=0.0000: P(`gnn-mappo_sage` at least as good as tau)=0.600  P(`ippo` at least as good as tau)=0.400
- **jains_fairness** (higher is better):
  - `gnn-mappo_sage` (proposed): IQM=0.3801  95% CI=[0.3358, 0.4069]
  - `ippo`: IQM=0.3408  95% CI=[0.3357, 0.3482]
    -> vs `ippo`: COMPARABLE (CIs overlap)
    -> P(`gnn-mappo_sage` > `ippo`) = 0.760  95% CI=[0.360, 1.000]
    -> performance profile @ tau=median(ippo)=0.3414: P(`gnn-mappo_sage` at least as good as tau)=0.800  P(`ippo` at least as good as tau)=0.400
- **urllc_delay_p99** (lower is better):
  - `gnn-mappo_sage` (proposed): IQM=10.0000  95% CI=[10.0000, 10.0000]
  - `ippo`: IQM=10.0000  95% CI=[10.0000, 10.0000]
    -> vs `ippo`: COMPARABLE (CIs overlap)
    -> P(`gnn-mappo_sage` > `ippo`) = 0.500  95% CI=[0.500, 0.500]
    -> performance profile @ tau=median(ippo)=10.0000: P(`gnn-mappo_sage` at least as good as tau)=0.000  P(`ippo` at least as good as tau)=0.000