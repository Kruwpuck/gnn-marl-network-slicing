# rliable Report — IQM + Stratified Bootstrap 95% CI

Readout: `primary per family — sampled for PPO (P3), argmax for DQN (2026-08-16)`.

Tag filter: `_v4`. DQN (200K steps) and PPO (1M steps) families are never pooled. A CI overlap means 'comparable', not 'proposed wins' — report accordingly.


## gnn-madqn_gat vs idqn, central-dqn (DQN family)

- **timely_throughput_mbps** (higher is better):
  - `gnn-madqn_gat` (proposed): IQM=245.9315  95% CI=[223.4191, 251.9195]
  - `idqn`: IQM=248.1153  95% CI=[163.8379, 258.2880]
    -> vs `idqn`: COMPARABLE (CIs overlap)
    -> P(`gnn-madqn_gat` > `idqn`) = 0.440  95% CI=[0.080, 0.840]
    -> performance profile @ tau=median(idqn)=246.8018: P(`gnn-madqn_gat` at least as good as tau)=0.400  P(`idqn` at least as good as tau)=0.400
- **sla_satisfaction_pct** (higher is better):
  - `gnn-madqn_gat` (proposed): IQM=81.7667  95% CI=[75.2822, 83.5543]
  - `idqn`: IQM=82.4296  95% CI=[56.9988, 85.8934]
    -> vs `idqn`: COMPARABLE (CIs overlap)
    -> P(`gnn-madqn_gat` > `idqn`) = 0.440  95% CI=[0.080, 0.840]
    -> performance profile @ tau=median(idqn)=82.0460: P(`gnn-madqn_gat` at least as good as tau)=0.400  P(`idqn` at least as good as tau)=0.400
- **embb_p5_mbps** (higher is better):
  - `gnn-madqn_gat` (proposed): IQM=0.5544  95% CI=[0.0000, 1.6997]
  - `idqn`: IQM=1.1424  95% CI=[0.0000, 1.7194]
    -> vs `idqn`: COMPARABLE (CIs overlap)
    -> P(`gnn-madqn_gat` > `idqn`) = 0.440  95% CI=[0.040, 0.840]
    -> performance profile @ tau=median(idqn)=1.7093: P(`gnn-madqn_gat` at least as good as tau)=0.200  P(`idqn` at least as good as tau)=0.400
- **jains_fairness** (higher is better):
  - `gnn-madqn_gat` (proposed): IQM=0.6615  95% CI=[0.4834, 0.7800]
  - `idqn`: IQM=0.4452  95% CI=[0.3095, 0.8190]
    -> vs `idqn`: COMPARABLE (CIs overlap)
    -> P(`gnn-madqn_gat` > `idqn`) = 0.760  95% CI=[0.400, 1.000]
    -> performance profile @ tau=median(idqn)=0.4476: P(`gnn-madqn_gat` at least as good as tau)=1.000  P(`idqn` at least as good as tau)=0.400
- **urllc_delay_p99** (lower is better):
  - `gnn-madqn_gat` (proposed): IQM=10.0000  95% CI=[10.0000, 10.0000]
  - `idqn`: IQM=10.0000  95% CI=[9.9644, 10.0000]
    -> vs `idqn`: COMPARABLE (CIs overlap)
    -> P(`gnn-madqn_gat` > `idqn`) = 0.400  95% CI=[0.200, 0.500]
    -> performance profile @ tau=median(idqn)=10.0000: P(`gnn-madqn_gat` at least as good as tau)=0.000  P(`idqn` at least as good as tau)=0.200

## gnn-madqn_sage vs idqn, central-dqn (DQN family)

- **timely_throughput_mbps** (higher is better):
  - `gnn-madqn_sage` (proposed): IQM=246.1274  95% CI=[242.9433, 250.6327]
  - `idqn`: IQM=248.1153  95% CI=[163.8379, 258.2880]
    -> vs `idqn`: COMPARABLE (CIs overlap)
    -> P(`gnn-madqn_sage` > `idqn`) = 0.480  95% CI=[0.120, 0.840]
    -> performance profile @ tau=median(idqn)=246.8018: P(`gnn-madqn_sage` at least as good as tau)=0.600  P(`idqn` at least as good as tau)=0.400
- **sla_satisfaction_pct** (higher is better):
  - `gnn-madqn_sage` (proposed): IQM=81.8079  95% CI=[80.8758, 83.1618]
  - `idqn`: IQM=82.4296  95% CI=[56.9988, 85.8934]
    -> vs `idqn`: COMPARABLE (CIs overlap)
    -> P(`gnn-madqn_sage` > `idqn`) = 0.460  95% CI=[0.080, 0.840]
    -> performance profile @ tau=median(idqn)=82.0460: P(`gnn-madqn_sage` at least as good as tau)=0.400  P(`idqn` at least as good as tau)=0.400
- **embb_p5_mbps** (higher is better):
  - `gnn-madqn_sage` (proposed): IQM=1.5977  95% CI=[1.0235, 1.7151]
  - `idqn`: IQM=1.1424  95% CI=[0.0000, 1.7194]
    -> vs `idqn`: COMPARABLE (CIs overlap)
    -> P(`gnn-madqn_sage` > `idqn`) = 0.460  95% CI=[0.060, 0.860]
    -> performance profile @ tau=median(idqn)=1.7093: P(`gnn-madqn_sage` at least as good as tau)=0.200  P(`idqn` at least as good as tau)=0.400
- **jains_fairness** (higher is better):
  - `gnn-madqn_sage` (proposed): IQM=0.4753  95% CI=[0.4531, 0.5373]
  - `idqn`: IQM=0.4452  95% CI=[0.3095, 0.8190]
    -> vs `idqn`: COMPARABLE (CIs overlap)
    -> P(`gnn-madqn_sage` > `idqn`) = 0.720  95% CI=[0.320, 1.000]
    -> performance profile @ tau=median(idqn)=0.4476: P(`gnn-madqn_sage` at least as good as tau)=0.800  P(`idqn` at least as good as tau)=0.400
- **urllc_delay_p99** (lower is better):
  - `gnn-madqn_sage` (proposed): IQM=10.0000  95% CI=[10.0000, 10.0000]
  - `idqn`: IQM=10.0000  95% CI=[9.9644, 10.0000]
    -> vs `idqn`: COMPARABLE (CIs overlap)
    -> P(`gnn-madqn_sage` > `idqn`) = 0.400  95% CI=[0.200, 0.500]
    -> performance profile @ tau=median(idqn)=10.0000: P(`gnn-madqn_sage` at least as good as tau)=0.000  P(`idqn` at least as good as tau)=0.200

## gnn-mappo_gat vs ippo, central-ppo (PPO family)

- **timely_throughput_mbps** (higher is better):
  - `gnn-mappo_gat` (proposed): IQM=256.0938  95% CI=[254.2550, 258.4387]
  - `ippo`: IQM=261.5471  95% CI=[260.6273, 263.5085]
    -> vs `ippo`: proposed WORSE (CIs disjoint)
    -> P(`gnn-mappo_gat` > `ippo`) = 0.000  95% CI=[0.000, 0.000]
    -> performance profile @ tau=median(ippo)=261.1540: P(`gnn-mappo_gat` at least as good as tau)=0.000  P(`ippo` at least as good as tau)=0.400
- **sla_satisfaction_pct** (higher is better):
  - `gnn-mappo_gat` (proposed): IQM=85.3873  95% CI=[84.9417, 86.1401]
  - `ippo`: IQM=86.8130  95% CI=[86.4914, 87.4472]
    -> vs `ippo`: proposed WORSE (CIs disjoint)
    -> P(`gnn-mappo_gat` > `ippo`) = 0.000  95% CI=[0.000, 0.000]
    -> performance profile @ tau=median(ippo)=86.6895: P(`gnn-mappo_gat` at least as good as tau)=0.000  P(`ippo` at least as good as tau)=0.400
- **embb_p5_mbps** (higher is better):
  - `gnn-mappo_gat` (proposed): IQM=0.0000  95% CI=[0.0000, 0.6989]
  - `ippo`: IQM=0.0000  95% CI=[0.0000, 0.0000]
    -> vs `ippo`: proposed BETTER (CIs disjoint)
    -> P(`gnn-mappo_gat` > `ippo`) = 1.000  95% CI=[1.000, 1.000]
    -> performance profile @ tau=median(ippo)=0.0000: P(`gnn-mappo_gat` at least as good as tau)=1.000  P(`ippo` at least as good as tau)=0.400
- **jains_fairness** (higher is better):
  - `gnn-mappo_gat` (proposed): IQM=0.3537  95% CI=[0.3475, 0.3770]
  - `ippo`: IQM=0.3142  95% CI=[0.3084, 0.3195]
    -> vs `ippo`: proposed BETTER (CIs disjoint)
    -> P(`gnn-mappo_gat` > `ippo`) = 1.000  95% CI=[1.000, 1.000]
    -> performance profile @ tau=median(ippo)=0.3160: P(`gnn-mappo_gat` at least as good as tau)=1.000  P(`ippo` at least as good as tau)=0.400
- **urllc_delay_p99** (lower is better):
  - `gnn-mappo_gat` (proposed): IQM=10.0000  95% CI=[10.0000, 10.0000]
  - `ippo`: IQM=10.0000  95% CI=[9.9956, 10.0000]
    -> vs `ippo`: COMPARABLE (CIs overlap)
    -> P(`gnn-mappo_gat` > `ippo`) = 0.400  95% CI=[0.200, 0.500]
    -> performance profile @ tau=median(ippo)=10.0000: P(`gnn-mappo_gat` at least as good as tau)=0.000  P(`ippo` at least as good as tau)=0.200

## gnn-mappo_sage vs ippo, central-ppo (PPO family)

- **timely_throughput_mbps** (higher is better):
  - `gnn-mappo_sage` (proposed): IQM=261.7421  95% CI=[257.7141, 265.9732]
  - `ippo`: IQM=261.5471  95% CI=[260.6273, 263.5085]
    -> vs `ippo`: COMPARABLE (CIs overlap)
    -> P(`gnn-mappo_sage` > `ippo`) = 0.560  95% CI=[0.160, 1.000]
    -> performance profile @ tau=median(ippo)=261.1540: P(`gnn-mappo_sage` at least as good as tau)=0.600  P(`ippo` at least as good as tau)=0.400
- **sla_satisfaction_pct** (higher is better):
  - `gnn-mappo_sage` (proposed): IQM=87.0223  95% CI=[85.7340, 88.2845]
  - `ippo`: IQM=86.8130  95% CI=[86.4914, 87.4472]
    -> vs `ippo`: COMPARABLE (CIs overlap)
    -> P(`gnn-mappo_sage` > `ippo`) = 0.560  95% CI=[0.160, 1.000]
    -> performance profile @ tau=median(ippo)=86.6895: P(`gnn-mappo_sage` at least as good as tau)=0.600  P(`ippo` at least as good as tau)=0.400
- **embb_p5_mbps** (higher is better):
  - `gnn-mappo_sage` (proposed): IQM=0.0000  95% CI=[0.0000, 0.0000]
  - `ippo`: IQM=0.0000  95% CI=[0.0000, 0.0000]
    -> vs `ippo`: COMPARABLE (CIs overlap)
    -> P(`gnn-mappo_sage` > `ippo`) = 0.800  95% CI=[0.400, 1.000]
    -> performance profile @ tau=median(ippo)=0.0000: P(`gnn-mappo_sage` at least as good as tau)=0.800  P(`ippo` at least as good as tau)=0.400
- **jains_fairness** (higher is better):
  - `gnn-mappo_sage` (proposed): IQM=0.3292  95% CI=[0.2954, 0.3444]
  - `ippo`: IQM=0.3142  95% CI=[0.3084, 0.3195]
    -> vs `ippo`: COMPARABLE (CIs overlap)
    -> P(`gnn-mappo_sage` > `ippo`) = 0.800  95% CI=[0.400, 1.000]
    -> performance profile @ tau=median(ippo)=0.3160: P(`gnn-mappo_sage` at least as good as tau)=0.800  P(`ippo` at least as good as tau)=0.400
- **urllc_delay_p99** (lower is better):
  - `gnn-mappo_sage` (proposed): IQM=10.0000  95% CI=[9.9956, 10.0000]
  - `ippo`: IQM=10.0000  95% CI=[9.9956, 10.0000]
    -> vs `ippo`: COMPARABLE (CIs overlap)
    -> P(`gnn-mappo_sage` > `ippo`) = 0.500  95% CI=[0.300, 0.700]
    -> performance profile @ tau=median(ippo)=10.0000: P(`gnn-mappo_sage` at least as good as tau)=0.200  P(`ippo` at least as good as tau)=0.200