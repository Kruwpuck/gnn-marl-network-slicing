# rliable Report — IQM + Stratified Bootstrap 95% CI

Tag filter: `_floorstatic`. DQN (200K steps) and PPO (1M steps) families are never pooled. A CI overlap means 'comparable', not 'proposed wins' — report accordingly.


## gnn-madqn_gat vs idqn, central-dqn (DQN family)

- **timely_throughput_mbps**: skipped (need >=2 seeds of eval data for `gnn-madqn_gat` and at least one baseline)
- **sla_satisfaction_pct**: skipped (need >=2 seeds of eval data for `gnn-madqn_gat` and at least one baseline)
- **embb_p5_mbps**: skipped (need >=2 seeds of eval data for `gnn-madqn_gat` and at least one baseline)
- **jains_fairness**: skipped (need >=2 seeds of eval data for `gnn-madqn_gat` and at least one baseline)
- **urllc_delay_p99**: skipped (need >=2 seeds of eval data for `gnn-madqn_gat` and at least one baseline)

## gnn-madqn_sage vs idqn, central-dqn (DQN family)

- **timely_throughput_mbps** (higher is better):
  - `gnn-madqn_sage` (proposed): IQM=30.7232  95% CI=[30.5250, 30.8923]
  - `idqn`: IQM=30.6474  95% CI=[30.2156, 30.7498]
    -> vs `idqn`: COMPARABLE (CIs overlap)
    -> P(`gnn-madqn_sage` > `idqn`) = 0.720  95% CI=[0.360, 1.000]
    -> performance profile @ tau=median(idqn)=30.7487: P(`gnn-madqn_sage` at least as good as tau)=0.600  P(`idqn` at least as good as tau)=0.400
- **sla_satisfaction_pct** (higher is better):
  - `gnn-madqn_sage` (proposed): IQM=96.1617  95% CI=[95.5016, 96.7107]
  - `idqn`: IQM=95.9084  95% CI=[94.4588, 96.2707]
    -> vs `idqn`: COMPARABLE (CIs overlap)
    -> P(`gnn-madqn_sage` > `idqn`) = 0.740  95% CI=[0.360, 1.000]
    -> performance profile @ tau=median(idqn)=96.2679: P(`gnn-madqn_sage` at least as good as tau)=0.600  P(`idqn` at least as good as tau)=0.400
- **embb_p5_mbps** (higher is better):
  - `gnn-madqn_sage` (proposed): IQM=1.7833  95% CI=[1.7740, 1.7948]
  - `idqn`: IQM=1.7837  95% CI=[0.5942, 1.7987]
    -> vs `idqn`: COMPARABLE (CIs overlap)
    -> P(`gnn-madqn_sage` > `idqn`) = 0.520  95% CI=[0.160, 0.880]
    -> performance profile @ tau=median(idqn)=1.7842: P(`gnn-madqn_sage` at least as good as tau)=0.600  P(`idqn` at least as good as tau)=0.400
- **jains_fairness** (higher is better):
  - `gnn-madqn_sage` (proposed): IQM=0.5183  95% CI=[0.5165, 0.5213]
  - `idqn`: IQM=0.5224  95% CI=[0.5171, 0.7106]
    -> vs `idqn`: COMPARABLE (CIs overlap)
    -> P(`gnn-madqn_sage` > `idqn`) = 0.280  95% CI=[0.000, 0.640]
    -> performance profile @ tau=median(idqn)=0.5175: P(`gnn-madqn_sage` at least as good as tau)=0.400  P(`idqn` at least as good as tau)=0.400
- **urllc_delay_p99** (lower is better):
  - `gnn-madqn_sage` (proposed): IQM=3.4444  95% CI=[3.3333, 3.5667]
  - `idqn`: IQM=3.3447  95% CI=[2.4336, 3.6000]
    -> vs `idqn`: COMPARABLE (CIs overlap)
    -> P(`gnn-madqn_sage` > `idqn`) = 0.540  95% CI=[0.160, 1.000]
    -> performance profile @ tau=median(idqn)=3.6000: P(`gnn-madqn_sage` at least as good as tau)=0.800  P(`idqn` at least as good as tau)=0.400

## gnn-mappo_gat vs ippo, central-ppo (PPO family)

- **timely_throughput_mbps** (higher is better):
  - `gnn-mappo_gat` (proposed): IQM=30.6065  95% CI=[30.4704, 30.8452]
  - `central-ppo`: IQM=30.7394  95% CI=[30.5500, 30.8894]
    -> vs `central-ppo`: COMPARABLE (CIs overlap)
    -> P(`gnn-mappo_gat` > `central-ppo`) = 0.280  95% CI=[0.000, 0.640]
    -> performance profile @ tau=median(central-ppo)=30.7636: P(`gnn-mappo_gat` at least as good as tau)=0.200  P(`central-ppo` at least as good as tau)=0.400
- **sla_satisfaction_pct** (higher is better):
  - `gnn-mappo_gat` (proposed): IQM=95.7637  95% CI=[95.2862, 96.5654]
  - `central-ppo`: IQM=96.1925  95% CI=[95.5767, 96.6930]
    -> vs `central-ppo`: COMPARABLE (CIs overlap)
    -> P(`gnn-mappo_gat` > `central-ppo`) = 0.360  95% CI=[0.000, 0.760]
    -> performance profile @ tau=median(central-ppo)=96.2595: P(`gnn-mappo_gat` at least as good as tau)=0.400  P(`central-ppo` at least as good as tau)=0.400
- **embb_p5_mbps** (higher is better):
  - `gnn-mappo_gat` (proposed): IQM=1.7842  95% CI=[0.5907, 1.8035]
  - `central-ppo`: IQM=1.7867  95% CI=[1.7764, 1.7943]
    -> vs `central-ppo`: COMPARABLE (CIs overlap)
    -> P(`gnn-mappo_gat` > `central-ppo`) = 0.520  95% CI=[0.120, 0.920]
    -> performance profile @ tau=median(central-ppo)=1.7836: P(`gnn-mappo_gat` at least as good as tau)=0.600  P(`central-ppo` at least as good as tau)=0.400
- **jains_fairness** (higher is better):
  - `gnn-mappo_gat` (proposed): IQM=0.5244  95% CI=[0.5169, 0.8437]
  - `central-ppo`: IQM=0.5198  95% CI=[0.5168, 0.5245]
    -> vs `central-ppo`: COMPARABLE (CIs overlap)
    -> P(`gnn-mappo_gat` > `central-ppo`) = 0.600  95% CI=[0.200, 0.920]
    -> performance profile @ tau=median(central-ppo)=0.5201: P(`gnn-mappo_gat` at least as good as tau)=0.600  P(`central-ppo` at least as good as tau)=0.400
- **urllc_delay_p99** (lower is better):
  - `gnn-mappo_gat` (proposed): IQM=3.2444  95% CI=[2.8560, 3.5667]
  - `central-ppo`: IQM=3.3889  95% CI=[3.2222, 3.5222]
    -> vs `central-ppo`: COMPARABLE (CIs overlap)
    -> P(`gnn-mappo_gat` > `central-ppo`) = 0.580  95% CI=[0.180, 0.920]
    -> performance profile @ tau=median(central-ppo)=3.3333: P(`gnn-mappo_gat` at least as good as tau)=0.400  P(`central-ppo` at least as good as tau)=0.200

## gnn-mappo_sage vs ippo, central-ppo (PPO family)

- **timely_throughput_mbps**: skipped (need >=2 seeds of eval data for `gnn-mappo_sage` and at least one baseline)
- **sla_satisfaction_pct**: skipped (need >=2 seeds of eval data for `gnn-mappo_sage` and at least one baseline)
- **embb_p5_mbps**: skipped (need >=2 seeds of eval data for `gnn-mappo_sage` and at least one baseline)
- **jains_fairness**: skipped (need >=2 seeds of eval data for `gnn-mappo_sage` and at least one baseline)
- **urllc_delay_p99**: skipped (need >=2 seeds of eval data for `gnn-mappo_sage` and at least one baseline)