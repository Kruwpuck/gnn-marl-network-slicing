# rliable Report — IQM + Stratified Bootstrap 95% CI

Tag filter: `_floornone`. DQN (200K steps) and PPO (1M steps) families are never pooled. A CI overlap means 'comparable', not 'proposed wins' — report accordingly.


## gnn-madqn_gat vs idqn, central-dqn (DQN family)

- **timely_throughput_mbps**: skipped (need >=2 seeds of eval data for `gnn-madqn_gat` and at least one baseline)
- **sla_satisfaction_pct**: skipped (need >=2 seeds of eval data for `gnn-madqn_gat` and at least one baseline)
- **embb_p5_mbps**: skipped (need >=2 seeds of eval data for `gnn-madqn_gat` and at least one baseline)
- **jains_fairness**: skipped (need >=2 seeds of eval data for `gnn-madqn_gat` and at least one baseline)
- **urllc_delay_p99**: skipped (need >=2 seeds of eval data for `gnn-madqn_gat` and at least one baseline)

## gnn-madqn_sage vs idqn, central-dqn (DQN family)

- **timely_throughput_mbps** (higher is better):
  - `gnn-madqn_sage` (proposed): IQM=30.7212  95% CI=[30.5127, 30.8947]
  - `idqn`: IQM=30.7499  95% CI=[30.5823, 30.8290]
    -> vs `idqn`: COMPARABLE (CIs overlap)
    -> P(`gnn-madqn_sage` > `idqn`) = 0.640  95% CI=[0.200, 1.000]
    -> performance profile @ tau=median(idqn)=30.7500: P(`gnn-madqn_sage` at least as good as tau)=0.600  P(`idqn` at least as good as tau)=0.400
- **sla_satisfaction_pct** (higher is better):
  - `gnn-madqn_sage` (proposed): IQM=96.1494  95% CI=[95.4436, 96.7196]
  - `idqn`: IQM=96.2704  95% CI=[95.6790, 96.5209]
    -> vs `idqn`: COMPARABLE (CIs overlap)
    -> P(`gnn-madqn_sage` > `idqn`) = 0.640  95% CI=[0.200, 1.000]
    -> performance profile @ tau=median(idqn)=96.2709: P(`gnn-madqn_sage` at least as good as tau)=0.600  P(`idqn` at least as good as tau)=0.200
- **embb_p5_mbps** (higher is better):
  - `gnn-madqn_sage` (proposed): IQM=1.7843  95% CI=[1.7739, 1.7957]
  - `idqn`: IQM=1.7832  95% CI=[1.6841, 1.7967]
    -> vs `idqn`: COMPARABLE (CIs overlap)
    -> P(`gnn-madqn_sage` > `idqn`) = 0.520  95% CI=[0.120, 0.880]
    -> performance profile @ tau=median(idqn)=1.7826: P(`gnn-madqn_sage` at least as good as tau)=0.600  P(`idqn` at least as good as tau)=0.400
- **jains_fairness** (higher is better):
  - `gnn-madqn_sage` (proposed): IQM=0.5184  95% CI=[0.5165, 0.5238]
  - `idqn`: IQM=0.5175  95% CI=[0.5143, 0.5544]
    -> vs `idqn`: COMPARABLE (CIs overlap)
    -> P(`gnn-madqn_sage` > `idqn`) = 0.440  95% CI=[0.080, 0.840]
    -> performance profile @ tau=median(idqn)=0.5174: P(`gnn-madqn_sage` at least as good as tau)=0.400  P(`idqn` at least as good as tau)=0.400
- **urllc_delay_p99** (lower is better):
  - `gnn-madqn_sage` (proposed): IQM=3.4222  95% CI=[3.3333, 3.5667]
  - `idqn`: IQM=3.5778  95% CI=[3.1111, 3.6000]
    -> vs `idqn`: COMPARABLE (CIs overlap)
    -> P(`gnn-madqn_sage` > `idqn`) = 0.700  95% CI=[0.340, 1.000]
    -> performance profile @ tau=median(idqn)=3.6000: P(`gnn-madqn_sage` at least as good as tau)=0.800  P(`idqn` at least as good as tau)=0.400

## gnn-mappo_gat vs ippo, central-ppo (PPO family)

- **timely_throughput_mbps** (higher is better):
  - `gnn-mappo_gat` (proposed): IQM=30.7970  95% CI=[30.6053, 30.8923]
  - `central-ppo`: IQM=30.8078  95% CI=[30.6804, 30.8931]
    -> vs `central-ppo`: COMPARABLE (CIs overlap)
    -> P(`gnn-mappo_gat` > `central-ppo`) = 0.400  95% CI=[0.080, 0.800]
    -> performance profile @ tau=median(central-ppo)=30.8289: P(`gnn-mappo_gat` at least as good as tau)=0.400  P(`central-ppo` at least as good as tau)=0.400
- **sla_satisfaction_pct** (higher is better):
  - `gnn-mappo_gat` (proposed): IQM=96.4171  95% CI=[95.7355, 96.7107]
  - `central-ppo`: IQM=96.4293  95% CI=[96.0204, 96.7130]
    -> vs `central-ppo`: COMPARABLE (CIs overlap)
    -> P(`gnn-mappo_gat` > `central-ppo`) = 0.400  95% CI=[0.080, 0.800]
    -> performance profile @ tau=median(central-ppo)=96.4957: P(`gnn-mappo_gat` at least as good as tau)=0.400  P(`central-ppo` at least as good as tau)=0.400
- **embb_p5_mbps** (higher is better):
  - `gnn-mappo_gat` (proposed): IQM=1.7774  95% CI=[1.7506, 1.7843]
  - `central-ppo`: IQM=1.7806  95% CI=[1.7740, 1.7882]
    -> vs `central-ppo`: COMPARABLE (CIs overlap)
    -> P(`gnn-mappo_gat` > `central-ppo`) = 0.240  95% CI=[0.000, 0.640]
    -> performance profile @ tau=median(central-ppo)=1.7813: P(`gnn-mappo_gat` at least as good as tau)=0.400  P(`central-ppo` at least as good as tau)=0.400
- **jains_fairness** (higher is better):
  - `gnn-mappo_gat` (proposed): IQM=0.5171  95% CI=[0.5165, 0.6531]
  - `central-ppo`: IQM=0.5181  95% CI=[0.5165, 0.5216]
    -> vs `central-ppo`: COMPARABLE (CIs overlap)
    -> P(`gnn-mappo_gat` > `central-ppo`) = 0.520  95% CI=[0.200, 0.880]
    -> performance profile @ tau=median(central-ppo)=0.5182: P(`gnn-mappo_gat` at least as good as tau)=0.200  P(`central-ppo` at least as good as tau)=0.400
- **urllc_delay_p99** (lower is better):
  - `gnn-mappo_gat` (proposed): IQM=3.4222  95% CI=[3.0444, 3.6000]
  - `central-ppo`: IQM=3.3222  95% CI=[3.2778, 3.4444]
    -> vs `central-ppo`: COMPARABLE (CIs overlap)
    -> P(`gnn-mappo_gat` > `central-ppo`) = 0.360  95% CI=[0.040, 0.720]
    -> performance profile @ tau=median(central-ppo)=3.3333: P(`gnn-mappo_gat` at least as good as tau)=0.200  P(`central-ppo` at least as good as tau)=0.400

## gnn-mappo_sage vs ippo, central-ppo (PPO family)

- **timely_throughput_mbps**: skipped (need >=2 seeds of eval data for `gnn-mappo_sage` and at least one baseline)
- **sla_satisfaction_pct**: skipped (need >=2 seeds of eval data for `gnn-mappo_sage` and at least one baseline)
- **embb_p5_mbps**: skipped (need >=2 seeds of eval data for `gnn-mappo_sage` and at least one baseline)
- **jains_fairness**: skipped (need >=2 seeds of eval data for `gnn-mappo_sage` and at least one baseline)
- **urllc_delay_p99**: skipped (need >=2 seeds of eval data for `gnn-mappo_sage` and at least one baseline)