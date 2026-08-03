# rliable Report — IQM + Stratified Bootstrap 95% CI

Tag filter: `(main wave)`. DQN (200K steps) and PPO (1M steps) families are never pooled. A CI overlap means 'comparable', not 'proposed wins' — report accordingly.


## gnn-madqn_gat vs idqn, central-dqn (DQN family)

- **timely_throughput_mbps** (higher is better):
  - `gnn-madqn_gat` (proposed): IQM=30.7759  95% CI=[30.6291, 30.8883]
  - `idqn`: IQM=30.6608  95% CI=[30.4315, 30.7491]
  - `central-dqn`: IQM=30.6331  95% CI=[30.4747, 30.8640]
    -> vs `idqn`: COMPARABLE (CIs overlap)
    -> vs `central-dqn`: COMPARABLE (CIs overlap)
- **sla_satisfaction_pct** (higher is better):
  - `gnn-madqn_gat` (proposed): IQM=96.3216  95% CI=[95.8081, 96.6975]
  - `idqn`: IQM=95.9658  95% CI=[95.1676, 96.2680]
  - `central-dqn`: IQM=95.8487  95% CI=[95.2977, 96.6121]
    -> vs `idqn`: COMPARABLE (CIs overlap)
    -> vs `central-dqn`: COMPARABLE (CIs overlap)
- **embb_p5_mbps** (higher is better):
  - `gnn-madqn_gat` (proposed): IQM=1.2469  95% CI=[0.0642, 1.7844]
  - `idqn`: IQM=1.7816  95% CI=[0.5922, 1.7923]
  - `central-dqn`: IQM=1.2115  95% CI=[0.3377, 1.7878]
    -> vs `idqn`: COMPARABLE (CIs overlap)
    -> vs `central-dqn`: COMPARABLE (CIs overlap)
- **jains_fairness** (higher is better):
  - `gnn-madqn_gat` (proposed): IQM=0.5879  95% CI=[0.5169, 0.7507]
  - `idqn`: IQM=0.5183  95% CI=[0.5016, 0.5374]
  - `central-dqn`: IQM=0.5680  95% CI=[0.5177, 0.6537]
    -> vs `idqn`: COMPARABLE (CIs overlap)
    -> vs `central-dqn`: COMPARABLE (CIs overlap)
- **urllc_delay_p99** (lower is better):
  - `gnn-madqn_gat` (proposed): IQM=3.3222  95% CI=[3.1222, 3.3556]
  - `idqn`: IQM=3.5667  95% CI=[3.1667, 3.9333]
  - `central-dqn`: IQM=3.3222  95% CI=[3.0222, 3.5667]
    -> vs `idqn`: COMPARABLE (CIs overlap)
    -> vs `central-dqn`: COMPARABLE (CIs overlap)

## gnn-madqn_sage vs idqn, central-dqn (DQN family)

- **timely_throughput_mbps** (higher is better):
  - `gnn-madqn_sage` (proposed): IQM=30.7044  95% CI=[30.4178, 30.8928]
  - `idqn`: IQM=30.6608  95% CI=[30.4315, 30.7491]
  - `central-dqn`: IQM=30.6331  95% CI=[30.4747, 30.8640]
    -> vs `idqn`: COMPARABLE (CIs overlap)
    -> vs `central-dqn`: COMPARABLE (CIs overlap)
- **sla_satisfaction_pct** (higher is better):
  - `gnn-madqn_sage` (proposed): IQM=96.1091  95% CI=[95.1244, 96.7121]
  - `idqn`: IQM=95.9658  95% CI=[95.1676, 96.2680]
  - `central-dqn`: IQM=95.8487  95% CI=[95.2977, 96.6121]
    -> vs `idqn`: COMPARABLE (CIs overlap)
    -> vs `central-dqn`: COMPARABLE (CIs overlap)
- **embb_p5_mbps** (higher is better):
  - `gnn-madqn_sage` (proposed): IQM=1.7777  95% CI=[1.7727, 1.7992]
  - `idqn`: IQM=1.7816  95% CI=[0.5922, 1.7923]
  - `central-dqn`: IQM=1.2115  95% CI=[0.3377, 1.7878]
    -> vs `idqn`: COMPARABLE (CIs overlap)
    -> vs `central-dqn`: COMPARABLE (CIs overlap)
- **jains_fairness** (higher is better):
  - `gnn-madqn_sage` (proposed): IQM=0.5190  95% CI=[0.5166, 0.5264]
  - `idqn`: IQM=0.5183  95% CI=[0.5016, 0.5374]
  - `central-dqn`: IQM=0.5680  95% CI=[0.5177, 0.6537]
    -> vs `idqn`: COMPARABLE (CIs overlap)
    -> vs `central-dqn`: COMPARABLE (CIs overlap)
- **urllc_delay_p99** (lower is better):
  - `gnn-madqn_sage` (proposed): IQM=3.4222  95% CI=[3.0893, 3.6000]
  - `idqn`: IQM=3.5667  95% CI=[3.1667, 3.9333]
  - `central-dqn`: IQM=3.3222  95% CI=[3.0222, 3.5667]
    -> vs `idqn`: COMPARABLE (CIs overlap)
    -> vs `central-dqn`: COMPARABLE (CIs overlap)

## gnn-mappo_gat vs ippo, central-ppo (PPO family)

- **timely_throughput_mbps** (higher is better):
  - `gnn-mappo_gat` (proposed): IQM=30.4868  95% CI=[30.3903, 30.6816]
  - `ippo`: IQM=30.5941  95% CI=[30.5448, 30.8025]
  - `central-ppo`: IQM=30.8415  95% CI=[30.7160, 30.8930]
    -> vs `ippo`: COMPARABLE (CIs overlap)
    -> vs `central-ppo`: proposed WORSE (CIs disjoint)
- **sla_satisfaction_pct** (higher is better):
  - `gnn-mappo_gat` (proposed): IQM=95.3443  95% CI=[95.0122, 96.0202]
  - `ippo`: IQM=95.6967  95% CI=[95.5120, 96.4084]
  - `central-ppo`: IQM=96.5499  95% CI=[96.1371, 96.7125]
    -> vs `ippo`: COMPARABLE (CIs overlap)
    -> vs `central-ppo`: proposed WORSE (CIs disjoint)
- **embb_p5_mbps** (higher is better):
  - `gnn-mappo_gat` (proposed): IQM=1.7964  95% CI=[0.5948, 1.8085]
  - `ippo`: IQM=0.0000  95% CI=[0.0000, 0.0000]
  - `central-ppo`: IQM=1.7756  95% CI=[1.7740, 1.7811]
    -> vs `ippo`: proposed BETTER (CIs disjoint)
    -> vs `central-ppo`: COMPARABLE (CIs overlap)
- **jains_fairness** (higher is better):
  - `gnn-mappo_gat` (proposed): IQM=0.5259  95% CI=[0.5188, 0.8417]
  - `ippo`: IQM=0.6243  95% CI=[0.4059, 0.9964]
  - `central-ppo`: IQM=0.5170  95% CI=[0.5165, 0.5187]
    -> vs `ippo`: COMPARABLE (CIs overlap)
    -> vs `central-ppo`: proposed BETTER (CIs disjoint)
- **urllc_delay_p99** (lower is better):
  - `gnn-mappo_gat` (proposed): IQM=3.1449  95% CI=[2.9224, 3.5667]
  - `ippo`: IQM=3.1333  95% CI=[2.9000, 3.6333]
  - `central-ppo`: IQM=3.3778  95% CI=[3.3333, 3.5444]
    -> vs `ippo`: COMPARABLE (CIs overlap)
    -> vs `central-ppo`: COMPARABLE (CIs overlap)

## gnn-mappo_sage vs ippo, central-ppo (PPO family)

- **timely_throughput_mbps** (higher is better):
  - `gnn-mappo_sage` (proposed): IQM=30.7972  95% CI=[30.5660, 30.8928]
  - `ippo`: IQM=30.5941  95% CI=[30.5448, 30.8025]
  - `central-ppo`: IQM=30.8415  95% CI=[30.7160, 30.8930]
    -> vs `ippo`: COMPARABLE (CIs overlap)
    -> vs `central-ppo`: COMPARABLE (CIs overlap)
- **sla_satisfaction_pct** (higher is better):
  - `gnn-mappo_sage` (proposed): IQM=96.4173  95% CI=[95.6143, 96.7121]
  - `ippo`: IQM=95.6967  95% CI=[95.5120, 96.4084]
  - `central-ppo`: IQM=96.5499  95% CI=[96.1371, 96.7125]
    -> vs `ippo`: COMPARABLE (CIs overlap)
    -> vs `central-ppo`: COMPARABLE (CIs overlap)
- **embb_p5_mbps** (higher is better):
  - `gnn-mappo_sage` (proposed): IQM=1.7803  95% CI=[1.7727, 1.7985]
  - `ippo`: IQM=0.0000  95% CI=[0.0000, 0.0000]
  - `central-ppo`: IQM=1.7756  95% CI=[1.7740, 1.7811]
    -> vs `ippo`: proposed BETTER (CIs disjoint)
    -> vs `central-ppo`: COMPARABLE (CIs overlap)
- **jains_fairness** (higher is better):
  - `gnn-mappo_sage` (proposed): IQM=0.5172  95% CI=[0.5166, 0.5360]
  - `ippo`: IQM=0.6243  95% CI=[0.4059, 0.9964]
  - `central-ppo`: IQM=0.5170  95% CI=[0.5165, 0.5187]
    -> vs `ippo`: COMPARABLE (CIs overlap)
    -> vs `central-ppo`: COMPARABLE (CIs overlap)
- **urllc_delay_p99** (lower is better):
  - `gnn-mappo_sage` (proposed): IQM=3.4222  95% CI=[2.9560, 3.6000]
  - `ippo`: IQM=3.1333  95% CI=[2.9000, 3.6333]
  - `central-ppo`: IQM=3.3778  95% CI=[3.3333, 3.5444]
    -> vs `ippo`: COMPARABLE (CIs overlap)
    -> vs `central-ppo`: COMPARABLE (CIs overlap)