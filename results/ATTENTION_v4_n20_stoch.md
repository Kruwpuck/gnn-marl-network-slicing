# Attention vs Interference Mechanism

> **VOID FOR THE DQN FAMILY — do not cite this file.** The `gnn-madqn_gat` half of every table and ablation were produced by the
> epsilon=1.0 readout, i.e. uniform random actions rather than the trained policy
> (`results/quarantine_eps1.0/README.md`). The source CSVs have been quarantined, so this
> file cannot be regenerated as-is and is kept only as a record of what was reported before
> the fault was found. Valid replacement: the greedy pass `results/ATTENTION_v4_greedy.md` for the DQN half; the PPO half of this file is unaffected.

Readout: `stochastic (P3 primary)`. Checkpoints: `results/logs/gnn-*_gat_v4_seed*.pt`. Config: `configs/generated/floor_none_area_size1000.0_n_gnb20.yaml`.

n edges pooled across all episodes/seeds: 7600000

**Pooled**: Pearson r = -0.0676 (p=0), Spearman r = -0.2399 (p=0) -- attention weight (layer 2, single head) vs raw path-loss dB (env-native scale, not the /100 scaled value fed to the model).

**Per receiving node** (400000 node-steps with a defined rank correlation): mean rho = -0.5222, median = -0.6632, fraction rho < 0 = 0.824. The pooled number mixes edges from nodes with different attention scales and different neighbour sets, which can hide a node that consistently prefers its strongest interferer; the softmax is normalised per receiving node, so that is the level the mechanism claim lives at. A negative rho means more attention on lower path-loss, i.e. on the stronger interferer -- the direction the mechanism story predicts.

**Causal ablation** (mandatory): attention forced uniform over neighbors (zero the learned `att` parameter of both GATv2Conv layers -> softmax degenerates to 1/degree). Both arms draw the same action noise (`torch.manual_seed` per episode) so the difference is the ablation, not sampling luck. Correlation alone would be decoration without this.

Reported on three KPIs, not on `embb_p5_mbps` alone: at this operating point most checkpoints already sit at the cell-edge floor, and a KPI pinned near zero cannot degrade however much the ablation changes. `timely_throughput_mbps` and `sla_satisfaction_pct` still have headroom, so they are where a real causal effect would have to show up.

| algo | seed | KPI | normal | uniform-attn | degradation |
|---|---|---|---|---|---|
| `gnn-madqn_gat` | 42 | `embb_p5_mbps` | 0.000005 | 0.000004 | +0.000000 |
| `gnn-madqn_gat` | 42 | `timely_throughput_mbps` | 268.821723 | 268.125789 | +0.695933 |
| `gnn-madqn_gat` | 42 | `sla_satisfaction_pct` | 89.217810 | 89.072499 | +0.145311 |
| `gnn-madqn_gat` | 43 | `embb_p5_mbps` | 0.000005 | 0.000005 | -0.000000 |
| `gnn-madqn_gat` | 43 | `timely_throughput_mbps` | 268.374276 | 268.111097 | +0.263179 |
| `gnn-madqn_gat` | 43 | `sla_satisfaction_pct` | 89.080160 | 88.971474 | +0.108686 |
| `gnn-madqn_gat` | 44 | `embb_p5_mbps` | 0.000005 | 0.000004 | +0.000000 |
| `gnn-madqn_gat` | 44 | `timely_throughput_mbps` | 268.615169 | 267.571971 | +1.043198 |
| `gnn-madqn_gat` | 44 | `sla_satisfaction_pct` | 89.150195 | 88.832373 | +0.317822 |
| `gnn-madqn_gat` | 45 | `embb_p5_mbps` | 0.000005 | 0.000005 | -0.000000 |
| `gnn-madqn_gat` | 45 | `timely_throughput_mbps` | 268.303615 | 268.219223 | +0.084392 |
| `gnn-madqn_gat` | 45 | `sla_satisfaction_pct` | 89.041078 | 88.965502 | +0.075576 |
| `gnn-madqn_gat` | 46 | `embb_p5_mbps` | 0.000004 | 0.000004 | +0.000000 |
| `gnn-madqn_gat` | 46 | `timely_throughput_mbps` | 268.441233 | 268.256564 | +0.184669 |
| `gnn-madqn_gat` | 46 | `sla_satisfaction_pct` | 89.085545 | 88.985033 | +0.100512 |
| `gnn-mappo_gat` | 42 | `embb_p5_mbps` | 0.000003 | 0.000004 | -0.000000 |
| `gnn-mappo_gat` | 42 | `timely_throughput_mbps` | 270.867597 | 268.281572 | +2.586025 |
| `gnn-mappo_gat` | 42 | `sla_satisfaction_pct` | 89.885168 | 88.960636 | +0.924533 |
| `gnn-mappo_gat` | 43 | `embb_p5_mbps` | 0.000004 | 0.000005 | -0.000000 |
| `gnn-mappo_gat` | 43 | `timely_throughput_mbps` | 269.043937 | 270.276552 | -1.232615 |
| `gnn-mappo_gat` | 43 | `sla_satisfaction_pct` | 89.304296 | 89.701369 | -0.397073 |
| `gnn-mappo_gat` | 44 | `embb_p5_mbps` | 1.490946 | 1.481334 | +0.009612 |
| `gnn-mappo_gat` | 44 | `timely_throughput_mbps` | 262.845882 | 262.864257 | -0.018374 |
| `gnn-mappo_gat` | 44 | `sla_satisfaction_pct` | 87.710743 | 87.717820 | -0.007077 |
| `gnn-mappo_gat` | 45 | `embb_p5_mbps` | 0.000003 | 0.000003 | -0.000000 |
| `gnn-mappo_gat` | 45 | `timely_throughput_mbps` | 266.692685 | 266.741243 | -0.048558 |
| `gnn-mappo_gat` | 45 | `sla_satisfaction_pct` | 88.915256 | 88.929169 | -0.013913 |
| `gnn-mappo_gat` | 46 | `embb_p5_mbps` | 0.000004 | 0.000004 | +0.000000 |
| `gnn-mappo_gat` | 46 | `timely_throughput_mbps` | 269.767297 | 271.352432 | -1.585135 |
| `gnn-mappo_gat` | 46 | `sla_satisfaction_pct` | 89.525711 | 89.968317 | -0.442606 |

`embb_p5_mbps`: largest absolute change 0.009612 (8.736% of the un-ablated value), median change +0.000000, checkpoints moved by >1% of their own value: 8/10.

`timely_throughput_mbps`: largest absolute change 2.586025 (0.955% of the un-ablated value), median change +0.134531, checkpoints moved by >1% of their own value: 0/10.

`sla_satisfaction_pct`: largest absolute change 0.924533 (1.029% of the un-ablated value), median change +0.088044, checkpoints moved by >1% of their own value: 1/10.