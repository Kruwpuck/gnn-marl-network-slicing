# Attention vs Interference Mechanism

> **VOID FOR THE DQN FAMILY — do not cite this file.** The `gnn-madqn_gat` half of every table and ablation were produced by the
> epsilon=1.0 readout, i.e. uniform random actions rather than the trained policy
> (`results/quarantine_eps1.0/README.md`). The source CSVs have been quarantined, so this
> file cannot be regenerated as-is and is kept only as a record of what was reported before
> the fault was found. Valid replacement: the greedy pass `results/ATTENTION_v4_greedy.md` for the DQN half; the PPO half of this file is unaffected.

Readout: `stochastic (P3 primary)`. Checkpoints: `results/logs/gnn-*_gat_v4_seed*.pt`. Config: `configs/experiment_config.yaml (n_gnb=5)`.

n edges pooled across all episodes/seeds: 400000

**Pooled**: Pearson r = -0.1308 (p=0), Spearman r = -0.2483 (p=0) -- attention weight (layer 2, single head) vs raw path-loss dB (env-native scale, not the /100 scaled value fed to the model).

**Per receiving node** (100000 node-steps with a defined rank correlation): mean rho = -0.5308, median = -0.8000, fraction rho < 0 = 0.784. The pooled number mixes edges from nodes with different attention scales and different neighbour sets, which can hide a node that consistently prefers its strongest interferer; the softmax is normalised per receiving node, so that is the level the mechanism claim lives at. A negative rho means more attention on lower path-loss, i.e. on the stronger interferer -- the direction the mechanism story predicts.

**Causal ablation** (mandatory): attention forced uniform over neighbors (zero the learned `att` parameter of both GATv2Conv layers -> softmax degenerates to 1/degree). Both arms draw the same action noise (`torch.manual_seed` per episode) so the difference is the ablation, not sampling luck. Correlation alone would be decoration without this.

Reported on three KPIs, not on `embb_p5_mbps` alone: at this operating point most checkpoints already sit at the cell-edge floor, and a KPI pinned near zero cannot degrade however much the ablation changes. `timely_throughput_mbps` and `sla_satisfaction_pct` still have headroom, so they are where a real causal effect would have to show up.

| algo | seed | KPI | normal | uniform-attn | degradation |
|---|---|---|---|---|---|
| `gnn-madqn_gat` | 42 | `embb_p5_mbps` | 0.000006 | 0.000007 | -0.000000 |
| `gnn-madqn_gat` | 42 | `timely_throughput_mbps` | 70.858898 | 70.439062 | +0.419835 |
| `gnn-madqn_gat` | 42 | `sla_satisfaction_pct` | 93.241861 | 92.756053 | +0.485809 |
| `gnn-madqn_gat` | 43 | `embb_p5_mbps` | 0.000006 | 0.000006 | +0.000000 |
| `gnn-madqn_gat` | 43 | `timely_throughput_mbps` | 71.033228 | 70.170405 | +0.862824 |
| `gnn-madqn_gat` | 43 | `sla_satisfaction_pct` | 93.454332 | 92.470688 | +0.983644 |
| `gnn-madqn_gat` | 44 | `embb_p5_mbps` | 0.000006 | 0.000006 | -0.000000 |
| `gnn-madqn_gat` | 44 | `timely_throughput_mbps` | 70.210517 | 70.417509 | -0.206992 |
| `gnn-madqn_gat` | 44 | `sla_satisfaction_pct` | 92.621003 | 92.737814 | -0.116811 |
| `gnn-madqn_gat` | 45 | `embb_p5_mbps` | 0.000007 | 0.000006 | +0.000000 |
| `gnn-madqn_gat` | 45 | `timely_throughput_mbps` | 70.666344 | 69.916969 | +0.749375 |
| `gnn-madqn_gat` | 45 | `sla_satisfaction_pct` | 92.898707 | 92.019345 | +0.879362 |
| `gnn-madqn_gat` | 46 | `embb_p5_mbps` | 0.000006 | 0.000006 | -0.000000 |
| `gnn-madqn_gat` | 46 | `timely_throughput_mbps` | 70.533127 | 70.592774 | -0.059648 |
| `gnn-madqn_gat` | 46 | `sla_satisfaction_pct` | 92.899659 | 92.841404 | +0.058255 |
| `gnn-mappo_gat` | 42 | `embb_p5_mbps` | 0.000005 | 0.000005 | -0.000000 |
| `gnn-mappo_gat` | 42 | `timely_throughput_mbps` | 71.404381 | 70.425153 | +0.979228 |
| `gnn-mappo_gat` | 42 | `sla_satisfaction_pct` | 93.972857 | 92.657921 | +1.314936 |
| `gnn-mappo_gat` | 43 | `embb_p5_mbps` | 0.000006 | 0.000007 | -0.000000 |
| `gnn-mappo_gat` | 43 | `timely_throughput_mbps` | 70.912244 | 71.177279 | -0.265036 |
| `gnn-mappo_gat` | 43 | `sla_satisfaction_pct` | 93.322506 | 93.667217 | -0.344710 |
| `gnn-mappo_gat` | 44 | `embb_p5_mbps` | 1.573219 | 1.573209 | +0.000010 |
| `gnn-mappo_gat` | 44 | `timely_throughput_mbps` | 70.705271 | 70.702494 | +0.002777 |
| `gnn-mappo_gat` | 44 | `sla_satisfaction_pct` | 93.275288 | 93.271609 | +0.003679 |
| `gnn-mappo_gat` | 45 | `embb_p5_mbps` | 0.000004 | 0.000004 | +0.000000 |
| `gnn-mappo_gat` | 45 | `timely_throughput_mbps` | 71.160574 | 71.160574 | +0.000000 |
| `gnn-mappo_gat` | 45 | `sla_satisfaction_pct` | 94.007629 | 94.007629 | +0.000000 |
| `gnn-mappo_gat` | 46 | `embb_p5_mbps` | 0.000006 | 0.000005 | +0.000000 |
| `gnn-mappo_gat` | 46 | `timely_throughput_mbps` | 70.580584 | 71.063284 | -0.482700 |
| `gnn-mappo_gat` | 46 | `sla_satisfaction_pct` | 92.867877 | 93.488900 | -0.621023 |

`embb_p5_mbps`: largest absolute change 0.000010 (7.137% of the un-ablated value), median change -0.000000, checkpoints moved by >1% of their own value: 8/10.

`timely_throughput_mbps`: largest absolute change 0.979228 (1.371% of the un-ablated value), median change +0.001389, checkpoints moved by >1% of their own value: 3/10.

`sla_satisfaction_pct`: largest absolute change 1.314936 (1.399% of the un-ablated value), median change +0.030967, checkpoints moved by >1% of their own value: 2/10.