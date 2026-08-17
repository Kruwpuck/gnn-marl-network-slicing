# Attention vs Interference Mechanism

Readout: `greedy (reported, never gates)`. Checkpoints: `results/logs/gnn-*_gat_v4_seed*.pt`. Config: `configs/experiment_config.yaml (n_gnb=5)`.

n edges pooled across all episodes/seeds: 400000

**Pooled**: Pearson r = -0.0991 (p=0), Spearman r = -0.2462 (p=0) -- attention weight (layer 2, single head) vs raw path-loss dB (env-native scale, not the /100 scaled value fed to the model).

**Per receiving node** (100000 node-steps with a defined rank correlation): mean rho = -0.5146, median = -0.8000, fraction rho < 0 = 0.765. The pooled number mixes edges from nodes with different attention scales and different neighbour sets, which can hide a node that consistently prefers its strongest interferer; the softmax is normalised per receiving node, so that is the level the mechanism claim lives at. A negative rho means more attention on lower path-loss, i.e. on the stronger interferer -- the direction the mechanism story predicts.

**Causal ablation** (mandatory): attention forced uniform over neighbors (zero the learned `att` parameter of both GATv2Conv layers -> softmax degenerates to 1/degree). Both arms draw the same action noise (`torch.manual_seed` per episode) so the difference is the ablation, not sampling luck. Correlation alone would be decoration without this.

Reported on three KPIs, not on `embb_p5_mbps` alone: at this operating point most checkpoints already sit at the cell-edge floor, and a KPI pinned near zero cannot degrade however much the ablation changes. `timely_throughput_mbps` and `sla_satisfaction_pct` still have headroom, so they are where a real causal effect would have to show up.

| algo | seed | KPI | normal | uniform-attn | degradation |
|---|---|---|---|---|---|
| `gnn-madqn_gat` | 42 | `embb_p5_mbps` | 1.880299 | 1.485612 | +0.394686 |
| `gnn-madqn_gat` | 42 | `timely_throughput_mbps` | 69.534591 | 69.289449 | +0.245142 |
| `gnn-madqn_gat` | 42 | `sla_satisfaction_pct` | 91.590229 | 91.269043 | +0.321186 |
| `gnn-madqn_gat` | 43 | `embb_p5_mbps` | 1.903821 | 1.903821 | +0.000000 |
| `gnn-madqn_gat` | 43 | `timely_throughput_mbps` | 68.350864 | 68.350864 | +0.000000 |
| `gnn-madqn_gat` | 43 | `sla_satisfaction_pct` | 90.009367 | 90.009367 | +0.000000 |
| `gnn-madqn_gat` | 44 | `embb_p5_mbps` | 0.000038 | 0.000038 | +0.000000 |
| `gnn-madqn_gat` | 44 | `timely_throughput_mbps` | 62.747315 | 62.765680 | -0.018364 |
| `gnn-madqn_gat` | 44 | `sla_satisfaction_pct` | 83.455749 | 83.432784 | +0.022965 |
| `gnn-madqn_gat` | 45 | `embb_p5_mbps` | 0.000039 | 0.000038 | +0.000000 |
| `gnn-madqn_gat` | 45 | `timely_throughput_mbps` | 69.446764 | 70.257698 | -0.810934 |
| `gnn-madqn_gat` | 45 | `sla_satisfaction_pct` | 91.334214 | 92.315621 | -0.981407 |
| `gnn-madqn_gat` | 46 | `embb_p5_mbps` | 0.000038 | 0.000038 | +0.000000 |
| `gnn-madqn_gat` | 46 | `timely_throughput_mbps` | 70.309145 | 70.289010 | +0.020136 |
| `gnn-madqn_gat` | 46 | `sla_satisfaction_pct` | 92.348095 | 92.319934 | +0.028161 |
| `gnn-mappo_gat` | 42 | `embb_p5_mbps` | 0.558158 | 0.190630 | +0.367528 |
| `gnn-mappo_gat` | 42 | `timely_throughput_mbps` | 44.849965 | 58.036185 | -13.186220 |
| `gnn-mappo_gat` | 42 | `sla_satisfaction_pct` | 61.660558 | 77.861865 | -16.201307 |
| `gnn-mappo_gat` | 43 | `embb_p5_mbps` | 1.912687 | 1.912482 | +0.000205 |
| `gnn-mappo_gat` | 43 | `timely_throughput_mbps` | 17.785672 | 11.243213 | +6.542460 |
| `gnn-mappo_gat` | 43 | `sla_satisfaction_pct` | 27.802730 | 19.786246 | +8.016484 |
| `gnn-mappo_gat` | 44 | `embb_p5_mbps` | 1.912687 | 1.912687 | +0.000000 |
| `gnn-mappo_gat` | 44 | `timely_throughput_mbps` | 0.000203 | 0.000203 | +0.000000 |
| `gnn-mappo_gat` | 44 | `sla_satisfaction_pct` | 5.447616 | 5.447616 | +0.000000 |
| `gnn-mappo_gat` | 45 | `embb_p5_mbps` | 1.912687 | 1.912687 | +0.000000 |
| `gnn-mappo_gat` | 45 | `timely_throughput_mbps` | 0.000203 | 0.000203 | +0.000000 |
| `gnn-mappo_gat` | 45 | `sla_satisfaction_pct` | 5.447616 | 5.447616 | +0.000000 |
| `gnn-mappo_gat` | 46 | `embb_p5_mbps` | 1.716552 | 1.792554 | -0.076002 |
| `gnn-mappo_gat` | 46 | `timely_throughput_mbps` | 69.409227 | 70.192467 | -0.783241 |
| `gnn-mappo_gat` | 46 | `sla_satisfaction_pct` | 91.322428 | 92.165681 | -0.843253 |

`embb_p5_mbps`: largest absolute change 0.394686 (65.847% of the un-ablated value), median change +0.000000, checkpoints moved by >1% of their own value: 3/10.

`timely_throughput_mbps`: largest absolute change 13.186220 (36.785% of the un-ablated value), median change +0.000000, checkpoints moved by >1% of their own value: 4/10.

`sla_satisfaction_pct`: largest absolute change 16.201307 (28.833% of the un-ablated value), median change +0.000000, checkpoints moved by >1% of their own value: 3/10.