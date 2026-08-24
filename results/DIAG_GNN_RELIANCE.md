# D2 GNN reliance + D3 over-smoothing

Checkpoints: `results/logs/gnn-*_v4_seed*.pt`. Episodes per arm: 10, seeds from `EVAL_SEED_BASE = 10000`.

**Readout is per family, never pooled** (Gate C3): sampled for PPO (P3, frozen 2026-08-08), argmax for DQN (determination of 2026-08-16). Each row states its own. All three arms of a row draw the same action noise (`torch.manual_seed(eval_seed)` before each), so a difference is the ablation and not sampling luck.

**Three KPIs, not `embb_p5_mbps` alone.** At this operating point most checkpoints already sit at the cell-edge floor and a KPI pinned near 1e-6 cannot degrade however much the ablation changes; reporting it alone would turn *no headroom* into a false *the GNN does not matter*. `timely_throughput_mbps` and `sla_satisfaction_pct` still have headroom, so that is where a real effect has to show up.

## D2a -- neighbour messages zeroed

GAT gets explicit self-loops with a zero edge attribute and PyG's own self-loop insertion turned off: fed an edge set that is only self-loops, `GATv2Conv` removes them and re-adds them with `fill_value='mean'` over an empty `edge_attr`, whose mean is NaN. SAGE gets an empty edge set instead -- `SAGEConv` keeps a separate root weight, so empty already means *self only*, and adding self-loops there would count a node's own features twice and understate the ablation.

## D2b -- edge attributes shuffled within each receiving node

**A limit of this topology, not of the test.** `build_interference_graph` (`envs/channel_model.py`) emits every ordered pair of gNB, so the graph is complete: every receiving node has the identical neighbour set, and permuting source labels within a destination group is a no-op. What is destroyed here is the edge-to-attribute pairing. On a complete graph D2b therefore tests sensitivity to edge *information*, not to *topology*.

`sage` rows are **N/A**, not zero: `SAGEConv` never reads `edge_attr`, so there is nothing for this arm to perturb. Writing 0 would read as *the model ignored it*.

| algo | seed | backbone | readout | KPI | normal | D2a zeroed | D2b shuffled |
|---|---|---|---|---|---|---|---|
| `gnn-madqn_gat` | 42 | gat | argmax | `embb_p5_mbps` | 1.880299 | 0.579987 | 1.878894 |
| `gnn-madqn_gat` | 42 | gat | argmax | `timely_throughput_mbps` | 69.534591 | 71.316023 | 69.578245 |
| `gnn-madqn_gat` | 42 | gat | argmax | `sla_satisfaction_pct` | 91.590229 | 93.772370 | 91.707613 |
| `gnn-madqn_gat` | 43 | gat | argmax | `embb_p5_mbps` | 1.903821 | 1.353389 | 1.903821 |
| `gnn-madqn_gat` | 43 | gat | argmax | `timely_throughput_mbps` | 68.350864 | 68.350621 | 68.350864 |
| `gnn-madqn_gat` | 43 | gat | argmax | `sla_satisfaction_pct` | 90.009367 | 90.009367 | 90.009367 |
| `gnn-madqn_gat` | 44 | gat | argmax | `embb_p5_mbps` | 0.000038 | 0.000006 | 0.000038 |
| `gnn-madqn_gat` | 44 | gat | argmax | `timely_throughput_mbps` | 62.747315 | 61.452316 | 62.747315 |
| `gnn-madqn_gat` | 44 | gat | argmax | `sla_satisfaction_pct` | 83.455749 | 82.600254 | 83.455749 |
| `gnn-madqn_gat` | 45 | gat | argmax | `embb_p5_mbps` | 0.000039 | 0.000003 | 0.000039 |
| `gnn-madqn_gat` | 45 | gat | argmax | `timely_throughput_mbps` | 69.446764 | 70.804182 | 69.166207 |
| `gnn-madqn_gat` | 45 | gat | argmax | `sla_satisfaction_pct` | 91.334214 | 93.142343 | 90.944945 |
| `gnn-madqn_gat` | 46 | gat | argmax | `embb_p5_mbps` | 0.000038 | 0.000003 | 0.000038 |
| `gnn-madqn_gat` | 46 | gat | argmax | `timely_throughput_mbps` | 70.309145 | 70.940361 | 70.288954 |
| `gnn-madqn_gat` | 46 | gat | argmax | `sla_satisfaction_pct` | 92.348095 | 93.084393 | 92.333132 |
| `gnn-madqn_sage` | 42 | sage | argmax | `embb_p5_mbps` | 1.287993 | 1.885919 | N/A |
| `gnn-madqn_sage` | 42 | sage | argmax | `timely_throughput_mbps` | 70.158659 | 70.205214 | N/A |
| `gnn-madqn_sage` | 42 | sage | argmax | `sla_satisfaction_pct` | 92.228075 | 92.229532 | N/A |
| `gnn-madqn_sage` | 43 | sage | argmax | `embb_p5_mbps` | 1.900294 | 1.156286 | N/A |
| `gnn-madqn_sage` | 43 | sage | argmax | `timely_throughput_mbps` | 69.121138 | 69.388335 | N/A |
| `gnn-madqn_sage` | 43 | sage | argmax | `sla_satisfaction_pct` | 90.910374 | 91.228308 | N/A |
| `gnn-madqn_sage` | 44 | sage | argmax | `embb_p5_mbps` | 1.893994 | 1.907188 | N/A |
| `gnn-madqn_sage` | 44 | sage | argmax | `timely_throughput_mbps` | 68.537972 | 62.289168 | N/A |
| `gnn-madqn_sage` | 44 | sage | argmax | `sla_satisfaction_pct` | 90.151713 | 83.508132 | N/A |
| `gnn-madqn_sage` | 45 | sage | argmax | `embb_p5_mbps` | 1.903916 | 0.966782 | N/A |
| `gnn-madqn_sage` | 45 | sage | argmax | `timely_throughput_mbps` | 68.393484 | 68.879971 | N/A |
| `gnn-madqn_sage` | 45 | sage | argmax | `sla_satisfaction_pct` | 90.064604 | 90.592649 | N/A |
| `gnn-madqn_sage` | 46 | sage | argmax | `embb_p5_mbps` | 1.898188 | 1.934470 | N/A |
| `gnn-madqn_sage` | 46 | sage | argmax | `timely_throughput_mbps` | 69.129466 | 68.607131 | N/A |
| `gnn-madqn_sage` | 46 | sage | argmax | `sla_satisfaction_pct` | 90.921193 | 90.170999 | N/A |
| `gnn-mappo_gat` | 42 | gat | sampled | `embb_p5_mbps` | 0.000005 | 0.000005 | 0.000005 |
| `gnn-mappo_gat` | 42 | gat | sampled | `timely_throughput_mbps` | 71.404381 | 70.209931 | 71.336206 |
| `gnn-mappo_gat` | 42 | gat | sampled | `sla_satisfaction_pct` | 93.972857 | 92.330582 | 93.881775 |
| `gnn-mappo_gat` | 43 | gat | sampled | `embb_p5_mbps` | 0.000006 | 0.000006 | 0.000006 |
| `gnn-mappo_gat` | 43 | gat | sampled | `timely_throughput_mbps` | 70.912244 | 70.769196 | 70.906121 |
| `gnn-mappo_gat` | 43 | gat | sampled | `sla_satisfaction_pct` | 93.322506 | 93.225384 | 93.317775 |
| `gnn-mappo_gat` | 44 | gat | sampled | `embb_p5_mbps` | 1.573219 | 1.573314 | 1.573219 |
| `gnn-mappo_gat` | 44 | gat | sampled | `timely_throughput_mbps` | 70.705271 | 70.672457 | 70.705271 |
| `gnn-mappo_gat` | 44 | gat | sampled | `sla_satisfaction_pct` | 93.275288 | 93.213449 | 93.275288 |
| `gnn-mappo_gat` | 45 | gat | sampled | `embb_p5_mbps` | 0.000004 | 0.000005 | 0.000004 |
| `gnn-mappo_gat` | 45 | gat | sampled | `timely_throughput_mbps` | 71.160574 | 71.074129 | 71.160574 |
| `gnn-mappo_gat` | 45 | gat | sampled | `sla_satisfaction_pct` | 94.007629 | 93.894330 | 94.007629 |
| `gnn-mappo_gat` | 46 | gat | sampled | `embb_p5_mbps` | 0.000006 | 0.000006 | 0.000006 |
| `gnn-mappo_gat` | 46 | gat | sampled | `timely_throughput_mbps` | 70.580584 | 70.681615 | 70.599716 |
| `gnn-mappo_gat` | 46 | gat | sampled | `sla_satisfaction_pct` | 92.867877 | 93.008094 | 92.896959 |
| `gnn-mappo_gat` | 47 | gat | sampled | `embb_p5_mbps` | 0.000007 | 0.000006 | 0.000007 |
| `gnn-mappo_gat` | 47 | gat | sampled | `timely_throughput_mbps` | 70.699888 | 70.630645 | 70.699888 |
| `gnn-mappo_gat` | 47 | gat | sampled | `sla_satisfaction_pct` | 92.885260 | 92.817549 | 92.885260 |
| `gnn-mappo_gat` | 48 | gat | sampled | `embb_p5_mbps` | 0.000005 | 0.000004 | 0.000005 |
| `gnn-mappo_gat` | 48 | gat | sampled | `timely_throughput_mbps` | 72.271779 | 72.200769 | 72.271779 |
| `gnn-mappo_gat` | 48 | gat | sampled | `sla_satisfaction_pct` | 95.148455 | 95.030394 | 95.148455 |
| `gnn-mappo_gat` | 49 | gat | sampled | `embb_p5_mbps` | 0.000006 | 0.000006 | 0.000006 |
| `gnn-mappo_gat` | 49 | gat | sampled | `timely_throughput_mbps` | 70.220091 | 70.695050 | 70.291907 |
| `gnn-mappo_gat` | 49 | gat | sampled | `sla_satisfaction_pct` | 92.357544 | 93.012118 | 92.452377 |
| `gnn-mappo_gat` | 50 | gat | sampled | `embb_p5_mbps` | 1.448006 | 1.447962 | 1.448006 |
| `gnn-mappo_gat` | 50 | gat | sampled | `timely_throughput_mbps` | 72.108871 | 72.172950 | 72.108871 |
| `gnn-mappo_gat` | 50 | gat | sampled | `sla_satisfaction_pct` | 94.738627 | 94.758706 | 94.738627 |
| `gnn-mappo_gat` | 51 | gat | sampled | `embb_p5_mbps` | 0.000004 | 0.000005 | 0.000004 |
| `gnn-mappo_gat` | 51 | gat | sampled | `timely_throughput_mbps` | 70.858267 | 70.830889 | 70.858267 |
| `gnn-mappo_gat` | 51 | gat | sampled | `sla_satisfaction_pct` | 93.270709 | 93.234260 | 93.270709 |
| `gnn-mappo_gat` | 52 | gat | sampled | `embb_p5_mbps` | 0.000005 | 0.000005 | 0.000005 |
| `gnn-mappo_gat` | 52 | gat | sampled | `timely_throughput_mbps` | 72.059590 | 72.141748 | 72.059365 |
| `gnn-mappo_gat` | 52 | gat | sampled | `sla_satisfaction_pct` | 94.830425 | 95.006890 | 94.830258 |
| `gnn-mappo_gat` | 53 | gat | sampled | `embb_p5_mbps` | 0.000007 | 0.000007 | 0.000007 |
| `gnn-mappo_gat` | 53 | gat | sampled | `timely_throughput_mbps` | 71.015521 | 70.286379 | 71.008481 |
| `gnn-mappo_gat` | 53 | gat | sampled | `sla_satisfaction_pct` | 93.473146 | 92.488069 | 93.460323 |
| `gnn-mappo_gat` | 54 | gat | sampled | `embb_p5_mbps` | 1.040919 | 0.908136 | 1.040950 |
| `gnn-mappo_gat` | 54 | gat | sampled | `timely_throughput_mbps` | 72.517934 | 72.527345 | 72.496949 |
| `gnn-mappo_gat` | 54 | gat | sampled | `sla_satisfaction_pct` | 95.186875 | 95.194760 | 95.159166 |
| `gnn-mappo_gat` | 55 | gat | sampled | `embb_p5_mbps` | 0.571680 | 0.722703 | 0.571680 |
| `gnn-mappo_gat` | 55 | gat | sampled | `timely_throughput_mbps` | 70.976980 | 71.455576 | 70.973615 |
| `gnn-mappo_gat` | 55 | gat | sampled | `sla_satisfaction_pct` | 93.433675 | 94.049972 | 93.434346 |
| `gnn-mappo_gat` | 56 | gat | sampled | `embb_p5_mbps` | 0.000005 | 0.000005 | 0.000005 |
| `gnn-mappo_gat` | 56 | gat | sampled | `timely_throughput_mbps` | 71.119740 | 70.961995 | 71.145946 |
| `gnn-mappo_gat` | 56 | gat | sampled | `sla_satisfaction_pct` | 93.573042 | 93.383910 | 93.607530 |
| `gnn-mappo_gat` | 57 | gat | sampled | `embb_p5_mbps` | 0.189401 | 0.190491 | 0.189401 |
| `gnn-mappo_gat` | 57 | gat | sampled | `timely_throughput_mbps` | 70.986545 | 71.486854 | 70.980095 |
| `gnn-mappo_gat` | 57 | gat | sampled | `sla_satisfaction_pct` | 93.552234 | 94.044816 | 93.543273 |
| `gnn-mappo_gat` | 58 | gat | sampled | `embb_p5_mbps` | 0.000003 | 0.000003 | 0.000003 |
| `gnn-mappo_gat` | 58 | gat | sampled | `timely_throughput_mbps` | 71.539042 | 71.487283 | 71.539042 |
| `gnn-mappo_gat` | 58 | gat | sampled | `sla_satisfaction_pct` | 94.230753 | 94.170964 | 94.230753 |
| `gnn-mappo_gat` | 59 | gat | sampled | `embb_p5_mbps` | 0.000007 | 0.000007 | 0.000007 |
| `gnn-mappo_gat` | 59 | gat | sampled | `timely_throughput_mbps` | 69.113343 | 69.267295 | 69.113343 |
| `gnn-mappo_gat` | 59 | gat | sampled | `sla_satisfaction_pct` | 91.323664 | 91.475556 | 91.323664 |
| `gnn-mappo_gat` | 60 | gat | sampled | `embb_p5_mbps` | 1.608690 | 1.609192 | 1.608690 |
| `gnn-mappo_gat` | 60 | gat | sampled | `timely_throughput_mbps` | 70.145772 | 70.132363 | 70.145772 |
| `gnn-mappo_gat` | 60 | gat | sampled | `sla_satisfaction_pct` | 92.204536 | 92.189695 | 92.204536 |
| `gnn-mappo_gat` | 61 | gat | sampled | `embb_p5_mbps` | 0.000006 | 0.000006 | 0.000006 |
| `gnn-mappo_gat` | 61 | gat | sampled | `timely_throughput_mbps` | 71.395277 | 70.580136 | 71.386562 |
| `gnn-mappo_gat` | 61 | gat | sampled | `sla_satisfaction_pct` | 93.973858 | 92.880591 | 93.962035 |
| `gnn-mappo_sage` | 42 | sage | sampled | `embb_p5_mbps` | 0.000003 | 0.000003 | N/A |
| `gnn-mappo_sage` | 42 | sage | sampled | `timely_throughput_mbps` | 72.192420 | 72.178885 | N/A |
| `gnn-mappo_sage` | 42 | sage | sampled | `sla_satisfaction_pct` | 94.823560 | 95.016094 | N/A |
| `gnn-mappo_sage` | 43 | sage | sampled | `embb_p5_mbps` | 0.000005 | 0.000005 | N/A |
| `gnn-mappo_sage` | 43 | sage | sampled | `timely_throughput_mbps` | 71.988211 | 71.313799 | N/A |
| `gnn-mappo_sage` | 43 | sage | sampled | `sla_satisfaction_pct` | 94.716041 | 93.839658 | N/A |
| `gnn-mappo_sage` | 44 | sage | sampled | `embb_p5_mbps` | 0.000006 | 0.000006 | N/A |
| `gnn-mappo_sage` | 44 | sage | sampled | `timely_throughput_mbps` | 70.925705 | 70.686537 | N/A |
| `gnn-mappo_sage` | 44 | sage | sampled | `sla_satisfaction_pct` | 93.300099 | 93.010148 | N/A |
| `gnn-mappo_sage` | 45 | sage | sampled | `embb_p5_mbps` | 0.000004 | 0.000004 | N/A |
| `gnn-mappo_sage` | 45 | sage | sampled | `timely_throughput_mbps` | 71.838878 | 71.806594 | N/A |
| `gnn-mappo_sage` | 45 | sage | sampled | `sla_satisfaction_pct` | 94.533253 | 94.500904 | N/A |
| `gnn-mappo_sage` | 46 | sage | sampled | `embb_p5_mbps` | 0.000005 | 0.000006 | N/A |
| `gnn-mappo_sage` | 46 | sage | sampled | `timely_throughput_mbps` | 70.966970 | 70.676586 | N/A |
| `gnn-mappo_sage` | 46 | sage | sampled | `sla_satisfaction_pct` | 93.385720 | 93.026155 | N/A |
| `gnn-mappo_sage` | 47 | sage | sampled | `embb_p5_mbps` | 0.000005 | 0.000005 | N/A |
| `gnn-mappo_sage` | 47 | sage | sampled | `timely_throughput_mbps` | 72.794572 | 71.722334 | N/A |
| `gnn-mappo_sage` | 47 | sage | sampled | `sla_satisfaction_pct` | 95.842106 | 94.472336 | N/A |
| `gnn-mappo_sage` | 48 | sage | sampled | `embb_p5_mbps` | 0.000004 | 0.000004 | N/A |
| `gnn-mappo_sage` | 48 | sage | sampled | `timely_throughput_mbps` | 71.859576 | 71.439281 | N/A |
| `gnn-mappo_sage` | 48 | sage | sampled | `sla_satisfaction_pct` | 94.620322 | 94.126942 | N/A |
| `gnn-mappo_sage` | 49 | sage | sampled | `embb_p5_mbps` | 0.000005 | 0.000006 | N/A |
| `gnn-mappo_sage` | 49 | sage | sampled | `timely_throughput_mbps` | 71.074699 | 70.875346 | N/A |
| `gnn-mappo_sage` | 49 | sage | sampled | `sla_satisfaction_pct` | 93.546936 | 93.271104 | N/A |
| `gnn-mappo_sage` | 50 | sage | sampled | `embb_p5_mbps` | 1.070322 | 0.000006 | N/A |
| `gnn-mappo_sage` | 50 | sage | sampled | `timely_throughput_mbps` | 71.571538 | 70.471302 | N/A |
| `gnn-mappo_sage` | 50 | sage | sampled | `sla_satisfaction_pct` | 94.163958 | 92.709255 | N/A |
| `gnn-mappo_sage` | 51 | sage | sampled | `embb_p5_mbps` | 0.000004 | 0.000006 | N/A |
| `gnn-mappo_sage` | 51 | sage | sampled | `timely_throughput_mbps` | 71.195339 | 70.635475 | N/A |
| `gnn-mappo_sage` | 51 | sage | sampled | `sla_satisfaction_pct` | 93.804297 | 92.994834 | N/A |
| `gnn-mappo_sage` | 52 | sage | sampled | `embb_p5_mbps` | 0.000005 | 0.000006 | N/A |
| `gnn-mappo_sage` | 52 | sage | sampled | `timely_throughput_mbps` | 70.890979 | 70.684835 | N/A |
| `gnn-mappo_sage` | 52 | sage | sampled | `sla_satisfaction_pct` | 93.253231 | 93.018875 | N/A |
| `gnn-mappo_sage` | 53 | sage | sampled | `embb_p5_mbps` | 0.000003 | 0.000004 | N/A |
| `gnn-mappo_sage` | 53 | sage | sampled | `timely_throughput_mbps` | 71.680610 | 71.070569 | N/A |
| `gnn-mappo_sage` | 53 | sage | sampled | `sla_satisfaction_pct` | 94.363888 | 93.501209 | N/A |
| `gnn-mappo_sage` | 54 | sage | sampled | `embb_p5_mbps` | 0.000004 | 0.000006 | N/A |
| `gnn-mappo_sage` | 54 | sage | sampled | `timely_throughput_mbps` | 71.637634 | 71.162144 | N/A |
| `gnn-mappo_sage` | 54 | sage | sampled | `sla_satisfaction_pct` | 94.298403 | 93.735435 | N/A |
| `gnn-mappo_sage` | 55 | sage | sampled | `embb_p5_mbps` | 0.000004 | 0.000006 | N/A |
| `gnn-mappo_sage` | 55 | sage | sampled | `timely_throughput_mbps` | 71.209883 | 70.553227 | N/A |
| `gnn-mappo_sage` | 55 | sage | sampled | `sla_satisfaction_pct` | 93.830829 | 92.854516 | N/A |
| `gnn-mappo_sage` | 56 | sage | sampled | `embb_p5_mbps` | 0.000003 | 0.000005 | N/A |
| `gnn-mappo_sage` | 56 | sage | sampled | `timely_throughput_mbps` | 71.813145 | 71.165420 | N/A |
| `gnn-mappo_sage` | 56 | sage | sampled | `sla_satisfaction_pct` | 94.387481 | 93.649905 | N/A |
| `gnn-mappo_sage` | 57 | sage | sampled | `embb_p5_mbps` | 0.000006 | 0.000006 | N/A |
| `gnn-mappo_sage` | 57 | sage | sampled | `timely_throughput_mbps` | 70.664554 | 70.691074 | N/A |
| `gnn-mappo_sage` | 57 | sage | sampled | `sla_satisfaction_pct` | 93.004383 | 93.042873 | N/A |
| `gnn-mappo_sage` | 58 | sage | sampled | `embb_p5_mbps` | 0.000004 | 0.000005 | N/A |
| `gnn-mappo_sage` | 58 | sage | sampled | `timely_throughput_mbps` | 70.849500 | 70.776138 | N/A |
| `gnn-mappo_sage` | 58 | sage | sampled | `sla_satisfaction_pct` | 93.244217 | 93.150305 | N/A |
| `gnn-mappo_sage` | 59 | sage | sampled | `embb_p5_mbps` | 0.000005 | 0.000006 | N/A |
| `gnn-mappo_sage` | 59 | sage | sampled | `timely_throughput_mbps` | 71.537622 | 71.106651 | N/A |
| `gnn-mappo_sage` | 59 | sage | sampled | `sla_satisfaction_pct` | 94.225671 | 93.589277 | N/A |
| `gnn-mappo_sage` | 60 | sage | sampled | `embb_p5_mbps` | 0.000004 | 0.000003 | N/A |
| `gnn-mappo_sage` | 60 | sage | sampled | `timely_throughput_mbps` | 71.983659 | 71.995580 | N/A |
| `gnn-mappo_sage` | 60 | sage | sampled | `sla_satisfaction_pct` | 94.957543 | 94.784300 | N/A |
| `gnn-mappo_sage` | 61 | sage | sampled | `embb_p5_mbps` | 0.000006 | 0.000006 | N/A |
| `gnn-mappo_sage` | 61 | sage | sampled | `timely_throughput_mbps` | 70.596227 | 70.487354 | N/A |
| `gnn-mappo_sage` | 61 | sage | sampled | `sla_satisfaction_pct` | 92.824279 | 92.721139 | N/A |

`embb_p5_mbps` / D2A: largest absolute change 1.300312 (99.999% of the un-ablated value), median -0.000000, checkpoints moved by >1% of their own value: 35/50.

`embb_p5_mbps` / D2B: largest absolute change 0.001405 (0.332% of the un-ablated value), median +0.000000, checkpoints moved by >1% of their own value: 0/25.

`timely_throughput_mbps` / D2A: largest absolute change 6.248804 (9.117% of the un-ablated value), median +0.072186, checkpoints moved by >1% of their own value: 9/50.

`timely_throughput_mbps` / D2B: largest absolute change 0.280557 (0.404% of the un-ablated value), median +0.000000, checkpoints moved by >1% of their own value: 0/25.

`sla_satisfaction_pct` / D2A: largest absolute change 6.643581 (7.369% of the un-ablated value), median +0.100131, checkpoints moved by >1% of their own value: 10/50.

`sla_satisfaction_pct` / D2B: largest absolute change 0.389270 (0.426% of the un-ablated value), median +0.000000, checkpoints moved by >1% of their own value: 0/25.

## D3 -- over-smoothing

Mean pairwise cosine similarity of the final node embeddings, with the same statistic on the raw observation alongside. The embedding number alone decides nothing: if the inputs are already near-identical, near-identical outputs are not the GNN's doing. State read after 50 policy steps, because `reset()` zeroes 7 of the 8 observation columns.

**The inter-gNB graph is complete, so its diameter is 1** -- one layer already reaches every node and the two configured layers aggregate the identical neighbour set twice. That raises the over-smoothing risk rather than lowering it, the opposite of what PLAN-01 D3 and PLAN-03 §4 assume when they call the diameter *small*.

| algo | seed | backbone | cos(embedding) | cos(obs) reference |
|---|---|---|---|---|
| `gnn-madqn_gat` | 42 | gat | 1.0000 | 0.8440 |
| `gnn-madqn_gat` | 43 | gat | 1.0000 | 0.8147 |
| `gnn-madqn_gat` | 44 | gat | 1.0000 | 0.9692 |
| `gnn-madqn_gat` | 45 | gat | 1.0000 | 0.9715 |
| `gnn-madqn_gat` | 46 | gat | 1.0000 | 0.9747 |
| `gnn-madqn_sage` | 42 | sage | 0.9972 | 0.8421 |
| `gnn-madqn_sage` | 43 | sage | 0.9981 | 0.8115 |
| `gnn-madqn_sage` | 44 | sage | 0.9966 | 0.8318 |
| `gnn-madqn_sage` | 45 | sage | 0.9985 | 0.8142 |
| `gnn-madqn_sage` | 46 | sage | 0.9464 | 0.8116 |
| `gnn-mappo_gat` | 42 | gat | 1.0000 | 0.9098 |
| `gnn-mappo_gat` | 43 | gat | 1.0000 | 0.9844 |
| `gnn-mappo_gat` | 44 | gat | 1.0000 | 0.9844 |
| `gnn-mappo_gat` | 45 | gat | 1.0000 | 0.9844 |
| `gnn-mappo_gat` | 46 | gat | 1.0000 | 0.7955 |
| `gnn-mappo_gat` | 47 | gat | 1.0000 | 0.8444 |
| `gnn-mappo_gat` | 48 | gat | 1.0000 | 0.9844 |
| `gnn-mappo_gat` | 49 | gat | 1.0000 | 0.8115 |
| `gnn-mappo_gat` | 50 | gat | 1.0000 | 0.8625 |
| `gnn-mappo_gat` | 51 | gat | 1.0000 | 0.8116 |
| `gnn-mappo_gat` | 52 | gat | 1.0000 | 0.9844 |
| `gnn-mappo_gat` | 53 | gat | 1.0000 | 0.9716 |
| `gnn-mappo_gat` | 54 | gat | 1.0000 | 0.9844 |
| `gnn-mappo_gat` | 55 | gat | 1.0000 | 0.9712 |
| `gnn-mappo_gat` | 56 | gat | 1.0000 | 0.8334 |
| `gnn-mappo_gat` | 57 | gat | 1.0000 | 0.8583 |
| `gnn-mappo_gat` | 58 | gat | 1.0000 | 0.9762 |
| `gnn-mappo_gat` | 59 | gat | 1.0000 | 0.7976 |
| `gnn-mappo_gat` | 60 | gat | 1.0000 | 0.8274 |
| `gnn-mappo_gat` | 61 | gat | 1.0000 | 0.8588 |
| `gnn-mappo_sage` | 42 | sage | 0.9998 | 0.9757 |
| `gnn-mappo_sage` | 43 | sage | 0.9997 | 0.9844 |
| `gnn-mappo_sage` | 44 | sage | 0.9994 | 0.9789 |
| `gnn-mappo_sage` | 45 | sage | 0.9996 | 0.9839 |
| `gnn-mappo_sage` | 46 | sage | 0.9917 | 0.8807 |
| `gnn-mappo_sage` | 47 | sage | 0.9996 | 0.9844 |
| `gnn-mappo_sage` | 48 | sage | 0.9999 | 0.9836 |
| `gnn-mappo_sage` | 49 | sage | 0.9987 | 0.9763 |
| `gnn-mappo_sage` | 50 | sage | 0.9995 | 0.9844 |
| `gnn-mappo_sage` | 51 | sage | 0.9992 | 0.9844 |
| `gnn-mappo_sage` | 52 | sage | 0.9678 | 0.8538 |
| `gnn-mappo_sage` | 53 | sage | 0.9910 | 0.8438 |
| `gnn-mappo_sage` | 54 | sage | 0.9997 | 0.9762 |
| `gnn-mappo_sage` | 55 | sage | 0.9557 | 0.7981 |
| `gnn-mappo_sage` | 56 | sage | 0.9996 | 0.9762 |
| `gnn-mappo_sage` | 57 | sage | 0.9841 | 0.8366 |
| `gnn-mappo_sage` | 58 | sage | 0.9607 | 0.8393 |
| `gnn-mappo_sage` | 59 | sage | 0.9996 | 0.9844 |
| `gnn-mappo_sage` | 60 | sage | 0.9998 | 0.9841 |
| `gnn-mappo_sage` | 61 | sage | 0.9933 | 0.8448 |
