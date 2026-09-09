# D2 GNN reliance + D3 over-smoothing

Checkpoints: `results/logs/gnn-*_v6_seed*.pt`. Episodes per arm: 10, seeds from `EVAL_SEED_BASE = 10000`.

**Readout is per family, never pooled** (Gate C3): sampled for PPO (P3, frozen 2026-08-08), argmax for DQN (determination of 2026-08-16). Each row states its own. All three arms of a row draw the same action noise (`torch.manual_seed(eval_seed)` before each), so a difference is the ablation and not sampling luck.

**Three KPIs, not `embb_p5_mbps` alone.** At this operating point most checkpoints already sit at the cell-edge floor and a KPI pinned near 1e-6 cannot degrade however much the ablation changes; reporting it alone would turn *no headroom* into a false *the GNN does not matter*. `timely_throughput_mbps` and `sla_satisfaction_pct` still have headroom, so that is where a real effect has to show up.

## D2a -- neighbour messages zeroed

GAT gets explicit self-loops with a zero edge attribute and PyG's own self-loop insertion turned off: fed an edge set that is only self-loops, `GATv2Conv` removes them and re-adds them with `fill_value='mean'` over an empty `edge_attr`, whose mean is NaN. SAGE gets an empty edge set instead -- `SAGEConv` keeps a separate root weight, so empty already means *self only*, and adding self-loops there would count a node's own features twice and understate the ablation.

## D2b -- edge attributes shuffled within each receiving node

**A limit of this topology, not of the test.** `build_interference_graph` (`envs/channel_model.py`) emits every ordered pair of gNB, so the graph is complete: every receiving node has the identical neighbour set, and permuting source labels within a destination group is a no-op. What is destroyed here is the edge-to-attribute pairing. On a complete graph D2b therefore tests sensitivity to edge *information*, not to *topology*.

Rows for a backbone without `edge_dim` are **N/A**, not zero: `SAGEConv` and `GCNConv` never read `edge_attr`, so there is nothing for those arms to perturb. Writing 0 would read as *the model ignored it*. All four GATv2 arms (`gat`, `gatres`, `gatedge`, `gatres-edge`) are measured.

| algo | seed | backbone | readout | KPI | normal | D2a zeroed | D2b shuffled |
|---|---|---|---|---|---|---|---|
| `gnn-madqn_gatedge` | 42 | gatedge | argmax | `embb_p5_mbps` | 0.478253 | 0.000003 | 0.355068 |
| `gnn-madqn_gatedge` | 42 | gatedge | argmax | `timely_throughput_mbps` | 65.188686 | 71.710728 | 65.228792 |
| `gnn-madqn_gatedge` | 42 | gatedge | argmax | `sla_satisfaction_pct` | 86.589740 | 94.081271 | 86.474561 |
| `gnn-madqn_gatedge` | 43 | gatedge | argmax | `embb_p5_mbps` | 1.900294 | 1.850572 | 1.900294 |
| `gnn-madqn_gatedge` | 43 | gatedge | argmax | `timely_throughput_mbps` | 69.121138 | 67.122110 | 69.121138 |
| `gnn-madqn_gatedge` | 43 | gatedge | argmax | `sla_satisfaction_pct` | 90.910374 | 88.346423 | 90.910374 |
| `gnn-madqn_gatedge` | 44 | gatedge | argmax | `embb_p5_mbps` | 0.385521 | 0.000003 | 0.385831 |
| `gnn-madqn_gatedge` | 44 | gatedge | argmax | `timely_throughput_mbps` | 67.635760 | 70.706689 | 67.678720 |
| `gnn-madqn_gatedge` | 44 | gatedge | argmax | `sla_satisfaction_pct` | 89.363696 | 92.697836 | 89.391647 |
| `gnn-madqn_gatedge` | 45 | gatedge | argmax | `embb_p5_mbps` | 0.193030 | 0.000003 | 0.192830 |
| `gnn-madqn_gatedge` | 45 | gatedge | argmax | `timely_throughput_mbps` | 69.384095 | 69.656710 | 69.389795 |
| `gnn-madqn_gatedge` | 45 | gatedge | argmax | `sla_satisfaction_pct` | 91.393058 | 91.660176 | 91.366710 |
| `gnn-madqn_gatedge` | 46 | gatedge | argmax | `embb_p5_mbps` | 0.666361 | 0.190484 | 0.667356 |
| `gnn-madqn_gatedge` | 46 | gatedge | argmax | `timely_throughput_mbps` | 68.908528 | 69.110781 | 68.723586 |
| `gnn-madqn_gatedge` | 46 | gatedge | argmax | `sla_satisfaction_pct` | 90.827142 | 91.032221 | 90.644707 |
| `gnn-madqn_gatres-edge` | 42 | gatres-edge | argmax | `embb_p5_mbps` | 0.000036 | 0.000003 | 0.000036 |
| `gnn-madqn_gatres-edge` | 42 | gatres-edge | argmax | `timely_throughput_mbps` | 70.803329 | 70.690906 | 70.801065 |
| `gnn-madqn_gatres-edge` | 42 | gatres-edge | argmax | `sla_satisfaction_pct` | 92.863232 | 92.842279 | 92.858394 |
| `gnn-madqn_gatres-edge` | 43 | gatres-edge | argmax | `embb_p5_mbps` | 1.765269 | 1.176290 | 1.657260 |
| `gnn-madqn_gatres-edge` | 43 | gatres-edge | argmax | `timely_throughput_mbps` | 69.268552 | 70.614380 | 69.304421 |
| `gnn-madqn_gatres-edge` | 43 | gatres-edge | argmax | `sla_satisfaction_pct` | 91.175670 | 93.244559 | 91.213917 |
| `gnn-madqn_gatres-edge` | 44 | gatres-edge | argmax | `embb_p5_mbps` | 1.057113 | 0.000006 | 0.963279 |
| `gnn-madqn_gatres-edge` | 44 | gatres-edge | argmax | `timely_throughput_mbps` | 67.668174 | 67.358705 | 67.931690 |
| `gnn-madqn_gatres-edge` | 44 | gatres-edge | argmax | `sla_satisfaction_pct` | 89.252977 | 89.281184 | 89.592671 |
| `gnn-madqn_gatres-edge` | 45 | gatres-edge | argmax | `embb_p5_mbps` | 0.000028 | 0.000003 | 0.000030 |
| `gnn-madqn_gatres-edge` | 45 | gatres-edge | argmax | `timely_throughput_mbps` | 68.241620 | 71.569022 | 67.927584 |
| `gnn-madqn_gatres-edge` | 45 | gatres-edge | argmax | `sla_satisfaction_pct` | 90.116982 | 93.942767 | 89.794276 |
| `gnn-madqn_gatres-edge` | 46 | gatres-edge | argmax | `embb_p5_mbps` | 0.361237 | 0.000003 | 0.360391 |
| `gnn-madqn_gatres-edge` | 46 | gatres-edge | argmax | `timely_throughput_mbps` | 68.204034 | 69.668854 | 68.113632 |
| `gnn-madqn_gatres-edge` | 46 | gatres-edge | argmax | `sla_satisfaction_pct` | 90.060986 | 91.475150 | 89.933937 |
| `gnn-madqn_gatres` | 42 | gatres | argmax | `embb_p5_mbps` | 1.715498 | 1.739726 | 1.908489 |
| `gnn-madqn_gatres` | 42 | gatres | argmax | `timely_throughput_mbps` | 65.569653 | 68.881840 | 65.629924 |
| `gnn-madqn_gatres` | 42 | gatres | argmax | `sla_satisfaction_pct` | 87.149811 | 90.894121 | 87.249324 |
| `gnn-madqn_gatres` | 43 | gatres | argmax | `embb_p5_mbps` | 1.900294 | 1.933945 | 1.900294 |
| `gnn-madqn_gatres` | 43 | gatres | argmax | `timely_throughput_mbps` | 69.116961 | 68.931430 | 69.115946 |
| `gnn-madqn_gatres` | 43 | gatres | argmax | `sla_satisfaction_pct` | 90.906046 | 90.694994 | 90.905048 |
| `gnn-madqn_gatres` | 44 | gatres | argmax | `embb_p5_mbps` | 0.154906 | 0.000004 | 0.000036 |
| `gnn-madqn_gatres` | 44 | gatres | argmax | `timely_throughput_mbps` | 69.435003 | 68.188038 | 69.343900 |
| `gnn-madqn_gatres` | 44 | gatres | argmax | `sla_satisfaction_pct` | 91.489169 | 90.093786 | 91.401592 |
| `gnn-madqn_gatres` | 45 | gatres | argmax | `embb_p5_mbps` | 0.000026 | 0.000003 | 0.000026 |
| `gnn-madqn_gatres` | 45 | gatres | argmax | `timely_throughput_mbps` | 70.907733 | 70.760998 | 70.614064 |
| `gnn-madqn_gatres` | 45 | gatres | argmax | `sla_satisfaction_pct` | 93.793257 | 93.069824 | 93.132126 |
| `gnn-madqn_gatres` | 46 | gatres | argmax | `embb_p5_mbps` | 0.000034 | 0.000004 | 0.158751 |
| `gnn-madqn_gatres` | 46 | gatres | argmax | `timely_throughput_mbps` | 69.374949 | 72.841126 | 69.242918 |
| `gnn-madqn_gatres` | 46 | gatres | argmax | `sla_satisfaction_pct` | 91.374905 | 95.470822 | 91.197064 |
| `gnn-mappo_gatedge` | 42 | gatedge | sampled | `embb_p5_mbps` | 0.000003 | 0.000003 | 0.000003 |
| `gnn-mappo_gatedge` | 42 | gatedge | sampled | `timely_throughput_mbps` | 71.144265 | 71.263593 | 71.144265 |
| `gnn-mappo_gatedge` | 42 | gatedge | sampled | `sla_satisfaction_pct` | 93.697239 | 93.857370 | 93.697239 |
| `gnn-mappo_gatedge` | 43 | gatedge | sampled | `embb_p5_mbps` | 0.000006 | 0.000006 | 0.000006 |
| `gnn-mappo_gatedge` | 43 | gatedge | sampled | `timely_throughput_mbps` | 69.952561 | 69.887476 | 69.952561 |
| `gnn-mappo_gatedge` | 43 | gatedge | sampled | `sla_satisfaction_pct` | 92.390750 | 92.313271 | 92.390750 |
| `gnn-mappo_gatedge` | 44 | gatedge | sampled | `embb_p5_mbps` | 0.000006 | 0.000006 | 0.000006 |
| `gnn-mappo_gatedge` | 44 | gatedge | sampled | `timely_throughput_mbps` | 70.606783 | 70.790800 | 70.608894 |
| `gnn-mappo_gatedge` | 44 | gatedge | sampled | `sla_satisfaction_pct` | 92.929281 | 93.087259 | 92.929704 |
| `gnn-mappo_gatedge` | 45 | gatedge | sampled | `embb_p5_mbps` | 0.000005 | 0.000005 | 0.000005 |
| `gnn-mappo_gatedge` | 45 | gatedge | sampled | `timely_throughput_mbps` | 70.008598 | 69.961915 | 70.008598 |
| `gnn-mappo_gatedge` | 45 | gatedge | sampled | `sla_satisfaction_pct` | 92.439709 | 92.379247 | 92.439709 |
| `gnn-mappo_gatedge` | 46 | gatedge | sampled | `embb_p5_mbps` | 0.000007 | 0.000007 | 0.000007 |
| `gnn-mappo_gatedge` | 46 | gatedge | sampled | `timely_throughput_mbps` | 70.960683 | 71.077011 | 70.949683 |
| `gnn-mappo_gatedge` | 46 | gatedge | sampled | `sla_satisfaction_pct` | 93.382596 | 93.557891 | 93.366228 |
| `gnn-mappo_gatedge` | 47 | gatedge | sampled | `embb_p5_mbps` | 0.000003 | 0.000003 | 0.000003 |
| `gnn-mappo_gatedge` | 47 | gatedge | sampled | `timely_throughput_mbps` | 70.764478 | 71.150919 | 70.736537 |
| `gnn-mappo_gatedge` | 47 | gatedge | sampled | `sla_satisfaction_pct` | 93.296503 | 93.796424 | 93.259058 |
| `gnn-mappo_gatedge` | 48 | gatedge | sampled | `embb_p5_mbps` | 0.000003 | 0.000003 | 0.000003 |
| `gnn-mappo_gatedge` | 48 | gatedge | sampled | `timely_throughput_mbps` | 73.284821 | 73.266423 | 73.284821 |
| `gnn-mappo_gatedge` | 48 | gatedge | sampled | `sla_satisfaction_pct` | 96.433338 | 96.380512 | 96.433338 |
| `gnn-mappo_gatedge` | 49 | gatedge | sampled | `embb_p5_mbps` | 0.000003 | 0.000003 | 0.000003 |
| `gnn-mappo_gatedge` | 49 | gatedge | sampled | `timely_throughput_mbps` | 70.935828 | 70.892619 | 70.935828 |
| `gnn-mappo_gatedge` | 49 | gatedge | sampled | `sla_satisfaction_pct` | 93.342961 | 93.296482 | 93.342961 |
| `gnn-mappo_gatedge` | 50 | gatedge | sampled | `embb_p5_mbps` | 0.000003 | 0.000003 | 0.000003 |
| `gnn-mappo_gatedge` | 50 | gatedge | sampled | `timely_throughput_mbps` | 72.529617 | 72.556244 | 72.529617 |
| `gnn-mappo_gatedge` | 50 | gatedge | sampled | `sla_satisfaction_pct` | 95.333339 | 95.360037 | 95.333339 |
| `gnn-mappo_gatedge` | 51 | gatedge | sampled | `embb_p5_mbps` | 0.000003 | 0.000003 | 0.000003 |
| `gnn-mappo_gatedge` | 51 | gatedge | sampled | `timely_throughput_mbps` | 71.613570 | 71.632146 | 71.613570 |
| `gnn-mappo_gatedge` | 51 | gatedge | sampled | `sla_satisfaction_pct` | 94.349695 | 94.382981 | 94.349695 |
| `gnn-mappo_gatedge` | 52 | gatedge | sampled | `embb_p5_mbps` | 0.000003 | 0.000003 | 0.000003 |
| `gnn-mappo_gatedge` | 52 | gatedge | sampled | `timely_throughput_mbps` | 73.436978 | 73.429547 | 73.436978 |
| `gnn-mappo_gatedge` | 52 | gatedge | sampled | `sla_satisfaction_pct` | 96.574879 | 96.556228 | 96.574879 |
| `gnn-mappo_gatedge` | 53 | gatedge | sampled | `embb_p5_mbps` | 1.372109 | 1.368906 | 1.372109 |
| `gnn-mappo_gatedge` | 53 | gatedge | sampled | `timely_throughput_mbps` | 69.348980 | 69.366867 | 69.348980 |
| `gnn-mappo_gatedge` | 53 | gatedge | sampled | `sla_satisfaction_pct` | 91.634724 | 91.663316 | 91.634724 |
| `gnn-mappo_gatedge` | 54 | gatedge | sampled | `embb_p5_mbps` | 0.000005 | 0.000005 | 0.000005 |
| `gnn-mappo_gatedge` | 54 | gatedge | sampled | `timely_throughput_mbps` | 70.653387 | 70.619584 | 70.653387 |
| `gnn-mappo_gatedge` | 54 | gatedge | sampled | `sla_satisfaction_pct` | 93.287276 | 93.235172 | 93.287276 |
| `gnn-mappo_gatedge` | 55 | gatedge | sampled | `embb_p5_mbps` | 0.000006 | 0.000006 | 0.000006 |
| `gnn-mappo_gatedge` | 55 | gatedge | sampled | `timely_throughput_mbps` | 70.954913 | 71.138041 | 70.922046 |
| `gnn-mappo_gatedge` | 55 | gatedge | sampled | `sla_satisfaction_pct` | 93.412669 | 93.615029 | 93.384722 |
| `gnn-mappo_gatedge` | 56 | gatedge | sampled | `embb_p5_mbps` | 0.000007 | 0.000007 | 0.000007 |
| `gnn-mappo_gatedge` | 56 | gatedge | sampled | `timely_throughput_mbps` | 70.339165 | 70.209653 | 70.343002 |
| `gnn-mappo_gatedge` | 56 | gatedge | sampled | `sla_satisfaction_pct` | 92.598297 | 92.498508 | 92.582467 |
| `gnn-mappo_gatedge` | 57 | gatedge | sampled | `embb_p5_mbps` | 0.000004 | 0.000004 | 0.000004 |
| `gnn-mappo_gatedge` | 57 | gatedge | sampled | `timely_throughput_mbps` | 70.791005 | 71.612408 | 70.799836 |
| `gnn-mappo_gatedge` | 57 | gatedge | sampled | `sla_satisfaction_pct` | 93.421210 | 94.441741 | 93.431653 |
| `gnn-mappo_gatedge` | 58 | gatedge | sampled | `embb_p5_mbps` | 0.190200 | 0.189400 | 0.190105 |
| `gnn-mappo_gatedge` | 58 | gatedge | sampled | `timely_throughput_mbps` | 70.711258 | 71.433689 | 70.823399 |
| `gnn-mappo_gatedge` | 58 | gatedge | sampled | `sla_satisfaction_pct` | 93.026159 | 93.983204 | 93.179817 |
| `gnn-mappo_gatedge` | 59 | gatedge | sampled | `embb_p5_mbps` | 0.000003 | 0.000003 | 0.000003 |
| `gnn-mappo_gatedge` | 59 | gatedge | sampled | `timely_throughput_mbps` | 72.628437 | 70.988144 | 72.597901 |
| `gnn-mappo_gatedge` | 59 | gatedge | sampled | `sla_satisfaction_pct` | 95.693344 | 93.591214 | 95.630082 |
| `gnn-mappo_gatedge` | 60 | gatedge | sampled | `embb_p5_mbps` | 0.571079 | 0.571173 | 0.571579 |
| `gnn-mappo_gatedge` | 60 | gatedge | sampled | `timely_throughput_mbps` | 70.141067 | 71.628020 | 70.092562 |
| `gnn-mappo_gatedge` | 60 | gatedge | sampled | `sla_satisfaction_pct` | 92.419199 | 94.336400 | 92.354365 |
| `gnn-mappo_gatedge` | 61 | gatedge | sampled | `embb_p5_mbps` | 0.000006 | 0.000006 | 0.000006 |
| `gnn-mappo_gatedge` | 61 | gatedge | sampled | `timely_throughput_mbps` | 70.593147 | 71.111546 | 70.648261 |
| `gnn-mappo_gatedge` | 61 | gatedge | sampled | `sla_satisfaction_pct` | 92.967555 | 93.641880 | 93.040412 |
| `gnn-mappo_gatres-edge` | 42 | gatres-edge | sampled | `embb_p5_mbps` | 1.158697 | 1.195092 | 1.158697 |
| `gnn-mappo_gatres-edge` | 42 | gatres-edge | sampled | `timely_throughput_mbps` | 70.288107 | 70.293087 | 70.288107 |
| `gnn-mappo_gatres-edge` | 42 | gatres-edge | sampled | `sla_satisfaction_pct` | 92.398365 | 92.389774 | 92.398365 |
| `gnn-mappo_gatres-edge` | 43 | gatres-edge | sampled | `embb_p5_mbps` | 0.000006 | 0.000006 | 0.000006 |
| `gnn-mappo_gatres-edge` | 43 | gatres-edge | sampled | `timely_throughput_mbps` | 70.593137 | 70.636571 | 70.558172 |
| `gnn-mappo_gatres-edge` | 43 | gatres-edge | sampled | `sla_satisfaction_pct` | 92.961129 | 93.025902 | 92.910459 |
| `gnn-mappo_gatres-edge` | 44 | gatres-edge | sampled | `embb_p5_mbps` | 0.000003 | 0.000003 | 0.000003 |
| `gnn-mappo_gatres-edge` | 44 | gatres-edge | sampled | `timely_throughput_mbps` | 72.604764 | 72.595997 | 72.604764 |
| `gnn-mappo_gatres-edge` | 44 | gatres-edge | sampled | `sla_satisfaction_pct` | 95.352701 | 95.382333 | 95.352701 |
| `gnn-mappo_gatres-edge` | 45 | gatres-edge | sampled | `embb_p5_mbps` | 0.192006 | 0.192005 | 0.192196 |
| `gnn-mappo_gatres-edge` | 45 | gatres-edge | sampled | `timely_throughput_mbps` | 70.566688 | 71.388179 | 70.645651 |
| `gnn-mappo_gatres-edge` | 45 | gatres-edge | sampled | `sla_satisfaction_pct` | 92.954654 | 93.985166 | 93.064765 |
| `gnn-mappo_gatres-edge` | 46 | gatres-edge | sampled | `embb_p5_mbps` | 0.000007 | 0.000007 | 0.000007 |
| `gnn-mappo_gatres-edge` | 46 | gatres-edge | sampled | `timely_throughput_mbps` | 70.529905 | 70.606101 | 70.529905 |
| `gnn-mappo_gatres-edge` | 46 | gatres-edge | sampled | `sla_satisfaction_pct` | 93.002063 | 93.120050 | 93.002063 |
| `gnn-mappo_gatres-edge` | 47 | gatres-edge | sampled | `embb_p5_mbps` | 0.000005 | 0.000005 | 0.000005 |
| `gnn-mappo_gatres-edge` | 47 | gatres-edge | sampled | `timely_throughput_mbps` | 72.047607 | 72.067605 | 72.047329 |
| `gnn-mappo_gatres-edge` | 47 | gatres-edge | sampled | `sla_satisfaction_pct` | 94.817208 | 94.848912 | 94.816874 |
| `gnn-mappo_gatres-edge` | 48 | gatres-edge | sampled | `embb_p5_mbps` | 0.000005 | 0.000005 | 0.000005 |
| `gnn-mappo_gatres-edge` | 48 | gatres-edge | sampled | `timely_throughput_mbps` | 70.119983 | 70.062036 | 70.113200 |
| `gnn-mappo_gatres-edge` | 48 | gatres-edge | sampled | `sla_satisfaction_pct` | 92.319289 | 92.245418 | 92.310609 |
| `gnn-mappo_gatres-edge` | 49 | gatres-edge | sampled | `embb_p5_mbps` | 0.000008 | 0.000008 | 0.000008 |
| `gnn-mappo_gatres-edge` | 49 | gatres-edge | sampled | `timely_throughput_mbps` | 72.694224 | 72.655515 | 72.694224 |
| `gnn-mappo_gatres-edge` | 49 | gatres-edge | sampled | `sla_satisfaction_pct` | 95.510097 | 95.487730 | 95.510097 |
| `gnn-mappo_gatres-edge` | 50 | gatres-edge | sampled | `embb_p5_mbps` | 0.000008 | 0.000008 | 0.000008 |
| `gnn-mappo_gatres-edge` | 50 | gatres-edge | sampled | `timely_throughput_mbps` | 70.266634 | 70.527546 | 70.246697 |
| `gnn-mappo_gatres-edge` | 50 | gatres-edge | sampled | `sla_satisfaction_pct` | 92.465912 | 92.787732 | 92.448429 |
| `gnn-mappo_gatres-edge` | 51 | gatres-edge | sampled | `embb_p5_mbps` | 0.000007 | 0.000007 | 0.000007 |
| `gnn-mappo_gatres-edge` | 51 | gatres-edge | sampled | `timely_throughput_mbps` | 71.816953 | 70.682410 | 71.715664 |
| `gnn-mappo_gatres-edge` | 51 | gatres-edge | sampled | `sla_satisfaction_pct` | 94.532694 | 92.976352 | 94.414982 |
| `gnn-mappo_gatres-edge` | 52 | gatres-edge | sampled | `embb_p5_mbps` | 0.000007 | 0.000007 | 0.000007 |
| `gnn-mappo_gatres-edge` | 52 | gatres-edge | sampled | `timely_throughput_mbps` | 71.816845 | 71.823470 | 71.816845 |
| `gnn-mappo_gatres-edge` | 52 | gatres-edge | sampled | `sla_satisfaction_pct` | 94.519063 | 94.523951 | 94.519063 |
| `gnn-mappo_gatres-edge` | 53 | gatres-edge | sampled | `embb_p5_mbps` | 0.146060 | 0.000007 | 0.154202 |
| `gnn-mappo_gatres-edge` | 53 | gatres-edge | sampled | `timely_throughput_mbps` | 70.238204 | 71.609759 | 70.218857 |
| `gnn-mappo_gatres-edge` | 53 | gatres-edge | sampled | `sla_satisfaction_pct` | 92.516851 | 94.276818 | 92.506732 |
| `gnn-mappo_gatres-edge` | 54 | gatres-edge | sampled | `embb_p5_mbps` | 0.000007 | 0.000007 | 0.000007 |
| `gnn-mappo_gatres-edge` | 54 | gatres-edge | sampled | `timely_throughput_mbps` | 71.390650 | 71.425248 | 71.390650 |
| `gnn-mappo_gatres-edge` | 54 | gatres-edge | sampled | `sla_satisfaction_pct` | 93.749418 | 93.801434 | 93.749418 |
| `gnn-mappo_gatres-edge` | 55 | gatres-edge | sampled | `embb_p5_mbps` | 1.545644 | 1.545197 | 1.545644 |
| `gnn-mappo_gatres-edge` | 55 | gatres-edge | sampled | `timely_throughput_mbps` | 70.917573 | 71.003454 | 70.917573 |
| `gnn-mappo_gatres-edge` | 55 | gatres-edge | sampled | `sla_satisfaction_pct` | 93.849570 | 93.904716 | 93.849570 |
| `gnn-mappo_gatres-edge` | 56 | gatres-edge | sampled | `embb_p5_mbps` | 0.000005 | 0.000005 | 0.000005 |
| `gnn-mappo_gatres-edge` | 56 | gatres-edge | sampled | `timely_throughput_mbps` | 71.002469 | 71.395521 | 71.018413 |
| `gnn-mappo_gatres-edge` | 56 | gatres-edge | sampled | `sla_satisfaction_pct` | 93.503176 | 94.093616 | 93.526535 |
| `gnn-mappo_gatres-edge` | 57 | gatres-edge | sampled | `embb_p5_mbps` | 1.531955 | 1.522808 | 1.531955 |
| `gnn-mappo_gatres-edge` | 57 | gatres-edge | sampled | `timely_throughput_mbps` | 70.756020 | 70.747908 | 70.756020 |
| `gnn-mappo_gatres-edge` | 57 | gatres-edge | sampled | `sla_satisfaction_pct` | 93.097331 | 93.085104 | 93.097331 |
| `gnn-mappo_gatres-edge` | 58 | gatres-edge | sampled | `embb_p5_mbps` | 0.000004 | 0.000005 | 0.000004 |
| `gnn-mappo_gatres-edge` | 58 | gatres-edge | sampled | `timely_throughput_mbps` | 72.114467 | 72.085173 | 72.114467 |
| `gnn-mappo_gatres-edge` | 58 | gatres-edge | sampled | `sla_satisfaction_pct` | 94.814875 | 94.759604 | 94.814875 |
| `gnn-mappo_gatres-edge` | 59 | gatres-edge | sampled | `embb_p5_mbps` | 0.000006 | 0.000006 | 0.000006 |
| `gnn-mappo_gatres-edge` | 59 | gatres-edge | sampled | `timely_throughput_mbps` | 70.567523 | 70.585233 | 70.567523 |
| `gnn-mappo_gatres-edge` | 59 | gatres-edge | sampled | `sla_satisfaction_pct` | 92.849513 | 92.882999 | 92.849513 |
| `gnn-mappo_gatres-edge` | 60 | gatres-edge | sampled | `embb_p5_mbps` | 0.000003 | 0.000003 | 0.000003 |
| `gnn-mappo_gatres-edge` | 60 | gatres-edge | sampled | `timely_throughput_mbps` | 71.382142 | 71.343055 | 71.382142 |
| `gnn-mappo_gatres-edge` | 60 | gatres-edge | sampled | `sla_satisfaction_pct` | 93.803308 | 93.748361 | 93.803308 |
| `gnn-mappo_gatres-edge` | 61 | gatres-edge | sampled | `embb_p5_mbps` | 1.684329 | 1.684184 | 1.684329 |
| `gnn-mappo_gatres-edge` | 61 | gatres-edge | sampled | `timely_throughput_mbps` | 69.300886 | 69.330268 | 69.300886 |
| `gnn-mappo_gatres-edge` | 61 | gatres-edge | sampled | `sla_satisfaction_pct` | 91.565815 | 91.596660 | 91.565815 |
| `gnn-mappo_gatres` | 42 | gatres | sampled | `embb_p5_mbps` | 0.000005 | 0.000005 | 0.000005 |
| `gnn-mappo_gatres` | 42 | gatres | sampled | `timely_throughput_mbps` | 71.188771 | 71.195532 | 71.188741 |
| `gnn-mappo_gatres` | 42 | gatres | sampled | `sla_satisfaction_pct` | 93.720604 | 93.726467 | 93.720437 |
| `gnn-mappo_gatres` | 43 | gatres | sampled | `embb_p5_mbps` | 1.608732 | 1.608747 | 1.608732 |
| `gnn-mappo_gatres` | 43 | gatres | sampled | `timely_throughput_mbps` | 71.052487 | 71.067890 | 71.052487 |
| `gnn-mappo_gatres` | 43 | gatres | sampled | `sla_satisfaction_pct` | 93.546109 | 93.535888 | 93.546109 |
| `gnn-mappo_gatres` | 44 | gatres | sampled | `embb_p5_mbps` | 0.000006 | 0.000005 | 0.000006 |
| `gnn-mappo_gatres` | 44 | gatres | sampled | `timely_throughput_mbps` | 70.914933 | 71.825227 | 70.914940 |
| `gnn-mappo_gatres` | 44 | gatres | sampled | `sla_satisfaction_pct` | 93.320801 | 94.451843 | 93.320801 |
| `gnn-mappo_gatres` | 45 | gatres | sampled | `embb_p5_mbps` | 0.189690 | 0.190200 | 0.189690 |
| `gnn-mappo_gatres` | 45 | gatres | sampled | `timely_throughput_mbps` | 71.755997 | 71.182668 | 71.776836 |
| `gnn-mappo_gatres` | 45 | gatres | sampled | `sla_satisfaction_pct` | 94.436757 | 93.588946 | 94.459501 |
| `gnn-mappo_gatres` | 46 | gatres | sampled | `embb_p5_mbps` | 0.000005 | 0.000005 | 0.000005 |
| `gnn-mappo_gatres` | 46 | gatres | sampled | `timely_throughput_mbps` | 70.708633 | 70.714023 | 70.708633 |
| `gnn-mappo_gatres` | 46 | gatres | sampled | `sla_satisfaction_pct` | 93.248176 | 93.255150 | 93.248176 |
| `gnn-mappo_gatres` | 47 | gatres | sampled | `embb_p5_mbps` | 0.000003 | 0.000003 | 0.000003 |
| `gnn-mappo_gatres` | 47 | gatres | sampled | `timely_throughput_mbps` | 72.844810 | 72.721203 | 72.844810 |
| `gnn-mappo_gatres` | 47 | gatres | sampled | `sla_satisfaction_pct` | 95.746879 | 95.601550 | 95.746879 |
| `gnn-mappo_gatres` | 48 | gatres | sampled | `embb_p5_mbps` | 0.000007 | 0.000006 | 0.000007 |
| `gnn-mappo_gatres` | 48 | gatres | sampled | `timely_throughput_mbps` | 71.222679 | 71.222216 | 71.222679 |
| `gnn-mappo_gatres` | 48 | gatres | sampled | `sla_satisfaction_pct` | 93.887827 | 93.852628 | 93.887827 |
| `gnn-mappo_gatres` | 49 | gatres | sampled | `embb_p5_mbps` | 1.417432 | 1.419600 | 1.417432 |
| `gnn-mappo_gatres` | 49 | gatres | sampled | `timely_throughput_mbps` | 70.572859 | 70.727484 | 70.572859 |
| `gnn-mappo_gatres` | 49 | gatres | sampled | `sla_satisfaction_pct` | 93.140867 | 93.351291 | 93.140867 |
| `gnn-mappo_gatres` | 50 | gatres | sampled | `embb_p5_mbps` | 0.818589 | 0.981473 | 0.818589 |
| `gnn-mappo_gatres` | 50 | gatres | sampled | `timely_throughput_mbps` | 70.917008 | 70.911506 | 70.917008 |
| `gnn-mappo_gatres` | 50 | gatres | sampled | `sla_satisfaction_pct` | 93.440967 | 93.430960 | 93.440967 |
| `gnn-mappo_gatres` | 51 | gatres | sampled | `embb_p5_mbps` | 1.506422 | 1.505605 | 1.506422 |
| `gnn-mappo_gatres` | 51 | gatres | sampled | `timely_throughput_mbps` | 70.365552 | 70.348001 | 70.365552 |
| `gnn-mappo_gatres` | 51 | gatres | sampled | `sla_satisfaction_pct` | 92.545665 | 92.521451 | 92.545665 |
| `gnn-mappo_gatres` | 52 | gatres | sampled | `embb_p5_mbps` | 0.190086 | 0.000008 | 0.190086 |
| `gnn-mappo_gatres` | 52 | gatres | sampled | `timely_throughput_mbps` | 70.499271 | 70.868757 | 70.482688 |
| `gnn-mappo_gatres` | 52 | gatres | sampled | `sla_satisfaction_pct` | 92.818595 | 93.314718 | 92.806903 |
| `gnn-mappo_gatres` | 53 | gatres | sampled | `embb_p5_mbps` | 0.000003 | 0.000003 | 0.000003 |
| `gnn-mappo_gatres` | 53 | gatres | sampled | `timely_throughput_mbps` | 72.258763 | 72.287336 | 72.258763 |
| `gnn-mappo_gatres` | 53 | gatres | sampled | `sla_satisfaction_pct` | 95.177430 | 95.216424 | 95.177430 |
| `gnn-mappo_gatres` | 54 | gatres | sampled | `embb_p5_mbps` | 0.000003 | 0.000003 | 0.000003 |
| `gnn-mappo_gatres` | 54 | gatres | sampled | `timely_throughput_mbps` | 70.920023 | 70.940701 | 70.920023 |
| `gnn-mappo_gatres` | 54 | gatres | sampled | `sla_satisfaction_pct` | 93.278302 | 93.294189 | 93.278302 |
| `gnn-mappo_gatres` | 55 | gatres | sampled | `embb_p5_mbps` | 1.611932 | 1.612477 | 1.611932 |
| `gnn-mappo_gatres` | 55 | gatres | sampled | `timely_throughput_mbps` | 70.384265 | 70.450724 | 70.384265 |
| `gnn-mappo_gatres` | 55 | gatres | sampled | `sla_satisfaction_pct` | 92.960686 | 93.037381 | 92.960686 |
| `gnn-mappo_gatres` | 56 | gatres | sampled | `embb_p5_mbps` | 0.000005 | 0.000005 | 0.000005 |
| `gnn-mappo_gatres` | 56 | gatres | sampled | `timely_throughput_mbps` | 71.355107 | 71.407054 | 71.355107 |
| `gnn-mappo_gatres` | 56 | gatres | sampled | `sla_satisfaction_pct` | 93.981085 | 93.992524 | 93.981085 |
| `gnn-mappo_gatres` | 57 | gatres | sampled | `embb_p5_mbps` | 0.000004 | 0.000004 | 0.000004 |
| `gnn-mappo_gatres` | 57 | gatres | sampled | `timely_throughput_mbps` | 70.869191 | 70.865103 | 70.869191 |
| `gnn-mappo_gatres` | 57 | gatres | sampled | `sla_satisfaction_pct` | 93.222837 | 93.217540 | 93.222837 |
| `gnn-mappo_gatres` | 58 | gatres | sampled | `embb_p5_mbps` | 0.000005 | 0.000005 | 0.000005 |
| `gnn-mappo_gatres` | 58 | gatres | sampled | `timely_throughput_mbps` | 71.544183 | 71.421431 | 71.544183 |
| `gnn-mappo_gatres` | 58 | gatres | sampled | `sla_satisfaction_pct` | 94.157266 | 94.002614 | 94.157266 |
| `gnn-mappo_gatres` | 59 | gatres | sampled | `embb_p5_mbps` | 0.177752 | 0.000008 | 0.177752 |
| `gnn-mappo_gatres` | 59 | gatres | sampled | `timely_throughput_mbps` | 69.814196 | 69.914917 | 69.814196 |
| `gnn-mappo_gatres` | 59 | gatres | sampled | `sla_satisfaction_pct` | 92.077757 | 92.186992 | 92.077757 |
| `gnn-mappo_gatres` | 60 | gatres | sampled | `embb_p5_mbps` | 0.000008 | 0.000007 | 0.000008 |
| `gnn-mappo_gatres` | 60 | gatres | sampled | `timely_throughput_mbps` | 70.435091 | 70.204358 | 70.431132 |
| `gnn-mappo_gatres` | 60 | gatres | sampled | `sla_satisfaction_pct` | 92.668553 | 92.422830 | 92.663558 |
| `gnn-mappo_gatres` | 61 | gatres | sampled | `embb_p5_mbps` | 1.671301 | 1.661627 | 1.671301 |
| `gnn-mappo_gatres` | 61 | gatres | sampled | `timely_throughput_mbps` | 68.601344 | 68.567706 | 68.601344 |
| `gnn-mappo_gatres` | 61 | gatres | sampled | `sla_satisfaction_pct` | 90.589059 | 90.557353 | 90.589059 |

`embb_p5_mbps` / D2A: largest absolute change 1.057108 (99.999% of the un-ablated value), median +0.000000, checkpoints moved by >1% of their own value: 39/75.

`embb_p5_mbps` / D2B: largest absolute change 0.192991 (467518.643% of the un-ablated value), median +0.000000, checkpoints moved by >1% of their own value: 12/75.

`timely_throughput_mbps` / D2A: largest absolute change 6.522043 (10.005% of the un-ablated value), median -0.018576, checkpoints moved by >1% of their own value: 17/75.

`timely_throughput_mbps` / D2B: largest absolute change 0.314037 (0.460% of the un-ablated value), median +0.000000, checkpoints moved by >1% of their own value: 0/75.

`sla_satisfaction_pct` / D2A: largest absolute change 7.491532 (8.652% of the un-ablated value), median -0.028592, checkpoints moved by >1% of their own value: 17/75.

`sla_satisfaction_pct` / D2B: largest absolute change 0.661131 (0.705% of the un-ablated value), median +0.000000, checkpoints moved by >1% of their own value: 0/75.

## D3 -- over-smoothing

Mean pairwise cosine similarity of the final node embeddings, with the same statistic on the raw observation alongside. The embedding number alone decides nothing: if the inputs are already near-identical, near-identical outputs are not the GNN's doing. State read after 50 policy steps, because `reset()` zeroes 7 of the 8 observation columns.

**The inter-gNB graph is complete, so its diameter is 1** -- one layer already reaches every node and the two configured layers aggregate the identical neighbour set twice. That raises the over-smoothing risk rather than lowering it, the opposite of what PLAN-01 D3 and PLAN-03 §4 assume when they call the diameter *small*.

| algo | seed | backbone | cos(embedding) | cos(obs) reference |
|---|---|---|---|---|
| `gnn-madqn_gatedge` | 42 | gatedge | 1.0000 | 0.8068 |
| `gnn-madqn_gatedge` | 43 | gatedge | 1.0000 | 0.8115 |
| `gnn-madqn_gatedge` | 44 | gatedge | 1.0000 | 0.9711 |
| `gnn-madqn_gatedge` | 45 | gatedge | 1.0000 | 0.9732 |
| `gnn-madqn_gatedge` | 46 | gatedge | 1.0000 | 0.8586 |
| `gnn-madqn_gatres-edge` | 42 | gatres-edge | 0.9994 | 0.9763 |
| `gnn-madqn_gatres-edge` | 43 | gatres-edge | 0.9999 | 0.8552 |
| `gnn-madqn_gatres-edge` | 44 | gatres-edge | 0.9988 | 0.7989 |
| `gnn-madqn_gatres-edge` | 45 | gatres-edge | 0.9999 | 0.9714 |
| `gnn-madqn_gatres-edge` | 46 | gatres-edge | 0.9999 | 0.7801 |
| `gnn-madqn_gatres` | 42 | gatres | 1.0000 | 0.7762 |
| `gnn-madqn_gatres` | 43 | gatres | 0.9999 | 0.8115 |
| `gnn-madqn_gatres` | 44 | gatres | 0.9990 | 0.9752 |
| `gnn-madqn_gatres` | 45 | gatres | 0.9995 | 0.9751 |
| `gnn-madqn_gatres` | 46 | gatres | 0.9986 | 0.8109 |
| `gnn-mappo_gatedge` | 42 | gatedge | 1.0000 | 0.9763 |
| `gnn-mappo_gatedge` | 43 | gatedge | 1.0000 | 0.7938 |
| `gnn-mappo_gatedge` | 44 | gatedge | 1.0000 | 0.8443 |
| `gnn-mappo_gatedge` | 45 | gatedge | 1.0000 | 0.9829 |
| `gnn-mappo_gatedge` | 46 | gatedge | 1.0000 | 0.8609 |
| `gnn-mappo_gatedge` | 47 | gatedge | 1.0000 | 0.8360 |
| `gnn-mappo_gatedge` | 48 | gatedge | 1.0000 | 0.9844 |
| `gnn-mappo_gatedge` | 49 | gatedge | 1.0000 | 0.9763 |
| `gnn-mappo_gatedge` | 50 | gatedge | 1.0000 | 0.9689 |
| `gnn-mappo_gatedge` | 51 | gatedge | 1.0000 | 0.9844 |
| `gnn-mappo_gatedge` | 52 | gatedge | 1.0000 | 0.9844 |
| `gnn-mappo_gatedge` | 53 | gatedge | 1.0000 | 0.7833 |
| `gnn-mappo_gatedge` | 54 | gatedge | 1.0000 | 0.9844 |
| `gnn-mappo_gatedge` | 55 | gatedge | 1.0000 | 0.8401 |
| `gnn-mappo_gatedge` | 56 | gatedge | 1.0000 | 0.8423 |
| `gnn-mappo_gatedge` | 57 | gatedge | 0.9999 | 0.7860 |
| `gnn-mappo_gatedge` | 58 | gatedge | 1.0000 | 0.9704 |
| `gnn-mappo_gatedge` | 59 | gatedge | 1.0000 | 0.9762 |
| `gnn-mappo_gatedge` | 60 | gatedge | 1.0000 | 0.7846 |
| `gnn-mappo_gatedge` | 61 | gatedge | 1.0000 | 0.9763 |
| `gnn-mappo_gatres-edge` | 42 | gatres-edge | 0.9999 | 0.8445 |
| `gnn-mappo_gatres-edge` | 43 | gatres-edge | 0.9992 | 0.7966 |
| `gnn-mappo_gatres-edge` | 44 | gatres-edge | 1.0000 | 0.9762 |
| `gnn-mappo_gatres-edge` | 45 | gatres-edge | 0.9999 | 0.9844 |
| `gnn-mappo_gatres-edge` | 46 | gatres-edge | 1.0000 | 0.8777 |
| `gnn-mappo_gatres-edge` | 47 | gatres-edge | 1.0000 | 0.9844 |
| `gnn-mappo_gatres-edge` | 48 | gatres-edge | 0.9999 | 0.8777 |
| `gnn-mappo_gatres-edge` | 49 | gatres-edge | 1.0000 | 0.9844 |
| `gnn-mappo_gatres-edge` | 50 | gatres-edge | 0.9991 | 0.8149 |
| `gnn-mappo_gatres-edge` | 51 | gatres-edge | 0.9998 | 0.8098 |
| `gnn-mappo_gatres-edge` | 52 | gatres-edge | 1.0000 | 0.9844 |
| `gnn-mappo_gatres-edge` | 53 | gatres-edge | 0.9998 | 0.8123 |
| `gnn-mappo_gatres-edge` | 54 | gatres-edge | 1.0000 | 0.8625 |
| `gnn-mappo_gatres-edge` | 55 | gatres-edge | 1.0000 | 0.9844 |
| `gnn-mappo_gatres-edge` | 56 | gatres-edge | 0.9996 | 0.8274 |
| `gnn-mappo_gatres-edge` | 57 | gatres-edge | 1.0000 | 0.8275 |
| `gnn-mappo_gatres-edge` | 58 | gatres-edge | 1.0000 | 0.9844 |
| `gnn-mappo_gatres-edge` | 59 | gatres-edge | 1.0000 | 0.8777 |
| `gnn-mappo_gatres-edge` | 60 | gatres-edge | 1.0000 | 0.9763 |
| `gnn-mappo_gatres-edge` | 61 | gatres-edge | 1.0000 | 0.8777 |
| `gnn-mappo_gatres` | 42 | gatres | 1.0000 | 0.8115 |
| `gnn-mappo_gatres` | 43 | gatres | 1.0000 | 0.8445 |
| `gnn-mappo_gatres` | 44 | gatres | 0.9989 | 0.8166 |
| `gnn-mappo_gatres` | 45 | gatres | 1.0000 | 0.8665 |
| `gnn-mappo_gatres` | 46 | gatres | 1.0000 | 0.8777 |
| `gnn-mappo_gatres` | 47 | gatres | 1.0000 | 0.9844 |
| `gnn-mappo_gatres` | 48 | gatres | 1.0000 | 0.9844 |
| `gnn-mappo_gatres` | 49 | gatres | 1.0000 | 0.9844 |
| `gnn-mappo_gatres` | 50 | gatres | 1.0000 | 0.8609 |
| `gnn-mappo_gatres` | 51 | gatres | 1.0000 | 0.8276 |
| `gnn-mappo_gatres` | 52 | gatres | 0.9989 | 0.8067 |
| `gnn-mappo_gatres` | 53 | gatres | 1.0000 | 0.9844 |
| `gnn-mappo_gatres` | 54 | gatres | 1.0000 | 0.8116 |
| `gnn-mappo_gatres` | 55 | gatres | 1.0000 | 0.8665 |
| `gnn-mappo_gatres` | 56 | gatres | 1.0000 | 0.9844 |
| `gnn-mappo_gatres` | 57 | gatres | 1.0000 | 0.7836 |
| `gnn-mappo_gatres` | 58 | gatres | 0.9999 | 0.8441 |
| `gnn-mappo_gatres` | 59 | gatres | 1.0000 | 0.8777 |
| `gnn-mappo_gatres` | 60 | gatres | 0.9997 | 0.7852 |
| `gnn-mappo_gatres` | 61 | gatres | 0.9999 | 0.7836 |
