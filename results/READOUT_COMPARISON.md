# Readout comparison — greedy vs stochastic (protocol P3)

Tag `_v4`, held-out evaluation, mean over all seeds x episodes. The stochastic column is the gating readout (`handoff/goal1.md` P3, frozen 2026-08-08, before any v4 result existed); the greedy column is reported, never used to gate, and never omitted.


## KPI under each readout

| algo | KPI | greedy | stochastic | stoch − greedy |
|---|---|---|---|---|
| `central-dqn` | timely_throughput_mbps | 64.3389 | 61.8710 | -2.4679 |
| `central-dqn` | sla_satisfaction_pct | 85.1645 | 82.4237 | -2.7407 |
| `central-dqn` | urllc_delay_p99 | 8.1253 | 8.9773 | +0.8520 |
| `central-dqn` | embb_p5_mbps | 0.8259 | 0.0265 | -0.7994 |
| `central-ppo` | timely_throughput_mbps | 65.2811 | 64.2696 | -1.0115 |
| `central-ppo` | sla_satisfaction_pct | 86.3129 | 85.1345 | -1.1785 |
| `central-ppo` | urllc_delay_p99 | 7.5560 | 7.9827 | +0.4267 |
| `central-ppo` | embb_p5_mbps | 1.7492 | 0.4337 | -1.3154 |
| `gnn-madqn_gat` | timely_throughput_mbps | 63.2810 | 67.0903 | +3.8093 |
| `gnn-madqn_gat` | sla_satisfaction_pct | 83.9557 | 88.8302 | +4.8745 |
| `gnn-madqn_gat` | urllc_delay_p99 | 8.3131 | 8.7133 | +0.4002 |
| `gnn-madqn_gat` | embb_p5_mbps | 0.6399 | 0.0000 | -0.6399 |
| `gnn-madqn_sage` | timely_throughput_mbps | 64.5659 | 67.0799 | +2.5140 |
| `gnn-madqn_sage` | sla_satisfaction_pct | 85.4417 | 88.7986 | +3.3570 |
| `gnn-madqn_sage` | urllc_delay_p99 | 8.2520 | 8.7600 | +0.5080 |
| `gnn-madqn_sage` | embb_p5_mbps | 1.5582 | 0.0000 | -1.5582 |
| `gnn-mappo_gat` | timely_throughput_mbps | 22.6947 | 67.9670 | +45.2723 |
| `gnn-mappo_gat` | sla_satisfaction_pct | 33.8009 | 90.0813 | +56.2803 |
| `gnn-mappo_gat` | urllc_delay_p99 | 4.7280 | 8.9967 | +4.2687 |
| `gnn-mappo_gat` | embb_p5_mbps | 1.5702 | 0.2805 | -1.2897 |
| `gnn-mappo_sage` | timely_throughput_mbps | 40.1689 | 68.9401 | +28.7712 |
| `gnn-mappo_sage` | sla_satisfaction_pct | 55.6089 | 91.1192 | +35.5103 |
| `gnn-mappo_sage` | urllc_delay_p99 | 6.6707 | 8.5525 | +1.8818 |
| `gnn-mappo_sage` | embb_p5_mbps | 1.3765 | 0.0000 | -1.3765 |
| `idqn` | timely_throughput_mbps | 59.2646 | 67.0633 | +7.7987 |
| `idqn` | sla_satisfaction_pct | 79.1375 | 88.7897 | +9.6522 |
| `idqn` | urllc_delay_p99 | 8.2880 | 8.7359 | +0.4479 |
| `idqn` | embb_p5_mbps | 1.0610 | 0.0000 | -1.0610 |
| `ippo` | timely_throughput_mbps | 68.1519 | 68.6121 | +0.4602 |
| `ippo` | sla_satisfaction_pct | 89.8484 | 90.6415 | +0.7932 |
| `ippo` | urllc_delay_p99 | 6.7853 | 8.2520 | +1.4667 |
| `ippo` | embb_p5_mbps | 0.0000 | 0.0000 | -0.0000 |

## Signs of a degenerate greedy readout

`sd_ep` = per-episode sd of `sla_violation_pct`. A readout that has stopped responding to state shows a collapsed sd (P3 measured 0.11 pp against a normal 11-14 pp). `agree` = fraction of steps where the argmax equals the action the policy actually sampled; `agree` < 0.5 means the argmax is not the policy's behaviour. Entropy is in nats against the `ln(n_actions)` ceiling.


| algo | sd_ep greedy | sd_ep stoch | p_max | 1/n | entropy | ln n | agree |
|---|---|---|---|---|---|---|---|
| `central-dqn` | 15.51 | 15.00 | — | — | — | — | — (DQN: no action distribution) |
| `central-ppo` | 13.80 | 14.25 | 0.217 | 0.091 | 2.191 | 2.398 | 0.215 |
| `gnn-madqn_gat` | 15.41 | 12.09 | — | — | — | — | — (DQN: no action distribution) |
| `gnn-madqn_sage` | 14.16 | 12.07 | — | — | — | — | — (DQN: no action distribution) |
| `gnn-mappo_gat` | 34.95 | 11.04 | 0.200 | 0.091 | 2.193 | 2.398 | 0.202 |
| `gnn-mappo_sage` | 37.82 | 10.45 | 0.203 | 0.091 | 2.217 | 2.398 | 0.200 |
| `idqn` | 19.66 | 12.00 | — | — | — | — | — (DQN: no action distribution) |
| `ippo` | 13.42 | 11.37 | 0.202 | 0.091 | 2.291 | 2.398 | 0.198 |

Among the algorithms with an action distribution, entropy spans only 0.100 nats (2.191-2.291, ceiling 2.398) while their throughput readout gap spans 46.28 Mbps (-1.01 to +45.27). The argmax is statistically degenerate to the same degree everywhere; whether that degeneracy wrecks the measured KPI is architecture-dependent. Entropy therefore does not predict which model the greedy readout will misrepresent — which is precisely why P3 gates on the stochastic readout rather than screening greedy numbers for plausibility.


Largest single-KPI readout gap in this wave: **45.27 Mbps** on `timely_throughput_mbps`. A gap that size is not a rounding difference between two ways of printing the same policy — it is the difference between measuring the policy and measuring its mode. Which architectures collapse and which do not is reported above as a finding, not filtered out.
