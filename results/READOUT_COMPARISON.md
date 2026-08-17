# Readout comparison — greedy vs non-greedy (protocol P3)

Tag `_v4`, held-out evaluation, mean over all seeds x episodes.

**The primary readout differs by family, and the non-greedy column is not the same quantity in both.** For PPO, P3 (`handoff/goal1.md`, frozen 2026-08-08 before any v4 result existed) makes the sampled readout primary. For DQN there is no action distribution to sample, and P3 never defined one; the human determination of 2026-08-16 makes argmax primary after a pre-registered diagnostic, with epsilon=0.05 (`epsilon_min`, the exploration floor the policy actually behaved under) reported beside it. Whichever column is primary is shown in **bold**; the other is reported and never gates.

Non-greedy source: PPO from `results/eval`, DQN from `results/eval_dqn_eps005`. The earlier DQN non-greedy files were epsilon=1.0, i.e. uniform random actions rather than the policy, and are quarantined (`results/quarantine_eps1.0/README.md`).


## KPI under each readout

| algo | KPI | greedy | non-greedy | non-greedy − greedy |
|---|---|---|---|---|
| `central-dqn` | timely_throughput_mbps | **64.3389** | 64.3053 | -0.0335 |
| `central-dqn` | sla_satisfaction_pct | **85.1645** | 85.1397 | -0.0248 |
| `central-dqn` | urllc_delay_p99 | **8.1253** | 8.1813 | +0.0560 |
| `central-dqn` | embb_p5_mbps | **0.8259** | 0.7820 | -0.0439 |
| `central-ppo` | timely_throughput_mbps | 65.2811 | **64.2696** | -1.0115 |
| `central-ppo` | sla_satisfaction_pct | 86.3129 | **85.1345** | -1.1785 |
| `central-ppo` | urllc_delay_p99 | 7.5560 | **7.9827** | +0.4267 |
| `central-ppo` | embb_p5_mbps | 1.7492 | **0.4337** | -1.3154 |
| `gnn-madqn_gat` | timely_throughput_mbps | **63.2810** | 63.4541 | +0.1731 |
| `gnn-madqn_gat` | sla_satisfaction_pct | **83.9557** | 84.2067 | +0.2510 |
| `gnn-madqn_gat` | urllc_delay_p99 | **8.3131** | 8.4547 | +0.1416 |
| `gnn-madqn_gat` | embb_p5_mbps | **0.6399** | 0.6349 | -0.0049 |
| `gnn-madqn_sage` | timely_throughput_mbps | **64.5659** | 64.8048 | +0.2389 |
| `gnn-madqn_sage` | sla_satisfaction_pct | **85.4417** | 85.7558 | +0.3142 |
| `gnn-madqn_sage` | urllc_delay_p99 | **8.2520** | 8.2240 | -0.0280 |
| `gnn-madqn_sage` | embb_p5_mbps | **1.5582** | 1.5385 | -0.0197 |
| `gnn-mappo_gat` | timely_throughput_mbps | 22.6947 | **67.9670** | +45.2723 |
| `gnn-mappo_gat` | sla_satisfaction_pct | 33.8009 | **90.0813** | +56.2803 |
| `gnn-mappo_gat` | urllc_delay_p99 | 4.7280 | **8.9967** | +4.2687 |
| `gnn-mappo_gat` | embb_p5_mbps | 1.5702 | **0.2805** | -1.2897 |
| `gnn-mappo_sage` | timely_throughput_mbps | 40.1689 | **68.9401** | +28.7712 |
| `gnn-mappo_sage` | sla_satisfaction_pct | 55.6089 | **91.1192** | +35.5103 |
| `gnn-mappo_sage` | urllc_delay_p99 | 6.6707 | **8.5525** | +1.8818 |
| `gnn-mappo_sage` | embb_p5_mbps | 1.3765 | **0.0000** | -1.3765 |
| `idqn` | timely_throughput_mbps | **59.2646** | 60.8114 | +1.5468 |
| `idqn` | sla_satisfaction_pct | **79.1375** | 81.0766 | +1.9392 |
| `idqn` | urllc_delay_p99 | **8.2880** | 8.2825 | -0.0055 |
| `idqn` | embb_p5_mbps | **1.0610** | 1.0463 | -0.0147 |
| `ippo` | timely_throughput_mbps | 68.1519 | **68.6121** | +0.4602 |
| `ippo` | sla_satisfaction_pct | 89.8484 | **90.6415** | +0.7932 |
| `ippo` | urllc_delay_p99 | 6.7853 | **8.2520** | +1.4667 |
| `ippo` | embb_p5_mbps | 0.0000 | **0.0000** | -0.0000 |
| `mlp-knn-ppo` | timely_throughput_mbps | 42.1123 | **68.1937** | +26.0814 |
| `mlp-knn-ppo` | sla_satisfaction_pct | 58.0398 | **90.2615** | +32.2217 |
| `mlp-knn-ppo` | urllc_delay_p99 | 9.0747 | **8.7707** | -0.3040 |
| `mlp-knn-ppo` | embb_p5_mbps | 0.6858 | **0.0000** | -0.6858 |

## Signs of a degenerate greedy readout

`sd_ep` = per-episode sd of `sla_violation_pct`. A readout that has stopped responding to state shows a collapsed sd (P3 measured 0.11 pp against a normal 11-14 pp); a readout locked into a degenerate trajectory shows the opposite, an inflated one. `agree` = fraction of steps where the argmax equals the action the policy actually sampled; `agree` < 0.5 means the argmax is not the policy's behaviour. Entropy is in nats against the `ln(n_actions)` ceiling.

The `sd_ep` ratio is what decided the DQN family, and it was declared before being measured: greedy/non-greedy above 2.0 would have invalidated argmax as primary. PPO separates cleanly on it (0.97 and 1.18 against 3.17 and 3.62); the four DQN algorithms came in at 1.01-1.10, none close to the threshold. `p_max`, entropy and `agree` are undefined for DQN — it has Q-values, not a distribution — which is exactly why an outcome-variance test was used instead of an action-distribution one.


| algo | sd_ep greedy | sd_ep non-greedy | p_max | 1/n | entropy | ln n | agree |
|---|---|---|---|---|---|---|---|
| `central-dqn` | 15.51 | 15.39 | — | — | — | — | — (DQN: no action distribution) |
| `central-ppo` | 13.80 | 14.25 | 0.217 | 0.091 | 2.191 | 2.398 | 0.215 |
| `gnn-madqn_gat` | 15.41 | 15.22 | — | — | — | — | — (DQN: no action distribution) |
| `gnn-madqn_sage` | 14.16 | 14.04 | — | — | — | — | — (DQN: no action distribution) |
| `gnn-mappo_gat` | 34.95 | 11.04 | 0.200 | 0.091 | 2.193 | 2.398 | 0.202 |
| `gnn-mappo_sage` | 37.82 | 10.45 | 0.203 | 0.091 | 2.217 | 2.398 | 0.200 |
| `idqn` | 19.66 | 17.85 | — | — | — | — | — (DQN: no action distribution) |
| `ippo` | 13.42 | 11.37 | 0.202 | 0.091 | 2.291 | 2.398 | 0.198 |
| `mlp-knn-ppo` | 33.08 | 11.12 | — | — | — | — | — (not in the policy_confidence sweep) |

Among the algorithms with an action distribution, entropy spans only 0.100 nats (2.191-2.291, ceiling 2.398) while their throughput readout gap spans 46.28 Mbps (-1.01 to +45.27). The argmax is statistically degenerate to the same degree everywhere; whether that degeneracy wrecks the measured KPI is architecture-dependent. Entropy therefore does not predict which model the greedy readout will misrepresent — which is precisely why P3 gates on the stochastic readout rather than screening greedy numbers for plausibility.


Largest single-KPI readout gap in this wave: **45.27 Mbps** on `timely_throughput_mbps`. A gap that size is not a rounding difference between two ways of printing the same policy — it is the difference between measuring the policy and measuring its mode. Which architectures collapse and which do not is reported above as a finding, not filtered out.


## A wrong readout protocol produces rows that look valid

This wave produced two independent instances of the same failure mode, and neither announced itself:

1. **Argmax on PPO.** Statistically degenerate to the same degree in all four PPO algorithms (entropy spans 0.100 nats), yet the KPI damage spans tens of Mbps. Read greedy, `gnn-mappo_gat` would have been reported as losing badly to `ippo`; read sampled, the same weights are COMPARABLE. Nothing cheap predicts which model gets wrecked.

2. **epsilon=1.0 on DQN.** Uniform random actions reported as the policy for a whole family. Aggregate KPIs barely moved — which is why it survived review — while one seed's cell-edge collapse verdict flipped. It was caught only because the same fault let `central-dqn` emit 150 rows per checkpoint at a topology where its `obs_dim` makes it structurally impossible to run: the network was never called, so nothing raised.

The methodological claim is therefore not "gate on the stochastic readout" but the stronger one: **the readout protocol is a pre-registration decision on the same footing as the choice of metric**, because a wrong one yields output that passes every plausibility check a reader can apply after the fact.
