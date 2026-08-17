# B3 diagnostic — is `urllc_delay_p99` censored by the deadline drop?

Readout: primary per family — sampled for PPO (P3), argmax for DQN (2026-08-16). 5 held-out episodes per checkpoint, seeds from `EVAL_SEED_BASE=10000`.

The previous version of this file read the DQN family at epsilon=1.0, i.e. uniform random actions rather than the policy; those four rows were void and have been recomputed here (`results/quarantine_eps1.0/README.md`).

**This does not change Gate B3.** The threshold stays >= 2 ms and B3 stays FAILED (measured range 1.01 ms, from `results\GATE_B_v4_primary.md`). This file only explains the mechanism.


Deadline `slices.urllc.max_delay_ms` = 10.0 ms, slot = 1.0 ms, so a delivered packet's delay can only take **11 values** ({0, 1, ..., 10} ms) and nothing above 10 ms can ever be observed: `envs/network_slicing_env.py:333` pops over-age packets before service, so they are counted as drops, not as slow deliveries.


| algo | p99 (ms) | mass @ 10 ms | mass >= 9 ms | delivered | censored (drops) | late:ovf |
|---|---|---|---|---|---|---|
| `central-dqn` | 9.480 | 9.55% | 11.24% | 1,347,766 | 9.09% | 134700:0 |
| `central-ppo` | 10.000 | 14.59% | 15.43% | 1,330,475 | 10.03% | 148282:0 |
| `gnn-madqn_gat` | 9.840 | 12.51% | 14.59% | 1,312,454 | 11.18% | 165166:0 |
| `gnn-madqn_sage` | 10.000 | 14.06% | 14.93% | 1,338,159 | 9.58% | 141851:0 |
| `gnn-mappo_gat` | 9.880 | 5.08% | 8.42% | 1,376,590 | 6.93% | 102542:0 |
| `gnn-mappo_sage` | 9.680 | 3.90% | 6.52% | 1,388,622 | 6.26% | 92787:0 |
| `idqn` | 9.480 | 17.84% | 19.39% | 1,209,851 | 17.65% | 259281:0 |
| `ippo` | 9.080 | 4.02% | 6.79% | 1,393,555 | 6.05% | 89782:0 |

Spread of p99 across algorithms: **0.920 ms** (min 9.080, max 10.000). One slot = 1 ms, so B3's 2 ms threshold asks for at least **2 lattice steps** of separation in a statistic whose whole support is 11 steps wide and whose upper end is a hard wall.


Read the `censored` column together with the p99 column: the packets that would have formed the discriminating tail are exactly the ones removed from the sample. Two policies with very different queueing behaviour report near-identical p99 while differing on `sla_satisfaction_pct` (which counts the drops) -- that is B2 passing at 11.98 pp while B3 fails at 1.01 ms, and it is one phenomenon seen through two metrics.


Methodological consequence for the paper: at this operating point `urllc_delay_p99` of delivered packets is not a discriminating KPI, and the honest latency statement lives in the drop/SLA metrics instead. Reported, not claimed (`handoff/goal1.md`, scoping decision 2026-08-15).
