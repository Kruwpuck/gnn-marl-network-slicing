# B3 diagnostic — is `urllc_delay_p99` censored by the deadline drop?

Readout: stochastic (P3 primary), 5 held-out episodes per checkpoint, seeds from `EVAL_SEED_BASE=10000`.

**This does not change Gate B3.** The threshold stays >= 2 ms and B3 stays FAILED (measured range 1.01 ms). This file only explains the mechanism.


Deadline `slices.urllc.max_delay_ms` = 10.0 ms, slot = 1.0 ms, so a delivered packet's delay can only take **11 values** ({0, 1, ..., 10} ms) and nothing above 10 ms can ever be observed: `envs/network_slicing_env.py:333` pops over-age packets before service, so they are counted as drops, not as slow deliveries.


| algo | p99 (ms) | mass @ 10 ms | mass >= 9 ms | delivered | censored (drops) | late:ovf |
|---|---|---|---|---|---|---|
| `central-dqn` | 10.000 | 17.88% | 21.41% | 1,266,337 | 13.81% | 202839:0 |
| `central-ppo` | 10.000 | 14.59% | 15.43% | 1,330,475 | 10.03% | 148282:0 |
| `gnn-madqn_gat` | 10.000 | 6.30% | 10.06% | 1,369,351 | 7.49% | 110807:0 |
| `gnn-madqn_sage` | 10.000 | 6.19% | 9.35% | 1,367,500 | 7.61% | 112635:0 |
| `gnn-mappo_gat` | 9.880 | 5.08% | 8.42% | 1,376,590 | 6.93% | 102542:0 |
| `gnn-mappo_sage` | 9.680 | 3.90% | 6.52% | 1,388,622 | 6.26% | 92787:0 |
| `idqn` | 10.000 | 6.22% | 9.79% | 1,368,815 | 7.54% | 111579:0 |
| `ippo` | 9.080 | 4.02% | 6.79% | 1,393,555 | 6.05% | 89782:0 |

Spread of p99 across algorithms: **0.920 ms** (min 9.080, max 10.000). One slot = 1 ms, so B3's 2 ms threshold asks for at least **2 lattice steps** of separation in a statistic whose whole support is 11 steps wide and whose upper end is a hard wall.


Read the `censored` column together with the p99 column: the packets that would have formed the discriminating tail are exactly the ones removed from the sample. Two policies with very different queueing behaviour report near-identical p99 while differing on `sla_satisfaction_pct` (which counts the drops) -- that is B2 passing at 8.70 pp while B3 fails at 1.01 ms, and it is one phenomenon seen through two metrics.


Methodological consequence for the paper: at this operating point `urllc_delay_p99` of delivered packets is not a discriminating KPI, and the honest latency statement lives in the drop/SLA metrics instead. Reported, not claimed (`handoff/goal1.md`, scoping decision 2026-08-15).
