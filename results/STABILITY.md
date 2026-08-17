# Stability Report — collapse rate over embb_p5_mbps

> **Readout: `greedy` (wave v3).** Label added 2026-08-17 by the provenance audit
> (`scripts/readout_audit.py`), and verified rather than assumed: rerunning
> `stability_report.py --tag "" --readout greedy` reproduces every number in the table
> below exactly, while `--readout stochastic` finds no rows at all for this tag — wave v3
> was evaluated greedy-only, before protocol P3 was frozen on 2026-08-08. v3 is reported
> as a calibration pilot (D1), not as a gate.

Tag filter: `(main wave)`. Collapse threshold: 0.01 Mbps. Unit of collapse is the seed (mean over its held-out episodes), not the episode.

| algo | seeds collapsed | rate | 95% Wilson CI | worst seed mean | CVaR@20% |
|---|---|---|---|---|---|
| `central-dqn` | 0/5 | 0.00 | [0.00, 0.43] | 0.254387 | 0.000038 |
| `central-ppo` | 0/5 | 0.00 | [0.00, 0.43] | 1.773975 | 1.147688 |
| `gnn-madqn_gat` | 1/5 | 0.20 | [0.04, 0.62] | 0.000038 | 0.000037 |
| `gnn-madqn_sage` | 0/5 | 0.00 | [0.00, 0.43] | 1.772099 | 1.175351 |
| `gnn-mappo_gat` | 1/5 | 0.20 | [0.04, 0.62] | 0.000038 | 0.000038 |
| `gnn-mappo_sage` | 0/5 | 0.00 | [0.00, 0.43] | 1.772099 | 1.181356 |
| `idqn` | 1/5 | 0.20 | [0.04, 0.62] | 0.000004 | 0.000004 |
| `ippo` | 5/5 | 1.00 | [0.57, 1.00] | 0.000002 | 0.000001 |

## Power analysis — N seed untuk Fase 3

Base rate kolaps terukur di algoritma non-`ippo` yang pernah kolaps
(`gnn-madqn_gat`, `gnn-mappo_gat`, `idqn`): **1/5 = 0,20**. Wilson 95% CI di N=5
untuk observasi ini adalah **[0,04, 0,62]** — lebar 0,59, tidak bisa membedakan
"kolaps jarang" dari "kolaps hampir separuh waktu".

Lebar CI Wilson pada base rate ~0,20–0,30, berbagai N:

| N seed | rate~0,20 | rate~0,30 |
|---|---|---|
| 5 (v3) | [0,04, 0,62] lebar 0,59 | [0,12, 0,77] lebar 0,65 |
| **10 (Fase 3)** | **[0,06, 0,51] lebar 0,45** | **[0,11, 0,60] lebar 0,50** |
| 15 | [0,07, 0,45] lebar 0,38 | [0,11, 0,52] lebar 0,41 |
| 20 | [0,08, 0,42] lebar 0,34 | [0,15, 0,52] lebar 0,37 |
| 30 | [0,10, 0,37] lebar 0,28 | [0,17, 0,48] lebar 0,31 |

**Dilaporkan apa adanya:** naik dari 5→10 seed (grid Fase 3 terpilih) memperkecil
lebar CI dari ~0,6 ke ~0,45–0,50 — perbaikan nyata tapi belum sempit. Mencapai
lebar <0,35 butuh N≈20-30 per algoritma per titik grid, di luar anggaran
GPU yang disepakati (§6.3 panduan implementasi). Dengan 10 seed, collapse-rate
Fase 3 akan tetap dilaporkan dengan CI lebar — itu bukan kegagalan analisis,
itu batasan anggaran yang harus dinyatakan eksplisit di prareg (§6.4).