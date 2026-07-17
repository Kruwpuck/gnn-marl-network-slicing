[← Run pertama (v1)](02_run1-v1-uncoupled.md) | [Index](00_INDEX.md) | [Revisi 2 (v3 CMDP) →](04_revisi2-v3-cmdp.md)

# 03 — Revisi 1: Env v2 (Coupled Interference)

**Commit:** `8a78a22` (2026-07-16)
**Config:** `results/v2_scalarized/`

## Recap masalah dari v1

1. Environment uncoupled → GNN tidak punya nilai tambah teoritis (gNB sudah optimal independen).
2. Reward clip `[-2,+2]` menjenuhkan penalti delay besar → gradien hilang saat starvation.
3. URLLC starvation (delay meledak, tak terikat).
4. Single seed.

## Solusi yang diterapkan di v2

| Perubahan | Detail |
|---|---|
| **Slice-coupled interference** | Interferensi antar-gNB kini terikat pada fraksi alokasi slice tetangga (reuse-1 realistis) — keputusan satu gNB benar-benar memengaruhi SINR gNB lain, mengaktifkan kebutuhan koordinasi yang jadi dasar hipotesis GNN |
| **Observasi dinamis** | SINR, delay, dan alokasi tetangga dimasukkan ke observation vector secara dinamis (bukan hampir statis seperti v1) |
| **Log-scale delay penalty** | Term `w3 * mean(delay/max_delay)` diganti `w3 * log1p(delay_ratio)` — gradien tidak jenuh saat delay besar |
| **Reward clip diperlebar** | `[-2,+2]` → `[-10,+10]` |

Reward weights tetap w1=0.4, w2=0.3, w3=0.2, w4=0.1; traffic dan budget training (DQN 200K, PPO 1M, seed=42) **tidak berubah** — supaya perbandingan v1-vs-v2 murni mengisolasi efek coupling + reward fix.

## Hasil (seed 42, 8 run)

| algo | family | reward/step (final 10%) | SLA sat. % | eMBB Mbps | delay ms mean | Jain's fairness |
|---|---|---|---|---|---|---|
| `gnn-mappo_gat` | proposed | **0.9840** | 97.1 | 6.898 | 15,913 | 0.347 |
| `ippo` | baseline | 0.8163 | 97.0 | 5.893 | 167,135 | 0.343 |
| `gnn-madqn_sage` | proposed | 0.6732 | 98.4 | 4.752 | 84,796 | 0.477 |
| `gnn-mappo_sage` | proposed | 0.5867 | 97.5 | 4.172 | 17,183 | 0.435 |
| `gnn-madqn_gat` | proposed | 0.5428 | 98.4 | 3.855 | 80,985 | 0.493 |
| `central-dqn` | baseline | 0.5361 | 98.5 | 3.786 | 3,590 | 0.497 |
| `central-ppo` | baseline | 0.5257 | 98.5 | 3.705 | 3,628 | 0.503 |
| `idqn` | baseline | 0.3466 | 94.1 | 3.365 | 932,183 | 0.532 |

**Pemenang keluarga DQN: `gnn-madqn_sage` — PROPOSED MENANG**
**Pemenang keluarga PPO: `gnn-mappo_gat` — PROPOSED MENANG**

## Perbandingan langsung v1 vs v2 (budget training identik)

| algo | reward/step v1 | reward/step v2 | delta | SLA% v1 | SLA% v2 | delay ms mean v1 | delay ms mean v2 |
|---|---|---|---|---|---|---|---|
| `central-dqn` | 0.0357 | 0.5361 | +0.5004 | 95.8 | 98.5 | 99,072 | 3,590 |
| `central-ppo` | 0.0039 | 0.5257 | +0.5218 | 96.5 | 98.5 | 12,429 | 3,628 |
| `gnn-madqn_gat` | -0.1354 | 0.5428 | +0.6782 | 86.6 | 98.4 | 15,190,804 | 80,985 |
| `gnn-madqn_sage` | -0.1593 | 0.6732 | +0.8325 | 84.3 | 98.4 | 12,691,671 | 84,796 |
| `gnn-mappo_gat` | -0.0288 | 0.9840 | +1.0128 | 95.7 | 97.1 | 174,121 | 15,913 |
| `gnn-mappo_sage` | -0.0158 | 0.5867 | +0.6025 | 96.1 | 97.5 | 45,017 | 17,183 |
| `idqn` | 0.1699 | 0.3466 | +0.1768 | 97.1 | 94.1 | 4,118,166 | 932,183 |
| `ippo` | -0.0009 | 0.8163 | +0.8172 | 97.1 | 97.0 | 17,455 | 167,135 |

**Interpretasi:** di v1, seluruh baseline mengalahkan proposed di kedua keluarga. Di v2, proposed (`gnn-mappo_gat` dan `gnn-madqn_sage`) menang di kedua keluarga — konsisten dengan hipotesis awal (file 01): GNN-MARL unggul justru saat environment benar-benar membutuhkan koordinasi antar-gNB lewat interferensi coupled, bukan saat gNB independen sudah optimal secara lokal.

## Masalah yang masih tersisa (dilaporkan jujur di v2 §8/§10)

1. **URLLC starvation masih terjadi di semua run**, lintas proposed maupun baseline. Log-scale penalty memperbaiki *pembelajaran dari kondisi starvation* (gradien tidak jenuh), tapi tidak menghilangkan starvation-nya sendiri — begitu antrean menumpuk di satu episode, delay mean tetap bisa meledak (`idqn` masih 932,183 ms mean meski SLA 94.1%).
2. **SLA proposed (97.1–98.4%) kalah tipis dari central baseline (98.5%)** — dicurigai artefak scalarization: bobot w1=0.4 (throughput) vs w2=0.3 (violation) membuat keduanya bisa saling ditukar oleh agent, bukan constraint keras.
3. **Jain's fairness rendah (0.347–0.532)** di semua run — sebagian struktural (reuse-1, gNB cell-edge kalah bersaing sinyal), butuh framing jujur + KPI cell-edge (P5 throughput) daripada rata-rata saja.
4. **Masih single seed** — belum publishable secara statistik.

## Rekomendasi (dari laporan v2)

1. ~~Revisi reward clipping~~ — sudah diterapkan (log-scale, clip lebih lebar).
2. Samakan/bandingkan budget step secara adil dalam keluarga.
3. Tambah minimal 3 seed.
4. Laporkan p95/median delay.
5. **Pertimbangkan floor alokasi PRB URLLC minimum** — starvation masih muncul meski reward sudah diperbaiki, mengindikasikan masalahnya bukan cuma bentuk reward tapi arsitektur constraint-nya.
6. Re-run multi-seed setelah floor URLLC diterapkan.

→ Poin 2, 3, 5, 6 dikerjakan lewat riset lebih dalam (compass artifact) dan redesain besar di **Revisi 2 (v3)**, lihat file 04.

## Data & artefak

- `results/v2_scalarized/RESULTS.md` — laporan lengkap termasuk §12 tabel v1-vs-v2
- `results/v2_scalarized/results_summary.csv`
- `results/v2_scalarized/figures/*.png`
- `results/v2_scalarized/logs/` — CSV mentah + checkpoint

> **Catatan validitas**: sama seperti v1, environment v2 mewarisi bug topologi UE-gNB yang sama (ditemukan saat kalibrasi v3, lihat file 05). Perbandingan v1-vs-v2 di atas tetap valid secara *internal* (mengisolasi efek coupling reward, kedua sisi memakai bug yang sama), tapi angka absolut (SLA%, delay ms) tidak representatif skenario UMa nyata dan tidak dipakai sebagai baseline akhir.

---

[← Run pertama (v1)](02_run1-v1-uncoupled.md) | [Index](00_INDEX.md) | [Revisi 2 (v3 CMDP) →](04_revisi2-v3-cmdp.md)
