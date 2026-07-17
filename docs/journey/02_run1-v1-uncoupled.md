[← Rencana awal](01_rencana-awal.md) | [Index](00_INDEX.md) | [Revisi 1 (v2 coupled) →](03_revisi1-v2-coupled.md)

# 02 — Run Pertama: Env v1 (Uncoupled Interference)

**Commit terkait:** `98bc371` (arsip) — berdasarkan skeleton awal `5414451` dkk (2026-06-26).
**Config:** `results/v1_uncoupled/` (arsip); lihat tabel di bawah untuk nilai kunci.

## Setup

Environment v1 memodelkan 5 gNB dengan interferensi antar-gNB yang **tidak bergantung pada alokasi tetangga** ("uncoupled") — setiap gNB efektif optimal secara independen tanpa perlu koordinasi eksplisit dengan tetangga.

| Parameter | Nilai |
|---|---|
| n_gnb | 5 |
| bandwidth | 10 MHz |
| skenario | UMa |
| area_size | 500 m |
| URLLC max delay (SLA bound) | 10.0 ms |
| eMBB min throughput | 4.0 Mbps |
| reward weights | w1=0.4 (eMBB rate), w2=0.3 (URLLC violation), w3=0.2 (URLLC delay ratio, linear), w4=0.1 (spectral efficiency) |
| reward clip | `[-2, +2]` |
| traffic URLLC | lambda_arrival=50 (pkt/step, implisit 1 slot = 1 detik) |
| traffic eMBB | lambda_burst=4.0 Mbps, lambda_idle=0.5 Mbps |
| budget | DQN 200,000 step; PPO 1,000,000 step; seed=42 (single seed) |

## Hasil (seed 42, 8 run)

| algo | family | reward/step (final 10%) | SLA sat. % | eMBB Mbps | delay ms mean | Jain's fairness |
|---|---|---|---|---|---|---|
| `idqn` | baseline | **0.1699** | 97.1 | 3.057 | 4,118,166 | 0.428 |
| `central-dqn` | baseline | 0.0357 | 95.8 | 2.757 | 99,072 | 0.489 |
| `central-ppo` | baseline | 0.0039 | 96.5 | 2.564 | 12,429 | 0.475 |
| `ippo` | baseline | -0.0009 | 97.1 | 1.957 | 17,455 | 0.422 |
| `gnn-mappo_sage` | proposed | -0.0158 | 96.1 | 2.499 | 45,017 | 0.449 |
| `gnn-mappo_gat` | proposed | -0.0288 | 95.7 | 2.619 | 174,121 | 0.455 |
| `gnn-madqn_gat` | proposed | -0.1354 | 86.6 | 2.812 | 15,190,804 | 0.486 |
| `gnn-madqn_sage` | proposed | -0.1593 | 84.3 | 2.836 | 12,691,671 | 0.461 |

**Pemenang keluarga DQN (budget setara, 200K step): `idqn` — BASELINE MENANG**
**Pemenang keluarga PPO (budget setara, 1M step): `central-ppo` — BASELINE MENANG**

## Diagnosis masalah

Empat masalah ditemukan dari analisis `scripts/analyze_results.py`:

### 1. Environment uncoupled → GNN tidak punya nilai tambah teoritis

Karena interferensi antar-gNB tidak bergantung pada alokasi tetangga, setiap gNB **sudah optimal secara independen** — tidak ada insentif nyata untuk koordinasi. GNN, yang secara desain menambah kompleksitas untuk memodelkan hubungan antar-node, kehilangan alasan untuk unggul: hipotesis di file 01 ("GNN menang saat koordinasi dibutuhkan") tidak bisa diuji dengan setup ini. Hasilnya konsisten — baseline non-graph menang di kedua keluarga.

### 2. URLLC starvation — delay meledak jauh melampaui SLA bound 10 ms

`gnn-madqn_gat` dan `gnn-madqn_sage` menunjukkan delay mean **15.2 juta ms** dan **12.7 juta ms** — jauh dari bound 10 ms. Root cause di kode: `urllc_delay_ms = queue_urllc / max(urllc_rate, 1.0) * 1000`. Saat agent mengalokasikan PRB URLLC mendekati 0%, `urllc_rate` jatuh ke lantai `1.0` bps sehingga delay menjadi `queue_bits * 1000` ms dan meledak seiring antrean menumpuk tanpa batas (tidak ada packet drop / deadline enforcement di v1 — antrean fluida murni).

### 3. Reward clip `[-2, +2]` menjenuhkan penalti delay

Term `-w3 * mean(delay / max_delay)` seharusnya menghukum delay besar secara proporsional, tapi `np.clip(r, -2.0, 2.0)` membuat reward mentok di batas bawah begitu delay sudah katastrofik — memburuk lebih jauh tidak menambah hukuman, gradien belajar hilang, agent kehilangan insentif untuk keluar dari kondisi starvation. Ini kandidat cacat desain reward, bukan sekadar kekurangan kapasitas model GNN.

### 4. Single seed — tidak cukup untuk klaim statistik

Semua hasil di atas dari satu seed (42). Selisih antar-algoritma yang tampak di tabel belum bisa diklaim signifikan secara statistik tanpa multi-seed dan uji signifikansi (mis. bootstrap CI).

## Rekomendasi (dari laporan v1)

1. Revisi reward clipping — naikkan batas atau ganti penalti delay ke skala log supaya gradien tidak hilang.
2. Samakan/bandingkan budget step DQN vs PPO secara adil (dalam keluarga, bukan lintas keluarga).
3. Tambah minimal 3 seed untuk error bar dan uji signifikansi.
4. Laporkan p95/median delay, bukan hanya mean (distribusi delay sangat skewed).
5. Pertimbangkan floor alokasi PRB URLLC minimum supaya agent tidak bisa menstarve URLLC sepenuhnya.
6. Re-run `gnn-madqn` setelah perbaikan #1 dan #5.

→ Poin 1 dan (implisit) coupling interferensi dikerjakan di **Revisi 1 (v2)**, lihat file 03.

## Data & artefak

- `results/v1_uncoupled/RESULTS.md` — laporan lengkap (11 bagian)
- `results/v1_uncoupled/results_summary.csv` — tabel machine-readable
- `results/v1_uncoupled/figures/*.png` — 11 grafik (reward curve, delay curve, SLA curve, dll.)
- `results/v1_uncoupled/logs/` — CSV mentah per-run + checkpoint

> **Catatan validitas** (lengkap di file 05): environment v1 punya bug independen dari 4 masalah di atas — posisi UE tidak dikorelasikan ke gNB penyedia layanannya, sehingga skenario ini bukan simulasi UMa yang valid secara fisik. Hasil di atas tetap sahih untuk *menunjukkan proses iterasi reward/starvation*, tapi tidak dipakai sebagai baseline pembanding kuantitatif di laporan akhir.

---

[← Rencana awal](01_rencana-awal.md) | [Index](00_INDEX.md) | [Revisi 1 (v2 coupled) →](03_revisi1-v2-coupled.md)
