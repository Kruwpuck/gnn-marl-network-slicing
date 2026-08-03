# Journey: GNN-MARL untuk Dynamic Resource Allocation pada Network Slicing 5G/6G

Dokumen ini adalah seri catatan perjalanan riset — dari rencana awal, tiga revisi besar environment/reward, sampai status training saat ini. Ditulis supaya bisa dipakai langsung sebagai bahan presentasi ke dosen pembimbing: setiap file menjelaskan *apa yang salah*, *kenapa salah*, *apa yang diperbaiki*, dan *apa hasilnya* — bukan cuma daftar perubahan kode.

## Cara baca

Baca berurutan 01 → 07. Setiap file punya bagian **"Data & artefak"** di akhir yang menunjuk ke file CSV/figure/config asli — bukan angka yang diketik ulang dari ingatan, semua disalin langsung dari laporan `RESULTS.md` yang di-generate otomatis oleh `scripts/analyze_results.py` di tiap tahap.

| # | File | Isi |
|---|---|---|
| 01 | [`01_rencana-awal.md`](01_rencana-awal.md) | Skema penelitian awal: problem statement, hipotesis, 8 algoritma, formulasi matematis, stack tools, hyperparameter |
| 02 | [`02_run1-v1-uncoupled.md`](02_run1-v1-uncoupled.md) | Run pertama (env v1, uncoupled interference) — hasil dan masalah yang ditemukan |
| 03 | [`03_revisi1-v2-coupled.md`](03_revisi1-v2-coupled.md) | Revisi 1: coupled interference + log-scale reward — hasil dan masalah tersisa |
| 04 | [`04_revisi2-v3-cmdp.md`](04_revisi2-v3-cmdp.md) | Revisi 2: redesain CMDP + packet-level URLLC + PRB floor — desain dan justifikasi |
| 05 | [`05_bugfix-topologi-kalibrasi.md`](05_bugfix-topologi-kalibrasi.md) | Bug-fix topologi UE-gNB (ditemukan saat gate kalibrasi v3) + proses kalibrasi ulang |
| 06 | [`06_status-training-v3.md`](06_status-training-v3.md) | Rencana training wave v3 + pipeline analisis + kriteria sukses pra-registrasi |
| 07 | [`07_hasil-evaluasi-v3.md`](07_hasil-evaluasi-v3.md) | **Hasil akhir**: 80 run selesai, IQM + bootstrap CI, saturasi KPI, temuan cell-edge, ablation floor, batasan |

## Timeline

| Tanggal | Commit | Milestone |
|---|---|---|
| 2026-06-26 | `5414451`, `824a6ca`, `ab28aa1`, `dedf8a3`, ... | Skeleton environment, agents, GNN backbones, test suite awal |
| 2026-07-02 | `5eba80d`, `5e3e4ab`, `046b724` | Notebook training cloud/Colab, perbaikan skip-logic |
| 2026-07-13 | `2363711` | Metrics logging kaya, resumable checkpoint, perbaikan MLP-PPO |
| 2026-07-13 | `98bc371` | Arsip hasil run pertama (v1) ke `results/v1_uncoupled/` |
| 2026-07-16 | `8a78a22` | **Revisi 1**: env v2 coupled interference — proposed menang di dua keluarga algoritma |
| 2026-07-16 | `84fa150` | Housekeeping .gitignore |
| 2026-07-17 | `87374bc` | **Revisi 2**: env v3 — packet-level URLLC + CMDP + PRB floor, plus bug-fix topologi UE-gNB |
| 2026-07-20 → 08-01 | — | Training wave v3 dijalankan di PC lab: 80 run (8 algoritma × 5 seed + 2 ablation × 4 algoritma × 5 seed), ±12 hari wall-clock |
| 2026-08-03 | — | **Pipeline analisis selesai**: evaluasi held-out + IQM/bootstrap CI + figures — hasil dan interpretasi di file 07 |

## Peta lokasi data di repo

```
gnn-marl-network-slicing/
├── results/
│   ├── v1_uncoupled/        # arsip run pertama — HANYA referensi historis, env-nya cacat (lihat 05)
│   │   ├── RESULTS.md
│   │   ├── results_summary.csv
│   │   └── figures/*.png
│   ├── v2_scalarized/       # arsip revisi 1 — HANYA referensi historis, env-nya cacat (lihat 05)
│   │   ├── RESULTS.md
│   │   ├── results_summary.csv
│   │   └── figures/*.png
│   ├── RESULTS.md           # KPI operator training-time v3 (72 run) — diagnosis konvergensi
│   ├── RLIABLE.md           # IQM + bootstrap CI, wave utama — sumber sah klaim statistik
│   ├── RLIABLE_floornone.md # idem, ablation A (floor=none)
│   ├── RLIABLE_floorstatic.md # idem, ablation B (floor=static)
│   ├── logs/                # CSV training v3, 80 run (lihat file 07 §1.3 soal seed 42)
│   ├── eval/                # 80 CSV evaluasi held-out v3 (30 episode greedy, seed >=10000)
│   ├── figures/             # 11 grafik + paper_metrics_long.csv + summary_table.csv
│   └── checkpoints/         # model .pt tersimpan
├── configs/experiment_config.yaml   # config aktif (v3)
├── envs/network_slicing_env.py      # environment (v3, dengan docstring bug-fix)
├── scripts/diag_*.py                # script diagnosis yang dipakai selama kalibrasi v3
└── docs/journey/                    # seri dokumen ini
```

> **Catatan penting** (dijelaskan detail di file 05): hasil v1 dan v2 dihasilkan dari environment yang punya bug pemodelan posisi UE-gNB. Keduanya **tidak representatif sebagai skenario 3GPP UMa yang valid** dan tidak dipakai sebagai baseline pembanding di laporan akhir — hanya diarsipkan untuk menunjukkan proses iterasi.

## Stack & hardware (konsisten di seluruh v1-v3, terverifikasi pada environment training aktif)

| Komponen | Versi |
|---|---|
| Python | 3.11.9 |
| PyTorch | 2.4.1+cu124 |
| PyTorch Geometric | 2.8.0 |
| Gymnasium | 1.1.0 |
| NumPy | 1.26.4 |
| Pandas | 2.2.3 (diturunkan dari 3.0.3 pada 2026-08-03 — lihat file 07 §1.4) |
| rliable | 1.2.0 (dipakai mulai v3, untuk IQM + bootstrap CI multi-seed) |
| arch | 7.2.0 (versi 8.0.0 tidak kompatibel dengan rliable 1.2.0 — lihat file 07 §1.4) |
| SciPy | 1.13.0 |
| Matplotlib | 3.11.0 |
| GPU | NVIDIA RTX 3060 |
| CPU | 24 core |
| OS | Windows 11 |
