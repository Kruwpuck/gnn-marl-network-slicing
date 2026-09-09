# Journey: GNN-MARL untuk Dynamic Resource Allocation pada Network Slicing 5G/6G

Dokumen ini adalah seri catatan perjalanan riset — dari rencana awal, tiga revisi besar environment/reward, rekalibrasi titik operasi, wave konfirmatori v4, diagnostik mekanisme, sampai wave arsitektur v6 yang masih berjalan. Ditulis supaya bisa dipakai langsung sebagai bahan presentasi ke dosen pembimbing: setiap file menjelaskan *apa yang salah*, *kenapa salah*, *apa yang diperbaiki*, dan *apa hasilnya* — bukan cuma daftar perubahan kode.

## Cara baca

Baca berurutan 01 → 11. Setiap file punya bagian **"Data & artefak"** di akhir yang menunjuk ke file CSV/figure/config asli — bukan angka yang diketik ulang dari ingatan. Sumbernya bertambah seiring jalur riset: `RESULTS.md` dari `scripts/analyze_results.py` untuk v1–v3, lalu laporan rliable / stability / gate / diagnostik yang masing-masing di-generate skripnya sendiri untuk v4 ke atas.

> **File 11 belum memuat hasil v6.** Wave DQN v6 masih berjalan dan nol KPI v6 dibaca, jadi file itu berhenti di keadaan operasional. Hasilnya jadi file berikutnya.

| # | File | Isi |
|---|---|---|
| 01 | [`01_rencana-awal.md`](01_rencana-awal.md) | Skema penelitian awal: problem statement, hipotesis, 8 algoritma, formulasi matematis, stack tools, hyperparameter |
| 02 | [`02_run1-v1-uncoupled.md`](02_run1-v1-uncoupled.md) | Run pertama (env v1, uncoupled interference) — hasil dan masalah yang ditemukan |
| 03 | [`03_revisi1-v2-coupled.md`](03_revisi1-v2-coupled.md) | Revisi 1: coupled interference + log-scale reward — hasil dan masalah tersisa |
| 04 | [`04_revisi2-v3-cmdp.md`](04_revisi2-v3-cmdp.md) | Revisi 2: redesain CMDP + packet-level URLLC + PRB floor — desain dan justifikasi |
| 05 | [`05_bugfix-topologi-kalibrasi.md`](05_bugfix-topologi-kalibrasi.md) | Bug-fix topologi UE-gNB (ditemukan saat gate kalibrasi v3) + proses kalibrasi ulang |
| 06 | [`06_status-training-v3.md`](06_status-training-v3.md) | Rencana training wave v3 + pipeline analisis + kriteria sukses pra-registrasi |
| 07 | [`07_hasil-evaluasi-v3.md`](07_hasil-evaluasi-v3.md) | **Hasil akhir**: 80 run selesai, IQM + bootstrap CI, saturasi KPI, temuan cell-edge, ablation floor, batasan |
| 08 | [`08_rekalibrasi-titik-operasi.md`](08_rekalibrasi-titik-operasi.md) | Rev 2 Fase 0–1, enam ronde Gate A, titik operasi dibekukan, protokol pembacaan P3 |
| 09 | [`09_wave-v4-dan-cacat-pembacaan.md`](09_wave-v4-dan-cacat-pembacaan.md) | Wave v4 40 run, tiga cacat pembacaan dalam satu wave, Gate B3 & C4 gagal, perluasan ke 20 seed |
| 10 | [`10_diagnostik-fase-0.md`](10_diagnostik-fase-0.md) | D1–D6 atas checkpoint v4: enam premis basi gugur, over-smoothing terkonfirmasi, `conv1` membuang 98,6% separasi |
| 11 | [`11_v5-diblokir-dan-wave-v6.md`](11_v5-diblokir-dan-wave-v6.md) | **Berjalan**: `f_min` tanpa kandidat jadi temuan, wave v6 tiga arm arsitektur, cacat instrumen #5–#7 — nol KPI v6 dibaca |

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
| 2026-08-06 | `aad4198` | **Titik operasi dibekukan** (`delta=0,085`, `lambda_arrival=60000`, `buffer=307200`, `dual_update_every=12500`, `floor.mode=none`) — sebelum checkpoint v4 pertama ditulis |
| 2026-08-08 | `528a726` | **Protokol pembacaan P3 dibekukan** dan Gate A lolos seluruhnya — keduanya sebelum ada hasil per-algoritma |
| 2026-08-09 → 08-14 | `44ad631` | **Wave v4**: 40 run (8 algoritma × 5 seed), 147,22 jam device-time, 40/40 OK |
| 2026-08-17 | `de481aa`, `9d43f6f` | Readout keluarga DQN dikoreksi ke argmax; 40 file ε=1,0 **dikarantina**; audit provenance pembacaan dibuat |
| 2026-08-18 | `78bbc34` | Keluarga PPO diperluas ke 20 seed — **dua temuan bergerak melawan narasi** dan tetap dilaporkan |
| 2026-08-24 → 08-25 | `8cf6db9`, `4177cc2`, `2cb78ba` | **Diagnostik Fase 0** D1–D6 atas checkpoint v4, nol training baru |
| 2026-08-25 | `bf54500` | Kalibrasi `f_min` tanpa kandidat **dicatat sebagai temuan**; urutan estafet ditukar |
| 2026-08-26 | `1a127f7`, `f1ec666` | Arm v6 (`gatres`, `gatedge`, `gatres-edge`); pembanding `gat` dipakai ulang, digerbangi uji identitas bit-per-bit |
| 2026-08-27 → 08-30 | — | **Wave PPO v6**: 60 job (3 arm × 20 seed) selesai dalam 71,27 jam, nol gagal |
| 2026-08-31 | `187b349` | Wave DQN v6 dilepas (15 job) — **berjalan saat dokumen ini ditulis**, nol KPI v6 dibaca |

## Peta lokasi data di repo

```
gnn-marl-network-slicing/
├── results/
│   ├── v1_uncoupled/        # ARSIP historis — env-nya cacat (lihat 05), tidak dipakai sebagai pembanding
│   ├── v2_scalarized/       # ARSIP historis — idem
│   ├── RESULTS.md           # KPI operator training-time v3 (72 run)
│   ├── RLIABLE.md           # IQM + bootstrap CI wave v3; RLIABLE_floornone/floorstatic untuk ablation
│   ├── RLIABLE_v4_primary.md    # wave v4, pembacaan primer per keluarga — sumber sah klaim statistik v4
│   ├── STABILITY_v4_primary.md  # collapse rate + Wilson CI + CVaR, n=20 untuk keluarga PPO
│   ├── GATE_B_v4_primary.md     # Gate B dihitung ulang (B3 GAGAL, ambang tidak diamandemen)
│   ├── GATE_C.md                # checklist validitas C1–C6 (C2 PARSIAL, C4 per keluarga)
│   ├── READOUT_PROVENANCE.md    # tiap laporan wajib menyatakan protokol pembacaannya
│   ├── READOUT_COMPARISON.md    # greedy vs non-greedy per algoritma
│   ├── ZEROSHOT_v4_primary.md   # transfer topologi 10/20 gNB, dua arm
│   ├── DIAG_*.md                # D1–D6 (equivariance, reliance, grad ratio, collision, separability)
│   ├── CALIBRATE_FMIN.md        # kalibrasi f_min — "no candidate", dilaporkan sebagai temuan
│   ├── quarantine_eps1.0/       # 40 file readout cacat — DILARANG dipakai di laporan mana pun
│   ├── logs/                # CSV training + checkpoint final .pt (v3, _v4, _v6)
│   ├── eval/                # CSV evaluasi held-out, dua readout (seed >= 10000)
│   ├── eval_zeroshot_v4/    # grid zero-shot v4
│   ├── figures/             # grafik + tabel paper
│   └── checkpoints/         # checkpoint periodik _best/_last
├── configs/experiment_config.yaml   # config aktif; komentar inline memuat aritmetika tiap nilai
├── envs/network_slicing_env.py      # environment
├── gnn/                             # backbone: gat (GATv2), sage, plus arm v6 gatres/gatedge/gatres-edge
├── scripts/diag_*.py                # diagnostik D1–D6
├── docs/revisi/                     # PLAN-00..07, PREREG-V5, PREREG-V6, HANDOVER-V6
├── runs/2026-08-05-run01/ledger.md  # ledger append-only: tiap keputusan bertanda waktu
├── STATE.json                       # status resmi estafet
└── docs/journey/                    # seri dokumen ini
```

> **Catatan penting** (dijelaskan detail di file 05): hasil v1 dan v2 dihasilkan dari environment yang punya bug pemodelan posisi UE-gNB. Keduanya **tidak representatif sebagai skenario 3GPP UMa yang valid** dan tidak dipakai sebagai baseline pembanding di laporan akhir — hanya diarsipkan untuk menunjukkan proses iterasi.

> **Status artefak `_v4` berbeda dari kedua arsip itu.** Sejak 2026-08-26 checkpoint `_v4` adalah **pembanding aktif** wave v6 — arm `gat` tidak dilatih ulang, dan keputusan itu digerbangi uji identitas numerik bit-per-bit (file 11 §3). Artefak `_v4` karena itu tidak boleh ditimpa, dipindah, atau dibersihkan.

## Stack & hardware (konsisten v1 sampai v6, diverifikasi ulang terhadap `.venv` aktif 2026-08-31 — nol versi bergeser)

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
