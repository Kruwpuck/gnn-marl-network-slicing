[← Bug-fix & kalibrasi](05_bugfix-topologi-kalibrasi.md) | [Index](00_INDEX.md)

# 06 — Status Training v3 (Saat Ini)

> Dokumen ini akan di-update setelah training wave selesai. Status di bawah adalah kondisi terakhir sebelum wave dijalankan.

## Kode: siap, sudah ter-commit

- Commit `87374bc` (2026-07-17): env v3 lengkap (packet-level URLLC + CMDP + PRB floor + bug-fix topologi UE-gNB), 80/80 test pass, smoke test 4 kombinasi algoritma sukses.
- Gate kalibrasi (`scripts/calibrate_load.py`) lolos dengan parameter final: `ue_radius_m=100`, `lambda_arrival=25000`, `delta=0.12` (lihat file 05 untuk proses lengkap).
- Commit dilakukan **sebelum** training wave dijalankan — supaya titik kerja yang mahal untuk diulang (bug-fix + kalibrasi rigorous) sudah checkpoint aman di git sebelum training multi-hari dijalankan di atasnya.

## Training wave: belum dijalankan

Wave dijalankan manual di PC lab (GPU RTX 3060 lokal) lewat 3 command `scripts/run_wave.py`, berurutan (satu GPU, tidak paralel antar-command):

```
1. Wave utama    — 8 algoritma × 5 seed, floor=dynamic
2. Ablation A     — 4 algoritma × 5 seed, floor=none
3. Ablation B     — 4 algoritma × 5 seed, floor=static
```

Subset 4 algoritma ablation: `gnn-madqn_sage`, `gnn-mappo_gat`, `idqn`, `central-ppo` (proposed + baseline terbaik dari masing-masing keluarga, berdasarkan hasil v2).

### Estimasi durasi

| Wave | Total GPU-jam (estimasi) | Wall-clock (max-parallel=4, GPU shared) |
|---|---|---|
| 1. Utama (8×5) | ~260 jam | ~3–5 hari |
| 2. Ablation floor=none (4×5) | ~100 jam | ~1–1.7 hari |
| 3. Ablation floor=static (4×5) | ~100 jam | ~1–1.7 hari |
| **Total** | **~460 jam** | **~5–8.5 hari** |

Estimasi berbasis wall-clock terukur v2 per-seed (`gnn-madqn_gat` 23.4 jam — algoritma paling lambat karena `DQNAgent.learn()` belum divektorisasi; `idqn` 8.0 jam; `gnn-madqn_sage` 7.4 jam; `central-dqn` 5.7 jam; `gnn-mappo_gat` 3.9 jam; `gnn-mappo_sage` 2.1 jam; `ippo` 0.74 jam; `central-ppo` 0.73 jam) plus overhead ~15% dari model FIFO packet-level v3.

`run_wave.py` mendukung `--resume` — command yang terhenti bisa dilanjutkan dari checkpoint terakhir tanpa mengulang dari nol.

## Pipeline analisis yang menyusul setelah wave selesai

```
scripts/evaluate_checkpoints.py   → results/eval/*.csv       (30 episode greedy, seed held-out ≥10000)
scripts/rliable_report.py         → results/RLIABLE.md       (IQM + bootstrap CI 95%, per keluarga budget)
scripts/analyze_results.py        → results/RESULTS.md       (KPI operator: violation rate, timely throughput, P99 delay, cell-edge P5)
scripts/make_paper_figures.py     → results/figures/*.png
```

## Kriteria sukses (ditetapkan sebelum melihat hasil, supaya tidak bias)

- **CI 95% tumpang tindih antara varian GNN dan baseline sepadan** → dilaporkan sebagai "comparable", bukan "superior". Tidak ada klaim keunggulan tanpa CI yang benar-benar terpisah.
- **Keunggulan GNN (jika ada) harus bertahan pada floor yang sama** di ablation — kalau proposed hanya menang pada `floor=dynamic` dan kalah di `floor=none`, itu artinya floor yang bekerja, bukan GNN, dan akan dilaporkan apa adanya.
- **Tidak ada angka yang disetel per-algoritma** — seluruh hyperparameter bersama (delta, lambda_lr, floor, traffic) ditulis satu kali di config yang di-commit, sama untuk semua 8 algoritma.
- Uji identitas perlakuan: aksi identik pada seed sama antara satu algoritma proposed dan satu baseline harus menghasilkan `(floor_applied, lam, delta, violation_rate)` yang identik — bukti eksplisit constraint/floor tidak menguntungkan proposed secara struktural.

## Di luar scope wave ini (ditunda)

- **α-fair utility sweep** — KPI cell-edge P5 throughput sudah masuk logging v3 (gratis, tanpa run tambahan), jadi framing efficiency-vs-fairness sudah bisa ditulis dari data wave ini tanpa run α-fair terpisah.
- **Curriculum learning / prioritized URLLC head** — ditunda sampai hasil v3 keluar; kalau proposed sudah menang SLA-at-equal-throughput, kompleksitas tambahan ini tidak diperlukan.

## Data & artefak (akan terisi setelah wave berjalan)

- `results/logs/*.csv` — CSV mentah per (algoritma, seed, floor-mode)
- `results/eval/*.csv` — hasil evaluasi held-out
- `results/RESULTS.md`, `results/RLIABLE.md` — laporan akhir
- `results/figures/*.png` — grafik untuk paper/presentasi

---

[← Bug-fix & kalibrasi](05_bugfix-topologi-kalibrasi.md) | [Index](00_INDEX.md)
