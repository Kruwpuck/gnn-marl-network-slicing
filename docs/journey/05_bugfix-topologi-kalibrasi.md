[← Revisi 2 (v3 CMDP)](04_revisi2-v3-cmdp.md) | [Index](00_INDEX.md) | [Status training v3 →](06_status-training-v3.md)

# 05 — Bug-fix Topologi UE-gNB + Proses Kalibrasi v3

Bagian ini mendokumentasikan sesuatu yang ditemukan **di tengah** implementasi v3, bukan direncanakan sejak awal — sebuah bug pemodelan kanal yang sudah ada sejak v1, tidak terdeteksi selama dua revisi sebelumnya, dan baru terungkap karena v3 menambahkan gate kalibrasi wajib sebelum training besar dimulai.

## Kenapa ada gate kalibrasi

Sebelum membakar ~460 GPU-jam untuk training wave penuh (8 algoritma × 5 seed), dibuat `scripts/calibrate_load.py`: menjalankan policy statis (alokasi URLLC tetap 0%, 10%, 20%, 30%, 50%, 80%) tanpa training, mengukur violation rate, delay P95/P99, throughput eMBB. Kriteria lolos:
- Ada policy dengan violation rate di rentang 5–30% (bukti daerah "contested" — bukan trivial 0%, bukan infeasible 100%).
- Floor 0% → violation mendekati 100% (starvation masih tertangkap).
- Delay P99 paket terkirim < 10 ms (invariant deadline-drop).
- Ada policy statis yang mencapai violation ≤ delta CMDP (feasibility check).

## Gate gagal — indikasi masalah struktural, bukan salah tuning traffic

Pada percobaan pertama (setelah traffic dikalibrasi ulang ke satuan per-detik yang benar dan floor dimatikan untuk pengukuran bersih), violation rate **flat di ~27–30% untuk SEMUA level alokasi statis, dari 0% sampai 80%**. Tidak ada policy yang bisa mencapai SLA rendah — bahkan alokasi 80% URLLC pun tetap ~27% violation. Ini bukan masalah traffic terlalu tinggi (yang bisa diperbaiki dengan menurunkan lambda_arrival); ini pola **infeasibilitas struktural** — sesuatu di model kanal membuat sebagian gNB tidak pernah bisa memenuhi SLA berapapun alokasinya.

## Diagnosis root cause

### Langkah 1 — cek apakah interference-limited atau noise-limited

`scripts/diag_channel3.py` mengukur SIR (Signal-to-Interference Ratio, tanpa noise) vs SINR (dengan noise) per-gNB. Hasil: **SIR ≈ SINR di hampir semua gNB** (gap noise 20–45 dB headroom) — ini genuinely interference-limited, bukan noise-limited. Artinya menaikkan tx_power tidak akan membantu; masalahnya ada di rasio sinyal-vs-interferer.

### Langkah 2 — sweep 3 kandidat fix, 10 seed masing-masing (bukan sekadar satu arah)

`scripts/diag_topology_sweep.py` menguji tiga hipotesis independen:

| Kandidat fix | Hasil (good-SIR fraction) | Kesimpulan |
|---|---|---|
| `area_size` 500 → 1000 → 2000 m (posisi independen) | 28% → 32% → 28% | **No-op.** Uniform log-distance path-loss scaling menggeser setiap link (sinyal DAN interferer) dengan konstanta aditif yang sama — saling membatalkan di rasio SIR. Memperbesar area tidak memperbaiki apa-apa. |
| `n_gnb` 5 → 3 (posisi independen) | 28% → 33% | Marginal — bukan soal jumlah interferer. |
| **UE di-cluster dalam radius `ue_radius_m` dari gNB penyedia layanannya** (area_size/n_gnb tetap) | **28% → 92-98%** | **Ini fix yang sebenarnya.** |

### Root cause ditemukan

`NetworkSlicingEnv.reset()` menggambar posisi gNB dan posisi UE **secara independen** dari kotak `uniform(0, area_size)` yang sama. Seorang UE yang "dilayani" gNB ke-i (berdasarkan slice index array, bukan jarak) **tidak punya korelasi spasial apapun** dengan posisi gNB ke-i itu sendiri. Ini bukan skenario 3GPP UMa yang valid — UE bisa jadi lebih dekat ke gNB tetangga daripada ke gNB penyedianya sendiri, membuat SINR-nya secara struktural buruk tidak peduli alokasi PRB berapapun.

Bug ini **sudah ada sejak v1** dan terbawa ke v2 tanpa terdeteksi — kedua revisi sebelumnya memfokuskan perbaikan pada reward shaping dan interference coupling, bukan model posisi, karena starvation yang teramati (file 02, 03) tampak konsisten dengan hipotesis reward yang salah, bukan topologi yang salah.

## Fix yang diterapkan

```python
gnb_repeat = np.repeat(self._gnb_positions, self.n_ue_per_gnb, axis=0)
ue_offsets = self._rng.uniform(-self.ue_radius_m, self.ue_radius_m, (n_ue_total, 2))
self._ue_positions = np.clip(gnb_repeat + ue_offsets, 0.0, self.area_size)
```

Parameter baru `env.ue_radius_m: 100` (terpisah dari `area_size`, yang sekarang murni mengontrol jarak antar-gNB). UE diklip ke batas peta (bukan direfleksikan) supaya UE dekat tepi peta tidak jatuh di luar area.

**r=100m dipilih di atas r=50m secara sengaja**: r=50m nyaris menghilangkan kasus cell-edge (p10 SIR ~10dB — semua UE "mudah"), sementara r=100m tetap menyisakan kasus cell-edge nyata (p10 SIR ~2dB) yang dibutuhkan supaya KPI fairness (`embb_p5_mbps`, cell-edge throughput) tetap bermakna untuk dianalisis, bukan trivial.

**Interferensi itu sendiri tidak disentuh** — perbaikan diisolasi murni ke generasi posisi UE; `_ue_positions` dikonfirmasi hanya direferensikan di dalam `network_slicing_env.py`.

## Konsekuensi metodologis

Karena bug ini memengaruhi model kanal fundamental (bukan reward atau training loop), **hasil v1 dan v2 dinyatakan tidak valid sebagai representasi skenario UMa** — bukan sekadar "perlu dikalibrasi ulang". Keduanya tetap diarsipkan (`results/v1_uncoupled/`, `results/v2_scalarized/`) sebagai catatan proses iterasi (file 02, 03), tapi **tidak dipakai sebagai baseline pembanding kuantitatif** di laporan akhir. Semua klaim performa final hanya berdasarkan v3 pasca-fix.

## Kalibrasi ulang setelah fix

Dengan topologi yang benar, traffic dan parameter CMDP perlu dikalibrasi ulang dari nol.

### Sweep traffic load

`scripts/diag_traffic_sweep.py` mencari `lambda_arrival` yang menghasilkan daerah "contested" (bukan trivial, bukan infeasible). Dipilih **25,000 pkt/detik** (~6.4 Mbps/gNB offered load @ 256 bit/paket).

### Breakdown per-gNB (bukan cuma agregat)

Karena constraint CMDP bersifat **network-wide** (rata-rata violation di seluruh 5 gNB per step, bukan per-agen — lihat formula di file 04), penting memastikan violation tidak didominasi satu gNB buruk sementara yang lain sehat. `scripts/diag_breakdown.py` mengonfirmasi violation **tersebar** di semua gNB, bukan terkonsentrasi — di frac=0.8 (alokasi URLLC statis tinggi), violation rate ~5.6% (structural floor dari gNB cell-edge), sementara frac=0.1 saja menghasilkan ~13% violation (daerah contested yang jelas).

### Delta CMDP: dari kandidat 0.10 ke keputusan final 0.12

Kandidat awal delta=0.10 tampak memberi margin positif (5.6% mean vs 10% target = +4.4pp), tapi **mean saja tidak cukup** — perlu tahu variance-nya.

**Episode-level variance** (`scripts/diag_seed_variance.py`) di frac=0.8: std = **8.3–8.4pp, distribusi bimodal**. Jauh di atas ambang aman.

Tapi episode-level bukan kuantitas yang relevan — CMDP dual update ($\lambda$) bekerja pada **rata-rata window `dual_update_every`=2000 step (~10 episode)**, bukan per-episode. Sebelum mempercayai argumen "window averaging akan menghaluskan noise", perlu dikonfirmasi dulu: **apakah topologi (posisi gNB) di-resample tiap episode atau tetap sepanjang training run?** Jawaban terkonfirmasi dari kode: topologi **di-resample tiap episode** (`env.reset(seed=seed+ep_count)`) — jadi window-level averaging memang menghaluskan noise topologi antar-episode secara sah, bukan cuma menyembunyikan varians di balik agregasi.

**Window-level variance** (`scripts/diag_window_variance.py`, 20 window independen @ 2000 step): std = **1.91pp**. Dinilai masih terlalu tipis dibanding ambang keamanan 2.00pp yang ditetapkan sendiri — dengan hanya 20 window tersampel, training run 200K/1M-step sungguhan akan melihat jauh lebih banyak window dan berisiko merealisasikan tail yang lebih buruk dari sampel kecil ini.

**Keputusan final: delta=0.12, lambda_arrival=25000** — menaikkan delta memberi margin lebih aman di atas structural floor ~5.6–8% sambil tetap menjaga gradien CMDP aktif kuat (frac=0.1 saja sudah 13% violation, jauh di atas delta manapun yang dipertimbangkan).

### Validasi sebelum lanjut ke smoke test dan wave

- Cek SIR topologi ulang per-seed individual (bukan pooled) — 92–98% good-SIR konsisten di seluruh 5 seed, bukan keberuntungan agregat.
- Re-run gate `calibrate_load.py` penuh (traffic + deadline + drop, floor dimatikan) dengan `ue_radius_m` fix — lolos semua kriteria.

## Data & artefak

- `scripts/diag_channel3.py`, `diag_topology_sweep.py`, `diag_traffic_sweep.py`, `diag_breakdown.py`, `diag_seed_variance.py`, `diag_window_variance.py` — seluruh script diagnosis disimpan permanen sebagai bukti metodologi (bukan dihapus setelah dipakai).
- `scripts/calibrate_load.py` — gate kalibrasi yang harus lolos sebelum training besar.
- `envs/network_slicing_env.py` — docstring bug-fix v2→v3 dituliskan verbatim di kode.
- `configs/experiment_config.yaml` — tiap nilai kalibrasi (ue_radius_m, lambda_arrival, delta) punya komentar inline yang merujuk script dan angka spesifik yang mendasarinya.
- `tests/test_env.py` — 80/80 test pass, termasuk konservasi paket dan invariant delay ≤ deadline.

---

[← Revisi 2 (v3 CMDP)](04_revisi2-v3-cmdp.md) | [Index](00_INDEX.md) | [Status training v3 →](06_status-training-v3.md)
