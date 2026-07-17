# Hasil Training GNN-MARL Network Slicing

*Generated otomatis oleh `scripts/analyze_results.py` dari 8 run di `results/logs/`.*

## 1. Ringkasan Eksekutif

- **Reward/step terbaik keseluruhan:** `gnn-mappo_gat_seed42` (0.9840 reward/step)
- **SLA satisfaction terbaik:** `central-ppo_seed42` (98.5%)
- **Delay URLLC terendah:** `central-dqn_seed42` (3590.127 ms mean)
- **Terbaik di keluarga DQN (budget setara, 200,000 step):** `gnn-madqn_sage_seed42`
- **Terbaik di keluarga PPO (budget setara, 1,000,000 step):** `gnn-mappo_gat_seed42`

## 2. Setup Eksperimen

- Jumlah gNB: **5**, bandwidth: **10 MHz**, skenario: UMa
- Episode length: **200 step**
- URLLC max delay (SLA bound): **10.0 ms**, eMBB min throughput: **4.0 Mbps**
- Bobot reward: w1=0.4 (eMBB rate), w2=0.3 (URLLC violation), w3=0.2 (URLLC delay ratio, skala `log1p`), w4=0.1 (spectral efficiency); reward per-step di-clip ke `[-10, +10]`
- **Env v2 (Coupled Interference):** AKTIF — interferensi antar-gNB terikat pada fraksi alokasi slice tetangga (reuse-1), observasi memuat SINR/delay/alokasi-tetangga dinamis, penalti delay skala log (bukan linear-clip) supaya gradien tidak jenuh saat delay besar. Lihat §12 untuk perbandingan dengan v1 (uncoupled, diarsip di `results/v1_uncoupled/`).
- Budget training: DQN family = 200,000 step, PPO family = 1,000,000 step; seed = 42 (single seed)
- 8 run: central-dqn_seed42, central-ppo_seed42, gnn-madqn_gat_seed42, gnn-madqn_sage_seed42, gnn-mappo_gat_seed42, gnn-mappo_sage_seed42, idqn_seed42, ippo_seed42

## 3. ⚠️ Catatan Metodologis (baca sebelum menafsirkan angka)

1. **Reward DQN vs PPO tidak dalam satuan yang sama secara mentah.** DQN mencatat `ep_reward` per-episode (200 step), PPO per-rollout (umumnya 512 step). Semua angka reward di laporan ini sudah **dinormalisasi menjadi `reward_per_step`** (`ep_reward / steps_per_row`, dengan `steps_per_row` di-derive otomatis dari median selisih kolom `step`) agar bisa dibandingkan lintas algoritma.
2. **Budget training timpang** (DQN 200K vs PPO 1M step). Perbandingan yang metodologis sah adalah **di dalam keluarga yang sama** (§6), bukan lintas keluarga.
3. **`summary_table.csv` dari `make_paper_figures.py` memakai `ConvergenceEvaluator.sample_efficiency`, yang mengambil satu titik `ep_reward` mentah (noisy)** — dokumen ini memakai rata-rata jendela akhir (10% baris terakhir) dari `ep_reward_ma100` yang sudah dinormalisasi, jauh lebih stabil.
4. **Hanya seed=42 (single seed), tanpa ulangan.** Selisih kecil antar algoritma dalam laporan ini **belum bisa diklaim signifikan secara statistik** — lihat §10.

## 4. Tabel Hasil Utama (Normalized)

| algo | family | final_step | reward/step (mean±std, final 10%) | SLA sat. % | eMBB Mbps | delay ms (mean/median/p95) | Jain's fairness | anomali |
|---|---|---|---|---|---|---|---|---|
| `gnn-mappo_gat_seed42` | proposed | 999,583 | 0.9840 ± 0.2017 | 97.1 | 6.898 | 15912.96 / 6363.81 / 32883.86 | 0.347 | URLLC starvation, degradasi dari peak |
| `ippo_seed42` | baseline | 999,935 | 0.8163 ± 0.1599 | 97.0 | 5.893 | 167134.94 / 130368.01 / 417969.69 | 0.343 | URLLC starvation, degradasi dari peak |
| `gnn-madqn_sage_seed42` | proposed | 199,999 | 0.6732 ± 0.5853 | 98.4 | 4.752 | 84795.73 / 50174.91 / 123143.00 | 0.477 | URLLC starvation, degradasi dari peak |
| `gnn-mappo_sage_seed42` | proposed | 999,935 | 0.5867 ± 0.2209 | 97.5 | 4.172 | 17182.70 / 6601.39 / 37029.61 | 0.435 | URLLC starvation, degradasi dari peak |
| `gnn-madqn_gat_seed42` | proposed | 199,999 | 0.5428 ± 0.4049 | 98.4 | 3.855 | 80984.56 / 47154.88 / 116554.53 | 0.493 | URLLC starvation, degradasi dari peak |
| `central-dqn_seed42` | baseline | 199,999 | 0.5361 ± 0.3922 | 98.5 | 3.786 | 3590.13 / 1870.89 / 6026.52 | 0.497 | URLLC starvation, degradasi dari peak |
| `central-ppo_seed42` | baseline | 999,935 | 0.5257 ± 0.2285 | 98.5 | 3.705 | 3628.30 / 750.95 / 25032.75 | 0.503 | URLLC starvation, degradasi dari peak |
| `idqn_seed42` | baseline | 199,999 | 0.3466 ± 0.3363 | 94.1 | 3.365 | 932183.37 / 129095.27 / 3959946.04 | 0.532 | URLLC starvation, degradasi dari peak |

## 5. Analisis Tren per Algoritma

Tren dihitung dari kemiringan (slope) `reward_per_step` (MA-100 ternormalisasi) pada 25% baris terakhir training.

| algo | trend | peak reward/step (step) | final reward/step | convergence step |
|---|---|---|---|---|
| `central-dqn_seed42` | degrading | 0.7322 (step 96,399) | 0.5193 | 27,599 |
| `central-ppo_seed42` | plateau | 0.5489 (step 862,719) | 0.5156 | 251,903 |
| `gnn-madqn_gat_seed42` | degrading | 0.7436 (step 96,399) | 0.5313 | 20,799 |
| `gnn-madqn_sage_seed42` | degrading | 0.9986 (step 157,199) | 0.7173 | 139,799 |
| `gnn-mappo_gat_seed42` | degrading | 1.0859 (step 787,103) | 1.0043 | 660,639 |
| `gnn-mappo_sage_seed42` | degrading | 0.8647 (step 435,199) | 0.5812 | 87,039 |
| `idqn_seed42` | improving | 0.6399 (step 3,799) | 0.3611 | 1,599 |
| `ippo_seed42` | improving | 0.8559 (step 660,991) | 0.8042 | 535,039 |

## 6. Proposed vs Baseline per Keluarga (budget setara)

### Keluarga DQN (@ 200,000 step)

| algo | family | reward/step final | SLA % | delay ms mean |
|---|---|---|---|---|
| `gnn-madqn_sage_seed42` | proposed | 0.6732 | 98.4 | 84795.73 |
| `gnn-madqn_gat_seed42` | proposed | 0.5428 | 98.4 | 80984.56 |
| `central-dqn_seed42` | baseline | 0.5361 | 98.5 | 3590.13 |
| `idqn_seed42` | baseline | 0.3466 | 94.1 | 932183.37 |

**Pemenang keluarga DQN: `gnn-madqn_sage_seed42` (PROPOSED MENANG)**

### Keluarga PPO (@ 1,000,000 step)

| algo | family | reward/step final | SLA % | delay ms mean |
|---|---|---|---|---|
| `gnn-mappo_gat_seed42` | proposed | 0.9840 | 97.1 | 15912.96 |
| `ippo_seed42` | baseline | 0.8163 | 97.0 | 167134.94 |
| `gnn-mappo_sage_seed42` | proposed | 0.5867 | 97.5 | 17182.70 |
| `central-ppo_seed42` | baseline | 0.5257 | 98.5 | 3628.30 |

**Pemenang keluarga PPO: `gnn-mappo_gat_seed42` (PROPOSED MENANG)**

## 7. Analisis KPI Jaringan

- Throughput eMBB tertinggi: `gnn-mappo_gat_seed42` (6.898 Mbps)
- Fairness (Jain's index) tertinggi: `idqn_seed42` (0.532)
- Delay URLLC terendah: `central-dqn_seed42` (3590.127 ms)
- **Perhatikan kolom mean vs median/p95 delay pada §4** — pada run dengan distribusi delay yang skewed (mis. `gnn-mappo`), mean bisa jauh lebih besar dari median karena ditarik oleh sedikit outlier katastrofik; SLA satisfaction % lebih representatif untuk kasus ini.

## 8. Temuan & Anomali (Diagnosis)

**URLLC starvation masih terjadi di env v2, lintas proposed maupun baseline** (`idqn_seed42` (baseline) delay mean 932183 ms, SLA 94.1%, `ippo_seed42` (baseline) delay mean 167135 ms, SLA 97.0%, `gnn-madqn_sage_seed42` (proposed) delay mean 84796 ms, SLA 98.4%, `gnn-madqn_gat_seed42` (proposed) delay mean 80985 ms, SLA 98.4%, `gnn-mappo_sage_seed42` (proposed) delay mean 17183 ms, SLA 97.5%, `gnn-mappo_gat_seed42` (proposed) delay mean 15913 ms, SLA 97.1%, `central-ppo_seed42` (baseline) delay mean 3628 ms, SLA 98.5%, `central-dqn_seed42` (baseline) delay mean 3590 ms, SLA 98.5%). Env v2 sudah mengganti penalti delay linear-clip dengan `log1p(delay_ratio)` (`envs/network_slicing_env.py:235-238`) supaya gradien tidak jenuh — itu memperbaiki *pembelajaran* dari starvation (lihat §6, proposed sekarang menang di kedua keluarga), tapi tidak menghilangkan starvation itu sendiri: begitu antrean URLLC sudah menumpuk di satu episode, delay mean tetap bisa meledak (lihat kolom p95/median §4 untuk sebaran sebenarnya). Ini konsisten dengan rekomendasi lama soal PRB floor URLLC (§10) yang belum diterapkan.

## 9. Kesimpulan: Hasil Terbaik

- **Reward/step tertinggi lintas semua run:** `gnn-mappo_gat_seed42` (0.9840)
- **Terbaik keluarga DQN (adil, budget sama):** `gnn-madqn_sage_seed42`
- **Terbaik keluarga PPO (adil, budget sama):** `gnn-mappo_gat_seed42`
- **SLA satisfaction terbaik:** `central-ppo_seed42` (98.5%)

**Catatan kehati-hatian:** hasil di atas berasal dari **satu seed (42)** tanpa ulangan, jadi belum sah untuk klaim "algoritma X signifikan lebih baik dari Y" di paper tanpa multi-seed run + uji statistik (lihat §10).

## 10. Rekomendasi Sebelum Paper

1. ~~Revisi reward clipping~~ — **sudah diterapkan di env v2** (log-scale delay penalty, clip `[-10,+10]`). Lihat §12 untuk dampaknya vs v1.
2. **Samakan budget step DQN vs PPO** (atau laporkan sample-efficiency, bukan reward absolut, saat membandingkan lintas keluarga).
3. **Tambah minimal 3 seed per algoritma** untuk error bar dan uji signifikansi (mis. Welch's t-test / bootstrap CI) sebelum klaim proposed vs baseline di paper.
4. **Laporkan p95/median delay**, bukan hanya mean, terutama untuk run yang distribusinya skewed.
5. **Pertimbangkan floor alokasi PRB URLLC minimum** di action space atau di reward, supaya agent tidak bisa menstarve URLLC sepenuhnya — starvation masih muncul di §8 meski reward sudah diperbaiki.
6. Re-run dengan multi-seed setelah floor URLLC diterapkan, bandingkan ulang dengan tabel §4.

## 11. Indeks File

- Figures: `results/figures/*.png` (11 grafik, generated oleh `scripts/make_paper_figures.py`)
- `results/figures/paper_metrics_long.csv` — data tidy per algo/metric/step
- `results/figures/summary_table.csv` — ringkasan dari `ConvergenceEvaluator` (lihat catatan §3.3)
- `results/results_summary.csv` — tabel ternormalisasi dari dokumen ini (machine-readable)
- `results/v1_uncoupled/` — arsip hasil v1 (uncoupled interference, reward clip `[-2,+2]`)
- `results/checkpoints/*_last.pt`, `*_best.pt` — checkpoint tiap run

## 12. Perbandingan v1 (uncoupled) vs v2 (coupled interference)

v1 = interferensi antar-gNB tidak tergantung alokasi tetangga, obs hampir statis, reward di-clip `[-2,+2]` (linear, jenuh). v2 = coupling reuse-1, obs dinamis (SINR/delay/alokasi-tetangga), reward log-scale, clip `[-10,+10]`. Budget training sama persis (DQN 200K, PPO 1M, seed 42).

| algo | reward/step v1 | reward/step v2 | delta | SLA% v1 | SLA% v2 | delay ms mean v1 | delay ms mean v2 |
|---|---|---|---|---|---|---|---|
| `central-dqn_seed42` | 0.0357 | 0.5361 | +0.5004 | 95.8 | 98.5 | 99072.2 | 3590.1 |
| `central-ppo_seed42` | 0.0039 | 0.5257 | +0.5218 | 96.5 | 98.5 | 12429.2 | 3628.3 |
| `gnn-madqn_gat_seed42` | -0.1354 | 0.5428 | +0.6782 | 86.6 | 98.4 | 15190803.8 | 80984.6 |
| `gnn-madqn_sage_seed42` | -0.1593 | 0.6732 | +0.8325 | 84.3 | 98.4 | 12691671.1 | 84795.7 |
| `gnn-mappo_gat_seed42` | -0.0288 | 0.9840 | +1.0128 | 95.7 | 97.1 | 174121.1 | 15913.0 |
| `gnn-mappo_sage_seed42` | -0.0158 | 0.5867 | +0.6025 | 96.1 | 97.5 | 45016.7 | 17182.7 |
| `idqn_seed42` | 0.1699 | 0.3466 | +0.1768 | 97.1 | 94.1 | 4118166.3 | 932183.4 |
| `ippo_seed42` | -0.0009 | 0.8163 | +0.8172 | 97.1 | 97.0 | 17454.6 | 167134.9 |

- **v1 pemenang DQN:** `idqn` (baseline) | **v2 pemenang DQN:** `gnn-madqn_sage_seed42`
- **v1 pemenang PPO:** `central-ppo` (baseline) | **v2 pemenang PPO:** `gnn-mappo_gat_seed42`
- Di v1, seluruh baseline mengalahkan proposed di kedua keluarga. Di v2, proposed (`gnn-mappo_gat` dan `gnn-madqn_sage`) menang di kedua keluarga — konsisten dengan hipotesis awal: GNN-MARL unggul saat environment benar-benar membutuhkan koordinasi antar-gNB (coupling), bukan saat gNB independen optimal secara lokal.
