# Hasil Training GNN-MARL Network Slicing

*Generated otomatis oleh `scripts/analyze_results.py` dari 8 run di `results/logs/`.*

## 1. Ringkasan Eksekutif

- **Reward/step terbaik keseluruhan:** `idqn_seed42` (0.1699 reward/step)
- **SLA satisfaction terbaik:** `idqn_seed42` (97.1%)
- **Delay URLLC terendah:** `central-ppo_seed42` (12429.214 ms mean)
- **Terbaik di keluarga DQN (budget setara, 200,000 step):** `idqn_seed42`
- **Terbaik di keluarga PPO (budget setara, 1,000,000 step):** `central-ppo_seed42`

**Temuan utama:** setelah reward dinormalisasi per-step, **algoritma proposed (GNN-MADQN/GNN-MAPPO) belum mengungguli baseline** pada eksperimen seed=42 ini. `gnn-madqn` khususnya menunjukkan tanda **starvation slice URLLC** (delay jauh lebih tinggi dan SLA satisfaction lebih rendah dari baseline (mis. `gnn-madqn_sage_seed42` SLA satisfaction turun ke 84.3%)). Lihat §8 untuk diagnosis dan §10 untuk rekomendasi sebelum data ini dipakai di paper.

## 2. Setup Eksperimen

- Jumlah gNB: **5**, bandwidth: **10 MHz**, skenario: UMa
- Episode length: **200 step**
- URLLC max delay (SLA bound): **10.0 ms**, eMBB min throughput: **4.0 Mbps**
- Bobot reward: w1=0.4 (eMBB rate), w2=0.3 (URLLC violation), w3=0.2 (URLLC delay ratio), w4=0.1 (spectral efficiency); reward per-step di-clip ke `[-2, +2]`
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
| `idqn_seed42` | baseline | 199,999 | 0.1699 ± 0.7219 | 97.1 | 3.057 | 4118166.26 / 64889.41 / 13198131.70 | 0.428 | URLLC starvation, degradasi dari peak |
| `central-dqn_seed42` | baseline | 199,999 | 0.0357 ± 0.8864 | 95.8 | 2.757 | 99072.22 / 57845.87 / 219836.58 | 0.489 | URLLC starvation, degradasi dari peak |
| `central-ppo_seed42` | baseline | 999,935 | 0.0039 ± 0.4851 | 96.5 | 2.564 | 12429.21 / 57.69 / 64141.36 | 0.475 | URLLC starvation, degradasi dari peak |
| `ippo_seed42` | baseline | 999,935 | -0.0009 ± 0.4013 | 97.1 | 1.957 | 17454.59 / 4512.60 / 51872.04 | 0.422 | URLLC starvation, degradasi dari peak |
| `gnn-mappo_sage_seed42` | proposed | 999,935 | -0.0158 ± 0.4503 | 96.1 | 2.499 | 45016.75 / 12225.98 / 238290.18 | 0.449 | URLLC starvation, degradasi dari peak |
| `gnn-mappo_gat_seed42` | proposed | 999,935 | -0.0288 ± 0.4777 | 95.7 | 2.619 | 174121.10 / 11200.33 / 1024686.77 | 0.455 | URLLC starvation, degradasi dari peak |
| `gnn-madqn_gat_seed42` | proposed | 199,999 | -0.1354 ± 0.8301 | 86.6 | 2.812 | 15190803.79 / 64933.36 / 96984874.03 | 0.486 | URLLC starvation, SLA rendah, degradasi dari peak |
| `gnn-madqn_sage_seed42` | proposed | 199,999 | -0.1593 ± 0.8888 | 84.3 | 2.836 | 12691671.14 / 98427.53 / 73770884.90 | 0.461 | URLLC starvation, SLA rendah, degradasi dari peak |

## 5. Analisis Tren per Algoritma

Tren dihitung dari kemiringan (slope) `reward_per_step` (MA-100 ternormalisasi) pada 25% baris terakhir training.

| algo | trend | peak reward/step (step) | final reward/step | convergence step |
|---|---|---|---|---|
| `central-dqn_seed42` | improving | 0.2381 (step 87,999) | -0.0611 | 24,199 |
| `central-ppo_seed42` | degrading | 0.1594 (step 679,423) | 0.0125 | 86,015 |
| `gnn-madqn_gat_seed42` | improving | 0.2491 (step 86,799) | -0.0477 | 15,199 |
| `gnn-madqn_sage_seed42` | improving | 0.1398 (step 60,399) | -0.2148 | 20,999 |
| `gnn-mappo_gat_seed42` | degrading | 0.0946 (step 848,383) | -0.0167 | 467,967 |
| `gnn-mappo_sage_seed42` | degrading | 0.1081 (step 856,063) | -0.0074 | 348,671 |
| `idqn_seed42` | improving | 0.2746 (step 51,799) | 0.1247 | 24,599 |
| `ippo_seed42` | degrading | 0.1193 (step 679,423) | 0.0041 | 233,983 |

## 6. Proposed vs Baseline per Keluarga (budget setara)

### Keluarga DQN (@ 200,000 step)

| algo | family | reward/step final | SLA % | delay ms mean |
|---|---|---|---|---|
| `idqn_seed42` | baseline | 0.1699 | 97.1 | 4118166.26 |
| `central-dqn_seed42` | baseline | 0.0357 | 95.8 | 99072.22 |
| `gnn-madqn_gat_seed42` | proposed | -0.1354 | 86.6 | 15190803.79 |
| `gnn-madqn_sage_seed42` | proposed | -0.1593 | 84.3 | 12691671.14 |

**Pemenang keluarga DQN: `idqn_seed42` (BASELINE MENANG)**

### Keluarga PPO (@ 1,000,000 step)

| algo | family | reward/step final | SLA % | delay ms mean |
|---|---|---|---|---|
| `central-ppo_seed42` | baseline | 0.0039 | 96.5 | 12429.21 |
| `ippo_seed42` | baseline | -0.0009 | 97.1 | 17454.59 |
| `gnn-mappo_sage_seed42` | proposed | -0.0158 | 96.1 | 45016.75 |
| `gnn-mappo_gat_seed42` | proposed | -0.0288 | 95.7 | 174121.10 |

**Pemenang keluarga PPO: `central-ppo_seed42` (BASELINE MENANG)**

## 7. Analisis KPI Jaringan

- Throughput eMBB tertinggi: `idqn_seed42` (3.057 Mbps)
- Fairness (Jain's index) tertinggi: `central-dqn_seed42` (0.489)
- Delay URLLC terendah: `central-ppo_seed42` (12429.214 ms)
- **Perhatikan kolom mean vs median/p95 delay pada §4** — pada run dengan distribusi delay yang skewed (mis. `gnn-mappo`), mean bisa jauh lebih besar dari median karena ditarik oleh sedikit outlier katastrofik; SLA satisfaction % lebih representatif untuk kasus ini.

## 8. Temuan & Anomali (Diagnosis)

**URLLC starvation pada `gnn-madqn`.** Pada `envs/network_slicing_env.py:174-178`, `urllc_delay_ms = queue_urllc / max(urllc_rate, 1.0) * 1000`. Saat agent mengalokasikan PRB URLLC mendekati 0%, `urllc_rate` jatuh ke lantai `1.0` bps sehingga delay menjadi `queue_bits * 1000` ms dan meledak seiring antrean menumpuk. Ini terlihat persis pada data: `gnn-madqn_gat_seed42` delay mean 15190804 ms, SLA 86.6%, `gnn-madqn_sage_seed42` delay mean 12691671 ms, SLA 84.3%.

**Reward clip `[-2, +2]` menjenuhkan penalti delay.** Term `-w3 * mean(delay / max_delay)` (`network_slicing_env.py:193`) seharusnya menghukum delay besar, tapi `np.clip(r, -2.0, 2.0)` (`:195`) membuat reward mentok di batas bawah begitu delay sudah katastrofik — memburuk lebih jauh tidak menambah hukuman, sehingga gradien hilang dan agent kehilangan insentif untuk keluar dari kondisi starvation. Ini kandidat **cacat desain reward**, bukan sekadar kekurangan kapasitas model GNN.

**Distribusi delay sangat skewed pada** `central-ppo_seed42`, `gnn-madqn_gat_seed42`, `gnn-madqn_sage_seed42`, `idqn_seed42` — SLA satisfaction tinggi tapi mean delay besar, artinya mayoritas step sehat dan sedikit outlier menarik mean naik jauh. Jangan pakai mean delay sendirian untuk klaim kualitas di paper; sertakan median/p95 (§4).

## 9. Kesimpulan: Hasil Terbaik

- **Reward/step tertinggi lintas semua run:** `idqn_seed42` (0.1699)
- **Terbaik keluarga DQN (adil, budget sama):** `idqn_seed42`
- **Terbaik keluarga PPO (adil, budget sama):** `central-ppo_seed42`
- **SLA satisfaction terbaik:** `idqn_seed42` (97.1%)

**Catatan kehati-hatian:** hasil di atas berasal dari **satu seed (42)** tanpa ulangan, jadi belum sah untuk klaim "algoritma X signifikan lebih baik dari Y" di paper tanpa multi-seed run + uji statistik (lihat §10).

## 10. Rekomendasi Sebelum Paper

1. **Revisi reward clipping** — naikkan batas `[-2,+2]` atau ganti penalti delay ke skala log (`log1p(delay/max_delay)`) supaya gradien tidak hilang saat delay besar.
2. **Samakan budget step DQN vs PPO** (atau laporkan sample-efficiency, bukan reward absolut, saat membandingkan lintas keluarga).
3. **Tambah minimal 3 seed per algoritma** untuk error bar dan uji signifikansi (mis. Welch's t-test / bootstrap CI) sebelum klaim proposed vs baseline di paper.
4. **Laporkan p95/median delay**, bukan hanya mean, terutama untuk `gnn-mappo` yang distribusinya skewed.
5. **Pertimbangkan floor alokasi PRB URLLC minimum** di action space atau di reward, supaya agent tidak bisa menstarve URLLC sepenuhnya.
6. Re-run `gnn-madqn` setelah perbaikan #1 dan #5, bandingkan ulang dengan tabel di §4.

## 11. Indeks File

- Figures: `results/figures/*.png` (11 grafik, generated oleh `scripts/make_paper_figures.py`)
- `results/figures/paper_metrics_long.csv` — data tidy per algo/metric/step
- `results/figures/summary_table.csv` — ringkasan dari `ConvergenceEvaluator` (lihat catatan §3.3)
- `results/results_summary.csv` — tabel ternormalisasi dari dokumen ini (machine-readable)
- `results/logs/_pre_fix_backup/` — backup CSV sebelum perbaikan header/skema
- `results/checkpoints/*_last.pt`, `*_best.pt` — checkpoint tiap run
