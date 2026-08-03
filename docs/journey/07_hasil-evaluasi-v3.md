[← Status training v3](06_status-training-v3.md) | [Index](00_INDEX.md)

# 07 — Hasil Training & Evaluasi v3

**Periode training:** 2026-07-20 15:44 → 2026-08-01 20:13 (±12 hari wall-clock, PC lab RTX 3060)
**Pipeline analisis dijalankan:** 2026-08-03
**Total run:** 80 = wave utama (8 algoritma × 5 seed, `floor=dynamic`) + ablation A (4 algoritma × 5 seed, `floor=none`) + ablation B (4 algoritma × 5 seed, `floor=static`)

---

## Ringkasan eksekutif

Dituliskan lebih dulu supaya tidak tenggelam di detail:

1. **Tidak ada keunggulan GNN yang konsisten.** Dari 40 perbandingan CI (4 varian GNN × 2 baseline sepadan × 5 KPI) pada wave utama: **35 `COMPARABLE`, 3 `proposed BETTER`, 2 `proposed WORSE`**. Sesuai kriteria yang ditetapkan sebelum melihat hasil (file 06), mayoritas ini dilaporkan sebagai *comparable* — bukan keunggulan.
2. **Satu-satunya separasi CI yang berulang di dua varian GNN sekaligus**: `gnn-mappo_gat` dan `gnn-mappo_sage` sama-sama mengungguli `ippo` pada cell-edge throughput (`embb_p5_mbps`) — dan itu karena `ippo` gagal total, bukan karena GNN unggul.
3. **Ada satu kekalahan nyata:** `gnn-mappo_gat` **kalah** dari `central-ppo` pada timely throughput dan SLA satisfaction (CI terpisah).
4. **Empat KPI dari lima mengalami saturasi** — semua algoritma, semua seed, semua floor-mode berkumpul di nilai yang nyaris sama. Task-nya ternyata terlalu mudah setelah bug-fix topologi; constraint CMDP tidak pernah mengikat.
5. **Ablation floor tidak menunjukkan efek terdeteksi.** Pada `floor=none` dan `floor=static`, 10 dari 10 perbandingan `COMPARABLE`. Ketiga separasi CI `gnn-mappo_gat` vs `central-ppo` dari wave utama (1 menang, 2 kalah) **tidak bertahan** di dua floor lain — jadi selisih itu sendiri tidak robust, baik yang menguntungkan maupun yang merugikan proposed.

**Kesimpulan jujur untuk paper: pada skenario v3 sebagaimana dikalibrasi, GNN-MARL setara dengan baseline, bukan lebih baik.** Nilai kontribusi terletak pada metodologi (CMDP + packet-level + protokol evaluasi ketat) dan pada temuan negatif yang bisa direproduksi, bukan pada klaim superioritas.

---

## 1. Empat insiden operasional selama wave (dan cara ditanganinya)

Bagian ini ditulis lengkap karena tiga dari empat insiden **mengubah angka atau ruang lingkup analisis**, jadi tidak boleh hilang dari catatan.

### 1.1 Ablation B gagal instan — venv tidak aktif

Percobaan pertama ablation B menghasilkan 20 dari 20 job `FAIL(rc=1)` dalam waktu 0.00 jam. Penyebabnya tunggal dan sepele:

```
ModuleNotFoundError: No module named 'gymnasium'
```

`run_wave.py` dipanggil dengan `python` sistem, bukan interpreter di `.venv`. Diperbaiki dengan mengaktifkan venv sebelum run. Tidak ada dampak ke data — tidak ada satu pun step yang sempat berjalan.

### 1.2 Tag ablation B salah format — 100 file salah nama

`run_wave.py` menurunkan tag default dari floor-mode (`run_wave.py:91`):

```python
tag = args.tag if args.tag is not None else ("" if args.floor_mode == "dynamic" else f"_floor{args.floor_mode}")
```

Default untuk `--floor-mode static` adalah `_floorstatic` (**dengan** garis bawah). Wave dijalankan dengan `--tag floorstatic` (**tanpa** garis bawah), yang meng-override default itu, sehingga nama run menjadi `gnn-madqn_sagefloorstatic_seed42` alih-alih `gnn-madqn_sage_floorstatic_seed42`.

Akibatnya regex parser di `rliable_report.py:53` salah memecah nama:

| Nama file aktual | Ter-parse jadi | Seharusnya |
|---|---|---|
| `gnn-madqn_sagefloorstatic_seed42` | algo=`gnn-madqn`, tag=`_sagefloorstatic` | algo=`gnn-madqn_sage`, tag=`_floorstatic` |
| `central-ppofloorstatic_seed42` | algo=`central-ppofloorstatic`, tag=`` | algo=`central-ppo`, tag=`_floorstatic` |

`analyze_results.py:32-40` juga terdampak: deteksi backbone memakai pola `_sage_`/`_gat_`, yang tidak lagi cocok, sehingga seluruh ablation B akan terklasifikasi salah dan **tidak muncul sama sekali** di laporan statistik.

**Penting: data hasil training-nya sendiri tetap valid** — kesalahan murni di nama file. Perbaikan yang dilakukan: rename 100 file (`results/logs/` 40, `results/eval/` 20, `results/checkpoints/` 40) menyisipkan garis bawah, lalu **evaluasi ulang 20 run ablation B** (`evaluate_checkpoints.py --runs "results/logs/*_floorstatic_*.pt"`) supaya kolom `run_name` di dalam CSV ikut konsisten dengan nama file, bukan hanya nama file-nya saja. Tidak ada training yang diulang.

### 1.3 Seed 42 wave utama: log training kosong, checkpoint valid

Kedelapan CSV `*_seed42.csv` dari wave utama hanya berisi baris header (0 baris data), sementara file `.pt`-nya berukuran normal dan menghasilkan hasil evaluasi yang wajar. Penyebabnya terbaca eksplisit di stdout log:

```
[resume] idqn_seed42: from step 200000 ep 1000
[idqn] done -> results\logs\idqn_seed42.csv
```

Seed 42 di-**resume dari run sebelumnya yang sudah selesai penuh** (200K step untuk keluarga DQN, 1M untuk PPO). Karena budget sudah tercapai, nol step baru dijalankan, dan CSV baru ditulis ulang hanya dengan header.

Konsekuensi yang harus dilaporkan apa adanya:

- **Evaluasi held-out (`RLIABLE.md`) tetap memakai 5 seed penuh dan sah** — evaluasi membaca checkpoint, dan checkpoint seed 42 terlatih penuh.
- **Analisis kurva training (`RESULTS.md`, figures) wave utama hanya punya 4 seed (43–46)** — 72 dari 80 log yang terbaca. Kurva pembelajaran seed 42 hilang permanen.

### 1.4 Dependency rusak — `rliable` tidak bisa jalan

Dua kegagalan berantai saat menjalankan `rliable_report.py`:

| Kombinasi | Error |
|---|---|
| `arch` 7.2.0 + `pandas` 3.0.3 | `TypeError: deprecate_kwarg() missing 1 required positional argument: 'new_arg_name'` (saat import) |
| `arch` 8.0.0 + `rliable` 1.2.0 | `Only NumPy arrays and pandas DataFrames and Series are supported in keyword arguments. Input 'random_state' has type <class 'NoneType'>` |

Penyebabnya: `arch` 8.0.0 mengganti parameter `random_state` menjadi `seed`, sementara `rliable` 1.2.0 masih meneruskan `random_state=` ke `arch_bs.IIDBootstrap`. Di `arch` 8 parameter tak dikenal itu dianggap sebagai *data*, lalu ditolak validasi tipe.

**Perbaikan: turunkan `pandas` ke 2.2.3 dan kembalikan `arch` ke 7.2.0.** Aman karena `pandas` di seluruh repo hanya dipakai oleh `scripts/rliable_report.py` (`grep` seluruh `scripts/ training/ evaluation/ envs/` — satu hit), dan kedua versi tetap memenuhi `requirements.txt` (`pandas>=2.0`, `arch>=6.2`). Setelah itu ketiga laporan rliable berhasil dibuat.

---

## 2. Hasil statistik — wave utama (`floor=dynamic`)

Sumber: `results/RLIABLE.md` — IQM + stratified bootstrap 95% CI atas 5 seed, dari 30 episode evaluasi greedy held-out (seed ≥ 10000) per run. Keluarga DQN (200K step) dan PPO (1M step) tidak pernah digabung.

### Rekapitulasi 40 perbandingan

| Varian proposed | vs baseline | Comparable | Better | Worse |
|---|---|---|---|---|
| `gnn-madqn_gat` | `idqn`, `central-dqn` | 10 | 0 | 0 |
| `gnn-madqn_sage` | `idqn`, `central-dqn` | 10 | 0 | 0 |
| `gnn-mappo_gat` | `ippo` | 4 | 1 | 0 |
| `gnn-mappo_gat` | `central-ppo` | 2 | 1 | 2 |
| `gnn-mappo_sage` | `ippo` | 4 | 1 | 0 |
| `gnn-mappo_sage` | `central-ppo` | 5 | 0 | 0 |
| **Total** | | **35** | **3** | **2** |

**Keluarga DQN: nol separasi CI dari 20 perbandingan.** Baik backbone GAT maupun SAGE tidak terbedakan dari `idqn` maupun `central-dqn` pada KPI manapun.

### Lima separasi CI yang benar-benar terjadi (semuanya di keluarga PPO)

| KPI | Perbandingan | IQM proposed | IQM baseline | Verdict |
|---|---|---|---|---|
| `embb_p5_mbps` | `gnn-mappo_gat` vs `ippo` | 1.7964 [0.5948, 1.8085] | 0.0000 [0.0000, 0.0000] | proposed BETTER |
| `embb_p5_mbps` | `gnn-mappo_sage` vs `ippo` | 1.7803 [1.7727, 1.7985] | 0.0000 [0.0000, 0.0000] | proposed BETTER |
| `jains_fairness` | `gnn-mappo_gat` vs `central-ppo` | 0.5259 [0.5188, 0.8417] | 0.5170 [0.5165, 0.5187] | proposed BETTER |
| `timely_throughput_mbps` | `gnn-mappo_gat` vs `central-ppo` | 30.4868 [30.3903, 30.6816] | 30.8415 [30.7160, 30.8930] | **proposed WORSE** |
| `sla_satisfaction_pct` | `gnn-mappo_gat` vs `central-ppo` | 95.3443 [95.0122, 96.0202] | 96.5499 [96.1371, 96.7125] | **proposed WORSE** |

Dua kemenangan atas `ippo` sepenuhnya berasal dari kegagalan `ippo` (lihat §4), bukan dari keunggulan GNN. Kemenangan fairness `gnn-mappo_gat` bernilai +0.009 IQM dengan batas bawah CI yang nyaris menempel — secara praktis tidak berarti operasional. Sementara dua kekalahannya pada throughput dan SLA justru selisih yang lebih besar dan searah.

---

## 3. Saturasi KPI — temuan utama yang membatasi seluruh analisis

Ini penjelasan struktural di balik 35 `COMPARABLE` di atas: **empat dari lima KPI praktis tidak membedakan apa pun.**

Rentang rata-rata antar 16 kombinasi (algoritma × floor-mode) pada data evaluasi held-out:

| KPI | Min | Max | Std antar-kombinasi | Rentang relatif |
|---|---|---|---|---|
| `timely_throughput_mbps` | 30.52 | 30.82 | 0.084 | **1.0%** |
| `sla_satisfaction_pct` | 95.46 | 96.49 | 0.282 | **1.1 pp** |
| `urllc_delay_p99` (ms) | 3.17 | 3.56 | 0.108 | **0.4 ms** |
| `jains_fairness` | 0.517 | 0.652 | 0.046 | 13.5% |
| `embb_p5_mbps` | 0.000017 | 1.786 | 0.466 | **seluruh rentang** |

Semua algoritma — termasuk baseline paling sederhana — mendarat di ~30.7 Mbps dengan SLA ~96% dan P99 delay ~3.3 ms (deadline 10 ms). Tidak ada ruang bagi arsitektur untuk membuat perbedaan.

### Kenapa saturasi terjadi

Gate kalibrasi (file 05) menargetkan daerah "contested" 5–30% violation, dan **memang tercapai — tapi diukur dengan policy statis, bukan policy terlatih.** Setelah training, seluruh algoritma menemukan kebijakan yang membawa violation rate ke ~3.5–4.5%, jauh di dalam batas `delta=0.12`. Bukti bahwa constraint tidak pernah mengikat:

- **SLA held-out 95.5–96.5%** → violation ~3.5–4.5%, sementara target constraint mengizinkan sampai 12%. Margin ~8 poin persentase tidak terpakai.
- **λ konvergen ke 0.58–0.88 (keluarga PPO) dan 0.93–1.02 (keluarga DQN)**, dari `lambda_init=1.0`, dengan `lambda_max=100.0`. λ tidak pernah mendekati 1% dari plafonnya — mekanisme dual bekerja benar, tapi tidak pernah perlu menekan.

Artinya: bug-fix topologi UE-gNB (file 05) yang menaikkan good-SIR dari 28% ke 92–98% tidak hanya memperbaiki validitas skenario, tapi **juga membuat task-nya cukup longgar sehingga kapasitas jaringan melebihi demand yang dikalibrasi.** Kalibrasi `lambda_arrival=25000` dilakukan terhadap policy statis; policy terlatih ternyata jauh melampaui titik acuan itu.

---

## 4. Cell-edge starvation — satu-satunya sinyal yang benar-benar hidup

`embb_p5_mbps` (persentil-5 throughput eMBB, proksi UE tepi sel) adalah satu-satunya KPI dengan variasi nyata, dan bentuk variasinya **bimodal**: sebuah run entah melayani UE tepi sel di ~1.78 Mbps, atau menelantarkannya sepenuhnya di ~0 Mbps. Tidak ada tengah-tengah.

### `ippo` gagal total dan sistematis

| Statistik `embb_p5_mbps` untuk `ippo` | Nilai |
|---|---|
| Rata-rata | 0.000017 Mbps |
| Maksimum (dari 150 episode) | 0.000041 Mbps |
| Fraksi episode < 0.01 Mbps | **100%** |
| Seed yang terdampak | **5 dari 5** |

Bukan kesialan seed — ini kegagalan arsitektural yang berulang di seluruh seed dan seluruh episode. PPO yang sepenuhnya independen (tanpa sharing antar-agen dan tanpa critic terpusat) konvergen ke kebijakan yang mengorbankan UE tepi sel secara total, sambil tetap mencatat timely throughput agregat 30.64 Mbps — setara semua algoritma lain. **KPI agregat menyembunyikan kegagalan ini sepenuhnya**; hanya KPI cell-edge yang memunculkannya. Ini justifikasi kuat untuk memasukkan `embb_p5_mbps` ke pelaporan, dan layak jadi temuan tersendiri di paper.

### Algoritma lain: kolaps sesekali per-seed

| Algoritma | Seed 42 | 43 | 44 | 45 | 46 |
|---|---|---|---|---|---|
| `gnn-madqn_gat` | 1.790 | **0.000** | **0.193** | 1.774 | 1.774 |
| `gnn-mappo_gat` | **0.000** | 1.808 | 1.784 | 1.796 | 1.808 |
| `gnn-mappo_gat_floorstatic` | 1.796 | 1.772 | 1.784 | 1.807 | **0.000** |
| `idqn` | 1.796 | 1.777 | 1.784 | **0.000** | 1.784 |
| `idqn_floorstatic` | 1.784 | **0.000** | 1.806 | 1.783 | 1.784 |

Umumnya 1 dari 5 seed kolaps (`gnn-madqn_gat`: 2 dari 5). Kombinasi yang **tidak pernah** kolaps di seed manapun: `central-ppo` (ketiga floor-mode), `gnn-mappo_sage`, `gnn-mappo_gat_floornone`, `gnn-madqn_sage_floornone`, `gnn-madqn_sage_floorstatic`.

Kolaps per-seed inilah yang melebarkan CI dan menjelaskan kenapa hampir semua perbandingan berakhir `COMPARABLE` — misalnya CI `embb_p5_mbps` untuk `gnn-madqn_gat` adalah [0.0642, 1.7844], melebar sepanjang hampir seluruh rentang yang mungkin. **Instabilitas antar-seed, bukan perbedaan mean, yang mendominasi ketidakpastian.**

---

## 5. Ablation floor: tidak ada efek yang terdeteksi

Subset 4 algoritma (`gnn-madqn_sage`, `gnn-mappo_gat`, `idqn`, `central-ppo`), masing-masing 5 seed, di bawah `floor=none` dan `floor=static`. Sumber: `results/RLIABLE_floornone.md`, `results/RLIABLE_floorstatic.md`.

| Floor-mode | Perbandingan yang bisa dihitung | Comparable | Better | Worse |
|---|---|---|---|---|
| `none` | 10 | **10** | 0 | 0 |
| `static` | 10 | **10** | 0 | 0 |

(10 = 2 pasangan yang tersedia dalam subset — `gnn-madqn_sage` vs `idqn`, `gnn-mappo_gat` vs `central-ppo` — × 5 KPI. Bagian `gnn-madqn_gat` dan `gnn-mappo_sage` di kedua laporan tertulis "skipped" karena keduanya memang di luar subset ablation, bukan karena data hilang.)

**Implikasi penting terhadap kriteria pra-registrasi.** Kriteria di file 06 berbunyi: *"keunggulan GNN (jika ada) harus bertahan pada floor yang sama."* Ketiga separasi CI yang melibatkan pasangan `gnn-mappo_gat` vs `central-ppo` — satu kemenangan (fairness) dan dua kekalahan (throughput, SLA) — semuanya muncul hanya di `floor=dynamic` dan **hilang menjadi `COMPARABLE` di kedua floor lain**. Ketiganya karenanya tidak robust dan tidak layak dilaporkan sebagai efek nyata: aturan yang sama diterapkan ke kemenangan maupun kekalahan.

Kesimpulan ablation: **pada rezim operasi yang tersaturasi ini, PRB floor tidak mengubah hasil apa pun secara terukur.** Ini konsisten dengan §3 — kalau constraint tidak pernah mengikat, mekanisme proteksi seperti floor memang tidak punya pekerjaan.

---

## 6. Evaluasi terhadap kriteria sukses yang ditetapkan sebelum melihat hasil

Empat kriteria dari file 06, dinilai apa adanya:

| # | Kriteria | Status |
|---|---|---|
| 1 | CI tumpang tindih → laporkan "comparable", bukan "superior" | ✅ **Dipatuhi.** 35 dari 40 comparable dan dilaporkan demikian; tidak ada klaim superioritas yang dinaikkan derajatnya. |
| 2 | Keunggulan GNN harus bertahan pada floor yang sama | ✅ **Diuji, tidak ada yang lolos.** Dari 5 separasi CI di wave utama, 3 bisa diuji lintas-floor (ketiganya pasangan `gnn-mappo_gat` vs `central-ppo`) dan **ketiganya menjadi `COMPARABLE`** di `floor=none` maupun `floor=static`. Dua sisanya melibatkan `ippo`/`gnn-mappo_sage` yang di luar subset ablation, jadi tidak bisa diuji. |
| 3 | Tidak ada angka yang disetel per-algoritma | ✅ **Dipatuhi.** Seluruh hyperparameter bersama berasal dari satu `configs/experiment_config.yaml` + varian floor yang di-generate otomatis oleh `run_wave.py`. |
| 4 | Uji identitas perlakuan (aksi identik → `(floor_applied, lam, delta, violation_rate)` identik) | ✅ **Dijalankan, PASS.** `scripts/test_treatment_identity.py`: dua instance env independen, seed sama, urutan aksi PRB identik → `(f_min, lam, delta, violation_rate, reward)` identik bit-per-bit di 200/200 step, diuji pada 3 floor-mode (seed 42) + 2 seed tambahan (44, 46) pada `floor=dynamic`. Env tidak membedakan siapa yang memanggilnya — mekanisme CMDP/floor terbukti diterapkan identik. |

---

## 7. Interpretasi & rekomendasi

### Yang bisa diklaim dari data ini

- **Metodologi**, bukan performa: formulasi CMDP dengan dual variable terpelajar, model antrean URLLC per-paket dengan deadline drop, PRB floor via action projection identik untuk 8 algoritma, dan protokol evaluasi multi-seed held-out dengan IQM + bootstrap CI.
- **Temuan negatif yang solid:** pada beban jaringan yang tidak menekan, keunggulan struktural GNN tidak terwujud — 8 algoritma yang sangat berbeda kompleksitasnya menghasilkan KPI operator yang praktis identik.
- **Temuan cell-edge:** `ippo` menelantarkan UE tepi sel di 100% episode sambil tampak sehat di seluruh KPI agregat. Ini argumen kuat bahwa evaluasi network slicing tidak boleh berhenti di throughput/SLA rata-rata.
- **Temuan instabilitas:** kolaps cell-edge per-seed (1 dari 5 pada beberapa algoritma) menunjukkan varians antar-seed sebagai sumber ketidakpastian dominan — pembenaran empiris untuk protokol multi-seed, bukan single-seed.

### Yang tidak boleh diklaim

- Bahwa GNN-MARL mengungguli baseline. Data tidak mendukung, dan pada satu kasus justru menunjukkan sebaliknya.
- Bahwa PRB floor efektif. Ablation tidak menunjukkan efek terukur di rezim ini.
- Perbandingan lintas keluarga budget (DQN 200K vs PPO 1M). Tetap tidak sah, sekalipun angka PPO tampak lebih tinggi di `RESULTS.md` §4.

### Rekomendasi teknis kalau wave berikutnya dijalankan

Semua bermuara pada satu hal: **task harus dibuat cukup menekan supaya arsitektur punya kesempatan berbeda.**

1. **Naikkan beban sampai constraint benar-benar mengikat.** Kalibrasi ulang `lambda_arrival` terhadap policy *terlatih*, bukan policy statis — targetkan violation pasca-training yang duduk di sekitar `delta`, bukan 8 poin persentase di bawahnya. Ini akar dari seluruh masalah saturasi.
2. **Turunkan `delta`** agar constraint mengikat pada beban yang sekarang, sebagai alternatif yang jauh lebih murah dari poin 1 (tidak perlu kalibrasi traffic ulang).
3. **Naikkan `buffer.urllc_max_bits`.** `RESULTS.md` §8 mencatat drop didominasi *overflow* (buffer penuh), bukan *deadline*, di seluruh 72 run. Untuk SLA URLLC, deadline-drop lebih representatif — saat ini mekanisme yang dominan bukan yang dimaksudkan dimodelkan.
4. **Jadikan cell-edge KPI kelas satu**, bukan pelengkap. Itu satu-satunya metrik yang membedakan apa pun di wave ini.

---

## 8. Data & artefak

Seluruh angka di dokumen ini disalin dari laporan yang di-generate otomatis, bukan diketik ulang dari ingatan:

| Artefak | Isi |
|---|---|
| `results/RLIABLE.md` | IQM + bootstrap CI 95%, wave utama (`floor=dynamic`) — **sumber sah untuk klaim statistik** |
| `results/RLIABLE_floornone.md` | Idem, ablation A (`floor=none`) |
| `results/RLIABLE_floorstatic.md` | Idem, ablation B (`floor=static`) |
| `results/RESULTS.md` | KPI operator training-time, 72 run — diagnosis konvergensi, **bukan** untuk klaim signifikansi |
| `results/eval/*_eval.csv` | 80 file, 30 episode greedy held-out per run (seed ≥ 10000) |
| `results/logs/*.csv` | Log training per (algoritma, seed, floor-mode); 8 file seed 42 wave utama hanya header (§1.3) |
| `results/figures/*.png` | 11 grafik kurva pembelajaran + KPI |
| `results/figures/paper_metrics_long.csv` | Data tidy per algoritma/metrik/step |
| `results/results_summary.csv` | Tabel training-time versi machine-readable |
| `scripts/test_treatment_identity.py` | Uji identitas perlakuan (§6 kriteria #4) — jalankan `python scripts/test_treatment_identity.py --seed 42 --floor-mode dynamic` |

Perintah persis yang menghasilkan seluruh artefak di atas:

```bash
python scripts/evaluate_checkpoints.py
python scripts/rliable_report.py --out results/RLIABLE.md
python scripts/rliable_report.py --tag _floornone   --out results/RLIABLE_floornone.md
python scripts/rliable_report.py --tag _floorstatic --out results/RLIABLE_floorstatic.md
python scripts/analyze_results.py
python scripts/make_paper_figures.py
```

> `rliable_report.py` **harus** dijalankan tiga kali dengan `--tag` berbeda. Tanpa `--tag`, hanya wave utama yang dilaporkan dan kedua ablation diam-diam terlewat tanpa peringatan.

## 9. Batasan yang harus disebut di paper

1. **Kurva training wave utama hanya 4 seed (43–46)**; seed 42 hilang log-nya karena resume dari run yang sudah selesai (§1.3). Evaluasi held-out tetap 5 seed penuh dan tidak terdampak.
2. **Rezim operasi tersaturasi** — hasil ini berlaku untuk beban yang dikalibrasi, dan tidak bisa digeneralisasi ke jaringan yang benar-benar terbatas kapasitas. Ini batasan terpenting dari seluruh studi.
3. **Drop didominasi overflow, bukan deadline**, di seluruh 72 run — mekanisme pelanggaran SLA yang teramati bukan mekanisme yang secara konseptual dimaksudkan dimodelkan.
4. **Ablation hanya mencakup 4 dari 8 algoritma**, sehingga `gnn-madqn_gat` dan `gnn-mappo_sage` tidak punya pembanding lintas-floor.

---

[← Status training v3](06_status-training-v3.md) | [Index](00_INDEX.md)
