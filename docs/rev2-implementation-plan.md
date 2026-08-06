# Panduan Implementasi Rev 2 — Fase 0 sampai 3

Panduan eksekusi untuk [`Rev 2.md`](Rev%202.md). Ditulis supaya bisa diikuti langsung tanpa perlu menurunkan ulang keputusannya.

- **Titik mulai:** commit `a085157` — 80 run v3 selesai, hasil di [`journey/07`](journey/07_hasil-evaluasi-v3.md)
- **Diurut berdasarkan biaya GPU**, bukan nomor Stage di Rev 2 (alasan di §1)
- **Scope Fase 3 sudah dipilih:** hybrid 4 algo × 10 seed × {5, 20} gNB = 80 run

---

## 1. Fakta kode terverifikasi — baca sebelum menyentuh apa pun

Empat hal ini dicek langsung dari kode. Dua di antaranya membantah premis Rev 2 dan mengubah urutan kerja.

### 1.1 `idqn`/`ippo` sudah size-agnostic — premis Rev 2 §RQ3 salah sebagian

Rev 2 mengasumsikan *"MLP/central baselines ... structurally cannot ingest a larger topology"*. Hanya benar untuk `central-*`.

`training/train_baselines.py:80,153`:
```python
obs_dim = n_gnb * 8 if algo == "central-dqn" else 8
```
`train_baselines.py:102` — satu agen, bobot sama, dipanggil per-gNB:
```python
_acts = [agent.act(obs_arr[i]) for i in range(n_gnb)]
```

| Algoritma | `obs_dim` | Jalan di n_gnb ≠ 5? |
|---|---|---|
| `gnn-*` (4 varian) | graph, tak terikat N | ✅ |
| `idqn`, `ippo` | 8 (per-agen, shared) | ✅ |
| `central-dqn`, `central-ppo` | `n_gnb × 8` = 40 | ❌ shape mismatch |

**Konsekuensi:** kontrol adil (ii) yang Rev 2 §RQ3 minta dibangun — "shared-param MLP" — **sudah ada**. Tapi klaim zero-shot jadi lebih berat: GNN harus mengalahkan MLP yang juga bisa transfer, bukan cuma model yang crash.

> ❌ Jangan tulis "MLP tidak bisa transfer".
> ✅ Tulis "*central* tidak bisa jalan; *independent shared-param* bisa, terdegradasi X%".

### 1.2 Observasi membocorkan agregat global ke semua agen

`envs/network_slicing_env.py` `_get_obs()`, kolom ke-7:
```python
total = self._prev_alloc.sum()
neighbor_mean = (total - self._prev_alloc) / (n - 1)
```

Tiap agen — termasuk `idqn`/`ippo` yang sepatutnya independen — menerima rata-rata alokasi seluruh gNB lain. Persis kondisi Rev 2 §RQ2: *"if your agents currently see too much, message passing has nothing to add."* Ditangani di Fase 2c.

### 1.3 Aritmetika buffer — diagnosis Rev 2 §RQ7 terkonfirmasi

| Besaran | Nilai | Sumber |
|---|---|---|
| `lambda_arrival` | 25.000 pkt/s **per gNB** | `configs/experiment_config.yaml` |
| `packet_size_bits` | 256 | idem |
| Offered load | 6,4 Mbps/gNB | 25.000 × 256 |
| Deadline | 10 ms | `slices.urllc.max_delay_ms` |
| **Delay–bandwidth product** | **64.000 bit** | 6,4e6 × 0,010 |
| `buffer.urllc_max_bits` sekarang | **40.960 bit** | config |
| Rasio | **0,64 × DBP** | — |

Buffer lebih kecil dari satu deadline-worth arrivals → penuh sebelum deadline sempat bekerja. Deterministik, menjelaskan overflow-dominance di seluruh 72 run.

### 1.4 Attention & matriks kopling sudah tersedia

- `gnn/gat_backbone.py` pakai `GATv2Conv` → dukung `return_attention_weights=True` bawaan PyG
- `envs/channel_model.py:109-130` `build_interference_graph()` → `edge_attr` = path-loss dB per pasangan gNB = matriks kopling ground-truth
- ⚠️ `gnn/base_backbone.py` `to_tensors()` membagi `edge_attr` dengan 100,0 sebelum masuk model. Untuk korelasi, pakai `edge_attr` mentah dari env (skala linear → Pearson & Spearman identik, tapi jangan sampai salah label satuan)

### 1.5 `n_gnb` sepenuhnya config-driven

`env.py:70` `self.n_gnb = cfg["env"]["n_gnb"]`; obs space `(n_gnb, 8)`, action `MultiDiscrete([11]*n_gnb)`, graph fully-connected `n(n-1)` edge — semua diturunkan. **Tidak perlu ubah kode env untuk menjalankan n_gnb=20.**

### 1.6 ⚠️ Koreksi: `area_size` SEKARANG jadi lever (docstring env sudah usang)

Docstring `envs/network_slicing_env.py:39-41` menyatakan *"independent of area_size ... area_size was never the lever"*. **Itu mengukur topologi PRA-bugfix** (posisi UE independen). Pasca-fix, UE di-cluster dalam `ue_radius_m=100` dari gNB-nya sendiri:

- jarak own-link ≈ tetap ≤100 m, **tidak** ikut `area_size`
- jarak interferer ∝ `area_size / sqrt(n_gnb)`

→ SIR sekarang **naik** seiring `area_size`. Ini penting untuk desain grid Fase 3 (§5.1). Perbaiki docstring-nya saat mengerjakan Fase 2.

---

## 2. Aturan main (jangan dilanggar — ini seluruh argumen legitimasi)

Diturunkan dari Rev 2 §RQ8 dan §"What Would Make This Indefensible".

1. **Justifikasi wajib merujuk properti task, bukan hasil.** "constraint mengikat", "deadline jadi mekanisme pengikat", "jumlah tetangga naik" — boleh. "supaya GNN menang" — tidak boleh, dalam bentuk apa pun.
2. **Setiap perubahan kena 4 algoritma secara identik.** Jangan beri GNN kapasitas/step/observasi/floor berbeda. Dijaga oleh `scripts/test_treatment_identity.py` (sudah ada, PASS).
3. **Prareg sebelum Fase 3.** `docs/preregistration-wave2.md` di-commit **sebelum** run pertama. Tanpa ini, perubahan regime tak bisa dibedakan dari p-hacking.
4. **Seed kolaps tidak boleh dibuang.** Kolaps itu data — lapor apa adanya, termasuk kalau merugikan proposed.
5. **Laporkan dua regime.** Hasil v3 (regime A, longgar) tetap dilaporkan berdampingan dengan wave 2 (regime B, mengikat). Jangan mengganti, jangan menyembunyikan.
6. **Jangan naikkan `pandas`/`arch`.** Pin `pandas==2.2.3` + `arch==7.2.0` — `arch` 8.x mengganti `random_state`→`seed` dan merusak `rliable` 1.2.0 (lihat [`journey/07 §1.4`](journey/07_hasil-evaluasi-v3.md)).
7. **Gunakan tag default `run_wave.py`.** Jangan pernah oper `--tag` tanpa garis bawah di depan — pernah merusak 100 nama file ([`journey/07 §1.2`](journey/07_hasil-evaluasi-v3.md)).

---

## 3. FASE 0 — Metrik risiko dari data yang sudah ada

**Biaya: jam. Nol GPU.** Rev 2 rekomendasi #8 + sebagian #10.
**Input:** `results/eval/*_eval.csv` (80 file × 30 episode) — sudah ter-commit.

### 3.1 Buat `scripts/stability_report.py` → `results/STABILITY.md`

CLI ikuti pola `rliable_report.py`:
```python
p.add_argument("--eval-dir", type=str, default="results/eval")
p.add_argument("--tag", type=str, default="")
p.add_argument("--out", type=str, default="results/STABILITY.md")
p.add_argument("--collapse-threshold", type=float, default=0.01)  # Mbps
```

**Pakai ulang** `parse_run_name()` dari `scripts/rliable_report.py:51-56` — jangan tulis parser nama run kedua. Regex-nya sudah benar untuk format `<algo>_<tag>_seed<N>`.

Tiga metrik, semuanya atas `embb_p5_mbps`:

| Metrik | Definisi | Catatan implementasi |
|---|---|---|
| **Collapse rate** | proporsi seed dengan mean `embb_p5_mbps` < threshold | unit = **seed**, bukan episode |
| **Wilson 95% CI** | CI binomial untuk proporsi di atas | ~5 baris manual, `scipy.stats.norm.ppf`. Jangan tambah dependency |
| **Worst-seed** | `min` over seed dari mean per-seed | — |
| **CVaR@20%** | mean kuintil terburuk dari distribusi seed-episode | `np.mean(np.sort(x)[:max(1, int(0.2*len(x)))])` |

Rumus Wilson (implementasi eksplisit, jangan pakai normal-approx):
```
center = (p_hat + z²/(2n)) / (1 + z²/n)
half   = z/(1 + z²/n) * sqrt(p_hat(1-p_hat)/n + z²/(4n²))
CI     = (center - half, center + half)
```

### 3.2 Nilai referensi — output harus cocok dengan ini

Sudah dihitung dari data yang sama ([`journey/07 §4`](journey/07_hasil-evaluasi-v3.md)). Kalau script menghasilkan angka lain, script-nya salah:

| Algoritma | Seed kolaps | Bukti |
|---|---|---|
| `ippo` | **5/5** | mean 0,000017 Mbps; max dari 150 episode 0,000041 |
| `gnn-madqn_gat` | 2/5 | seed 43 = 0,000; seed 44 = 0,193 |
| `gnn-mappo_gat` | 1/5 | seed 42 |
| `gnn-mappo_gat_floorstatic` | 1/5 | seed 46 |
| `idqn` | 1/5 | seed 45 |
| `idqn_floorstatic` | 1/5 | seed 43 |
| `central-ppo` (3 floor), `gnn-mappo_sage`, `gnn-madqn_sage_floor*` | 0/5 | tak pernah kolaps |

### 3.3 Tambahan ke `scripts/rliable_report.py`

Tambah `P(improvement)` + performance profile. rliable 1.2.0 sudah menyediakannya. Jangan ubah `parse_run_name()` maupun `MATCHED_BASELINES`.

### 3.4 Analisis power → menentukan N seed Fase 3

Dari base rate kolaps yang terukur (~20–40%), hitung N seed untuk lebar CI yang layak (Rev 2 §Caveat 3 minta ini dihitung ulang dari data sendiri, bukan pakai angka 20–30 mentah).

> **Wajib dilaporkan apa adanya:** dengan 5 seed, Wilson CI untuk 2/5 membentang ≈ **[0,12, 0,74]** — terlalu lebar untuk menyimpulkan apa pun. Fase 0 **melaporkan** kelebaran itu, tidak menyembunyikannya. Gunanya menetapkan base rate untuk Fase 3.

**Selesai kalau:** `results/STABILITY.md` ada, `ippo` collapse rate = 1,00, dan angka tabel §3.2 tereproduksi persis.

---

## 4. FASE 1 — Eval-only dari checkpoint yang sudah ada

**Biaya: menit–jam GPU. Nol training.** Rev 2 rekomendasi #7 + #9.

Bisa dilakukan karena `scripts/evaluate_checkpoints.py:52-72` `load_agent()` merekonstruksi agen **tanpa referensi `n_gnb`**: GNN dari `BACKBONES[ckpt["backbone"]]()`, MLP dari `ckpt["obs_dim"]`.

### 4.1 Prasyarat — generalisasi `make_variant_config()`

`scripts/run_wave.py:39-45` sekarang hanya bisa override `floor_mode`. Ubah jadi menerima override sembarang, **backward-compatible**:

```python
def make_variant_config(floor_mode: str, overrides: dict | None = None) -> Path:
    """overrides: {"env.n_gnb": 20, "buffer.urllc_max_bits": 131072} — dotted path."""
```
- Nama file output harus mencerminkan override, mis. `floor_dynamic_ngnb20.yaml`, supaya config tidak saling menimpa
- Pemanggilan lama `make_variant_config("dynamic")` harus tetap jalan tanpa perubahan

### 4.2 Zero-shot topology transfer (Rev 2 #7) → `results/ZEROSHOT.md`

Buat `scripts/zeroshot_eval.py`, atau tambahkan flag ke `evaluate_checkpoints.py` — pilih mana pun yang diff-nya lebih kecil.

1. Emit config `n_gnb ∈ {10, 20}` lewat §4.1
2. Jalankan checkpoint **5-gNB yang sudah ada** terhadap config tersebut
3. **`central-*` akan gagal shape-mismatch → tangkap `RuntimeError`, catat `CANNOT_RUN`, lanjut.** Jangan biarkan crash menghentikan batch. Itu datanya, bukan bug

Matriks hasil yang diharapkan:

| Algoritma | n_gnb=5 | 10 | 20 |
|---|---|---|---|
| `gnn-madqn_{gat,sage}`, `gnn-mappo_{gat,sage}` | baseline | angka | angka |
| `idqn`, `ippo` | baseline | angka | angka |
| `central-dqn`, `central-ppo` | baseline | `CANNOT_RUN` | `CANNOT_RUN` |

> Ingat §1.1: `idqn`/`ippo` **akan** menghasilkan angka. Itu kontrol adilnya. Klaimnya adalah selisih degradasi GNN vs shared-param MLP.

### 4.3 Mekanisme: attention vs interferensi (Rev 2 #9) → `results/ATTENTION.md`

Buat `scripts/attention_analysis.py`. Hanya berlaku untuk `gnn-madqn_gat` dan `gnn-mappo_gat` (SAGE tak punya attention).

**Langkah 1 — ekspos attention.** Tambah parameter opsional di `GATBackbone.forward()`:
```python
def forward(self, x, edge_index, edge_attr, return_attention: bool = False):
```
Default `False` → jalur training **tidak berubah sama sekali**. `GATv2Conv` mengembalikan `(out, (edge_index, alpha))` saat `return_attention_weights=True`.

**Langkah 2 — korelasi.** Bobot attention per-edge vs `edge_attr` (path-loss dB, ambil **mentah dari env**, bukan yang sudah dibagi 100 — lihat §1.4). Lapor Pearson + Spearman, diagregat lintas episode dan lintas seed.

**Langkah 3 — ablation kausal (WAJIB).** Rev 2 §RQ6 + §Caveat 2: seragamkan attention (uniform 1/N) saat eval, ukur degradasi KPI. **Tanpa langkah 3, analisis korelasi hanya dekorasi dan akan ditolak reviewer.** Lapor keduanya berdampingan atau jangan lapor sama sekali.

**Selesai kalau:** `ZEROSHOT.md` punya angka untuk 6 algoritma + `CANNOT_RUN` untuk 2, dan `ATTENTION.md` memuat korelasi **beserta** hasil ablation.

---

## 5. FASE 2 — Perbaikan task + rekalibrasi (Rev 2 Stage 1)

**Biaya: 1–3 hari GPU** (satu baseline pilot, bukan wave).

### 5.1 Perubahan config — deterministik, bukan tuning

`configs/experiment_config.yaml`:

| Parameter | Dari | Ke | Justifikasi (properti task) |
|---|---|---|---|
| `buffer.urllc_max_bits` | 40.960 | **131.072** | 2 × DBP (§1.3) + burst margin → deadline yang mengikat, bukan overflow |
| `cmdp.delta` | 0,12 | **0,05** | nilai awal; §5.2 yang menetapkan final |

Tulis komentar inline yang merujuk aritmetika §1.3 — ikuti gaya config yang sudah ada (tiap nilai kalibrasi punya komentar yang menyebut script + angka dasarnya).

### 5.2 Rekalibrasi terhadap policy TERLATIH — aksi bernilai tertinggi (Rev 2 #1)

`scripts/calibrate_load.py` yang ada mengukur **policy statis**. Itu akar masalahnya: gate lolos, tapi policy terlatih jauh melampaui titik acuannya.

Buat `scripts/calibrate_trained.py` — protokol Rev 2 §RQ1:

1. Latih **satu** baseline termurah sampai konvergen — **`ippo`** (0,74 jam/seed di v2). Dipilih karena **paling murah**, bukan karena hasilnya
2. Ukur violation held-out pasca-training
3. Naikkan `lambda_arrival` (atau turunkan `delta`) sampai violation steady-state **duduk dekat δ** — target **5–12%**, bukan 8pp di bawahnya
4. **Bekukan** `lambda_arrival` dan `delta`, commit, sebelum wave

**Gate (Rev 2 §Stage 1):** kalau setelah 5.1+5.2 violation terlatih masih >5pp di bawah δ → naikkan load lagi atau tambah gNB **sebelum** lanjut ke Fase 3.

### 5.3 Lokalkan observasi (Rev 2 #6)

Hapus/ganti kolom `neighbor_mean` (§1.2) supaya observasi murni lokal. Terapkan identik ke 4 algoritma.

Yang ikut terdampak — perbarui semua:
- `envs/network_slicing_env.py` — `_get_obs()`, docstring baris 20, observation space baris 128
- `tests/test_env.py` — assertion `obs.shape == (env.n_gnb, 8)` di baris 26, 34, 52; `(8,)` di baris 79. Kalau lebar kolom berubah, semua ikut
- `training/train_baselines.py:80,153` — `obs_dim = n_gnb * 8 if ... else 8`, angka 8 hardcoded
- `agents/mlp_agent.py` docstring baris 12-13

**Paling murah:** pertahankan lebar 8 kolom, ganti isi `neighbor_mean` dengan fitur lokal (mis. `prev_alloc` milik sendiri pada lag berbeda, atau backlog terlambat). Tidak ada shape yang berubah → tidak ada checkpoint/test yang rusak.

Sekalian perbaiki docstring `area_size` yang usang (§1.6).

**Selesai kalau:** `pytest tests/` hijau, `scripts/test_treatment_identity.py` PASS di 3 floor-mode, violation held-out policy terlatih ∈ [5%, 12%].

---

## 6. FASE 3 — Wave konfirmatori

**Grid terpilih: hybrid.** 80 run, **~4–5 minggu** wall-clock.

### 6.1 Grid

```
algo   : gnn-madqn_sage, gnn-mappo_gat, idqn, central-ppo   (4)
seed   : 42..51                                              (10)
n_gnb  : 5, 20                                               (2)
       = 80 run
```

Subset algoritma sama dengan ablation v3 → sebanding langsung dengan regime A.

**Yang didapat grid ini:**
- ✅ Collapse rate dengan CI binomial layak (10 seed, bukan 5)
- ✅ Apakah GNN separate saat N naik — hipotesis inti Rev 2 §RQ2
- ✅ **`central-ppo` dilatih native di 5 DAN 20** → itu *retrain upper bound* Rev 2 §RQ3 kontrol (i), gratis dari grid ini. Klaim zero-shot-nya datang dari Fase 4.2 (checkpoint 5-gNB dievaluasi di 20). Dua-duanya tercakup
- ❌ 4 algoritma tak diuji: `gnn-madqn_gat`, `gnn-mappo_sage`, `central-dqn`, `ippo` — **sebutkan eksplisit sebagai batasan**, jangan diam-diam

### 6.2 ⚠️ Keputusan desain: `area_size` saat n_gnb=20

Karena §1.6, `area_size` sekarang mengubah SIR. Menaikkan `n_gnb` 5→20 sambil menahan `area_size=500` akan melipatempatkan kerapatan **dan** memperkuat kopling sekaligus — dua variabel bercampur, padahal cuma ada 2 titik data. Tidak bisa diatribusikan.

**Tetapkan: kerapatan konstan.**

| n_gnb | `area_size` | Alasan |
|---|---|---|
| 5 | 500 | baseline |
| 20 | **1000** | 500 × √(20/5) → kerapatan areal gNB tetap |

Ini mengisolasi variabel yang Rev 2 §RQ2 sebut ("number of agents / neighbors") dari kekuatan kopling. **Masukkan ke prareg beserta alasannya.**

**Gate sebelum wave:** jalankan cek SIR satu kali di n_gnb=20/area_size=1000 (pakai ulang `scripts/diag_channel3.py`) dan pastikan fraksi good-SIR sebanding dengan 92–98% pada n_gnb=5. Kalau meleset jauh, regime-nya bukan konstan dan `area_size` perlu disetel ulang — **sebelum** wave, bukan sesudah.

### 6.3 Gate biaya — ukur dulu, jangan bakar sebulan berdasarkan tebakan

`env.step()` punya beberapa loop Python `for i in range(n)` (baris 237, 245, 256, 289, 318). Itu bottleneck-nya, dan skalanya linear terhadap `n_gnb` — bukan GPU.

**Sebelum melepas wave penuh:** jalankan 1 run pendek (~5.000 step) di n_gnb=20, ukur SPS, ekstrapolasi.

Anggaran yang diasumsikan (dari wall-clock v2 per-seed: `gnn-madqn_sage` 7,4 j; `idqn` 8,0 j; `gnn-mappo_gat` 3,9 j; `central-ppo` 0,73 j ≈ **20 j per set-seed di 5 gNB**):

| Bagian | GPU-jam |
|---|---|
| 10 seed @ 5 gNB | ~200 |
| 10 seed @ 20 gNB (4–6×) | ~800–1.200 |
| **Total** | **~1.000–1.400** |

Referensi throughput: wave v3 ≈ 460 GPU-jam → 12 hari wall-clock (≈38 GPU-jam/hari efektif, `--max-parallel=6`). → **~31 hari.**

Kalau ekstrapolasi menunjukkan >6×, hentikan dan vektorkan loop `step()` dulu — lebih murah daripada membakar tambahan dua minggu.

### 6.4 Prareg — WAJIB sebelum run pertama

Commit `docs/preregistration-wave2.md` **sebelum** wave jalan. Isi minimum:

- Hipotesis (dinyatakan sebelum melihat hasil)
- Metrik utama — **`embb_p5_mbps` + collapse rate**, dinyatakan a priori sebagai proksi cell-edge fairness yang memang dilindungi mekanisme koordinasi. Rev 2 §"Indefensible" secara khusus melarang mempromosikan metrik ini *setelah* melihat siapa yang menang
- `lambda_arrival` dan `delta` beku dari §5.2, beserta protokol penetapannya
- Aturan buffer `≳ arrival_rate × deadline` beserta aritmetikanya
- Jumlah seed (10) beserta perhitungan power dari §3.4
- Grid `n_gnb` {5, 20} + keputusan kerapatan konstan §6.2
- Kriteria berhenti / apa yang dianggap memfalsifikasi hipotesis

### 6.5 Command

```bash
# n_gnb=5, floor=dynamic, 10 seed
python scripts/run_wave.py --seeds 42,43,44,45,46,47,48,49,50,51 \
    --algos gnn-madqn_sage,gnn-mappo_gat,idqn,central-ppo --max-parallel 6

# n_gnb=20 (butuh flag baru dari §4.1 diteruskan ke run_wave)
python scripts/run_wave.py --seeds 42,43,44,45,46,47,48,49,50,51 \
    --algos gnn-madqn_sage,gnn-mappo_gat,idqn,central-ppo \
    --n-gnb 20 --area-size 1000 --max-parallel 6
```

Perlu tambahan `--n-gnb`/`--area-size` di `run_wave.py`, diteruskan lewat `make_variant_config()` dari §4.1. **Biarkan tag default menurun otomatis** — jangan oper `--tag` manual (§2 aturan 7).

### 6.6 Setelah wave

```bash
python scripts/evaluate_checkpoints.py
python scripts/rliable_report.py --out results/RLIABLE_wave2.md
python scripts/stability_report.py --out results/STABILITY_wave2.md
python scripts/analyze_results.py
python scripts/make_paper_figures.py
```

Lalu tulis `docs/journey/08_hasil-wave2.md` mengikuti format file 07: insiden operasional, statistik, penilaian terhadap kriteria prareg apa adanya, batasan.

---

## 7. Matriks verifikasi

| Fase | Output | Kriteria selesai |
|---|---|---|
| 0 | `results/STABILITY.md` | Angka tabel §3.2 tereproduksi; `ippo` collapse = 1,00; Wilson CI 2/5 ≈ [0,12, 0,74] dilaporkan apa adanya |
| 1a | `results/ZEROSHOT.md` | 6 algoritma menghasilkan angka di n_gnb=20; `central-*` tercatat `CANNOT_RUN`, bukan crash |
| 1b | `results/ATTENTION.md` | Korelasi attention↔path-loss dilaporkan **bersama** ablation uniform-attention |
| 2a | config ter-commit | `pytest tests/` hijau; `urllc_max_bits` ≥ 2×DBP |
| 2b | load/δ beku + log | violation held-out policy terlatih ∈ [5%, 12%] |
| 2c | env ter-update | test identitas perlakuan PASS di 3 floor-mode; obs shape tetap `(n_gnb, 8)` |
| 3-gate | cek SIR + SPS | good-SIR @20gNB sebanding 5gNB; ekstrapolasi ≤6× |
| 3-prareg | `docs/preregistration-wave2.md` | **ter-commit sebelum run pertama** |
| 3 | 80 run + `journey/08` | seluruh 80 run selesai atau kegagalannya tercatat |

---

## 8. Urutan & dependensi

```
Fase 0  ─┐
         ├─→ (independen, boleh paralel — keduanya tak menyentuh env)
Fase 1  ─┘
              │
              ↓  §3.4 base rate kolaps → jumlah seed Fase 3
         Fase 2  (ubah env + config → hasil v3 tak lagi sebanding)
              │
              ↓
         Gate §6.3 (ukur SPS)  →  Prareg §6.4  →  Fase 3 wave
```

- **Fase 0 dan 1 tidak menyentuh env** → hasil v3 yang sudah ter-commit tetap valid. Kerjakan lebih dulu, hasilnya bisa dipakai di paper apa pun yang terjadi kemudian
- **Fase 2 mengubah env** → mulai titik ini v3 = "regime A", wave 2 = "regime B". Keduanya dilaporkan berdampingan (§2 aturan 5)
- **Prareg adalah gerbang terakhir sebelum Fase 3.** Jangan lewati

---

## 9. Di luar scope

- **Baseline zero-padding/truncation** (Rev 2 §RQ3 kontrol iii) — dilabeli "weak baseline, for completeness" oleh Rev 2 sendiri, sementara kontrol kuat (ii) sudah gratis (§1.1). Tambahkan hanya kalau reviewer meminta
- **n_gnb=10** — dilewati demi 10 seed. Punya 2 titik dengan CI sempit lebih berguna daripada 3 titik yang semuanya inconclusive
- **α-fair utility sweep** — ditunda sejak [`journey/06`](journey/06_status-training-v3.md), tidak dihidupkan Rev 2
