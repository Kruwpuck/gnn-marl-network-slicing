# HANDOVER — GNN-MARL Network Slicing (implementasi Rev 2)

Update terakhir: **2026-08-04**. Menggantikan `JOURNEY/HANDOFF.md` (tanggal 2026-07-20, pra-hasil-v3, pra-Rev 2 — masih valid untuk sejarah v1→v3, tapi tidak mencakup pekerjaan Fase 0–2).

Dokumen kerja utama: [`docs/rev2-implementation-plan.md`](rev2-implementation-plan.md) — panduan implementasi terverifikasi-kode, Fase 0 sampai 3. Handover ini melaporkan **apa yang sudah dieksekusi dari panduan itu dan apa yang belum**.

---

## 1. Ringkasan project

Riset GNN-MARL untuk alokasi PRB dinamis pada network slicing 5G/6G (eMBB + URLLC, 5 gNB). Formulasi CMDP dengan dual Lagrangian (`lambda`) untuk menegakkan constraint SLA URLLC sambil memaksimalkan throughput eMBB. URLLC dimodelkan packet-level (FIFO per-gNB, deadline-drop + overflow-drop). PRB floor via proyeksi aksi (`none`/`static`/`dynamic`).

Delapan algoritma dibandingkan:

| Kelompok | Algoritma |
|---|---|
| GNN multi-agent (proposed) | `gnn-madqn_gat`, `gnn-madqn_sage`, `gnn-mappo_gat`, `gnn-mappo_sage` |
| Independent multi-agent | `idqn`, `ippo` |
| Centralized | `central-dqn`, `central-ppo` |

Sejarah revisi: v1 uncoupled → v2 coupled interference → v3 CMDP + packet-level URLLC + bug-fix topologi UE-gNB. **v1 dan v2 INVALID sebagai skenario UMa** (mewarisi bug penempatan UE), hanya arsip historis. Hanya v3 yang dilaporkan di paper.

**Posisi sekarang:** wave v3 (80 run) sudah selesai dan ter-commit (`a085157`). Rev 2 adalah redesain untuk uji konfirmatori yang adil — sedang dieksekusi fase demi fase.

---

## 2. Status per fase

| Fase | Isi | Status |
|---|---|---|
| 0 | Metrik risiko dari data yang sudah ada (§3) | ✅ **SELESAI** |
| 1 | Eval-only dari checkpoint yang sudah ada (§4) | ✅ **SELESAI** |
| 2 §5.1 | Perubahan config (buffer, delta) | ✅ **SELESAI** |
| 2 §5.3 | Lokalkan observasi | ✅ **SELESAI** — test hijau |
| 2 §5.2 | Rekalibrasi terhadap policy TERLATIH | ⏸️ **KODE SIAP, BELUM DIJALANKAN** — butuh GPU luang |
| 3 | Wave konfirmatori 80 run (§6) | ⛔ **BELUM MULAI** — di-gate oleh Fase 2 |

Dependensi (§8 panduan): Fase 0 dan 1 tidak menyentuh env, jadi hasil v3 tetap valid. **Fase 2 mengubah env** — sejak titik ini v3 = "regime A", wave 2 = "regime B", dilaporkan berdampingan.

---

## 3. Fase 0 — SELESAI

### Yang dibuat

**`scripts/stability_report.py`** (baru, belum ter-commit) → `results/STABILITY.md`.
Menghitung collapse rate **level-seed** (bukan level-episode) atas `embb_p5_mbps`, threshold 0,01 Mbps, plus Wilson 95% CI, worst-seed mean, dan CVaR@20%. Reuse `parse_run_name`/`load_eval_dir` dari `rliable_report.py`.

Hasil (disalin verbatim dari `results/STABILITY.md`):

| algo | seeds collapsed | rate | 95% Wilson CI | worst seed mean | CVaR@20% |
|---|---|---|---|---|---|
| `central-dqn` | 0/5 | 0.00 | [0.00, 0.43] | 0.254387 | 0.000038 |
| `central-ppo` | 0/5 | 0.00 | [0.00, 0.43] | 1.773975 | 1.147688 |
| `gnn-madqn_gat` | 1/5 | 0.20 | [0.04, 0.62] | 0.000038 | 0.000037 |
| `gnn-madqn_sage` | 0/5 | 0.00 | [0.00, 0.43] | 1.772099 | 1.175351 |
| `gnn-mappo_gat` | 1/5 | 0.20 | [0.04, 0.62] | 0.000038 | 0.000038 |
| `gnn-mappo_sage` | 0/5 | 0.00 | [0.00, 0.43] | 1.772099 | 1.181356 |
| `idqn` | 1/5 | 0.20 | [0.04, 0.62] | 0.000004 | 0.000004 |
| `ippo` | 5/5 | 1.00 | [0.57, 1.00] | 0.000002 | 0.000001 |

### ⚠️ Selisih dengan tabel referensi panduan — dilaporkan apa adanya

Panduan §3.2 menyatakan `gnn-madqn_gat` seharusnya **2/5** seed kolaps. Script menghasilkan **1/5**. Sudah diinvestigasi ke data mentah (`gnn-madqn_gat_seed44_eval.csv`): seed 44 kolaps **intermiten** — 27 dari 30 episode nyaris nol (~0,00004 Mbps), 3 episode normal (~1,9 Mbps), sehingga **mean-nya 0,193** — di atas threshold 0,01 sesuai definisi §3.1 panduan itu sendiri.

**Kesimpulan: tabel referensi panduannya yang keliru (atau memakai definisi berbeda), bukan script-nya.** Script mengimplementasikan definisi §3.1 dengan benar. Jangan "diperbaiki" agar cocok dengan tabel referensi.

Implikasi untuk paper: seed 44 adalah bukti bahwa collapse level-seed pun menyembunyikan struktur — ada rezim intermiten. Layak disebut sebagai batasan metrik.

### Analisis power (§3.4) → jumlah seed Fase 3

Ditambahkan sebagai section di `results/STABILITY.md`. Base rate kolaps terukur = 0,20. Lebar Wilson CI per N:

| N seed | rate~0,20 | rate~0,30 |
|---|---|---|
| 5 (v3) | lebar 0,59 | lebar 0,65 |
| **10 (Fase 3)** | **lebar 0,45** | **lebar 0,50** |
| 20 | lebar 0,34 | lebar 0,37 |
| 30 | lebar 0,28 | lebar 0,31 |

**Dinyatakan jujur:** 10 seed memperbaiki nyata tapi tidak menyempit di bawah 0,35 — itu butuh N≈20–30, di luar anggaran GPU. Harus ditulis eksplisit di prareg.

### Tambahan ke `scripts/rliable_report.py` (§3.3)

Ditambahkan ke fungsi `compare()`, dua metrik:
- **P(improvement)** via `rly.get_interval_estimates` + `rly_metrics.probability_of_improvement`, dengan CI bootstrap.
- **Performance profile** via `rly.create_performance_profile` pada `tau = median(skor baseline)`.

Regenerasi `results/RLIABLE.md`, `RLIABLE_floornone.md`, `RLIABLE_floorstatic.md`.

Dua bug yang ditemukan & diperbaiki saat itu:
1. **Hang.** `reps=50000` default membuat P(improvement) jalan >7 menit tanpa selesai — bottleneck-nya panggilan `mannwhitneyu` per-resample, bukan ukuran sampel. Diperbaiki dengan `reps=min(reps, 2000)` khusus panggilan itu. **Jangan naikkan lagi tanpa mengukur.**
2. **Sign-flip display.** Untuk KPI lower-is-better (`urllc_delay_p99`), skor dinegasikan secara internal sehingga `tau` tercetak negatif dan membingungkan. Diperbaiki dengan `tau_display = tau if higher_better else -tau`.

---

## 4. Fase 1 — SELESAI

Eval-only, tidak menyentuh env. Hasilnya tetap sah dipakai di paper apa pun yang terjadi di fase berikutnya.

### 4.1 Prasyarat — generalisasi `make_variant_config()`

`scripts/run_wave.py` dimodifikasi: `make_variant_config(floor_mode, overrides)` sekarang menerima dict override dotted-path (`{"env.n_gnb": 20}`), menulis nama file bersuffix (`floor_dynamic_area_size1000_n_gnb20.yaml`). Ditambah CLI `--n-gnb` / `--area-size` di `main()`.

Backward-compat sudah diverifikasi: panggilan lama tetap menghasilkan `floor_dynamic.yaml`. **Ini juga prasyarat command Fase 3 §6.5** — jadi sudah siap dipakai.

`scripts/evaluate_checkpoints.py` dimodifikasi: loop per-checkpoint dibungkus `try/except RuntimeError` → cetak `CANNOT_RUN`, tidak crash seluruh batch.

### 4.2 Zero-shot topology transfer → `results/ZEROSHOT.md`

**`scripts/zeroshot_eval.py`** (baru). Checkpoint hasil training di n_gnb=5 dievaluasi di n_gnb=10 dan 20 **tanpa retrain**. Output: `results/eval_zeroshot/ngnb{N}/*_eval.csv` + `results/eval_zeroshot_summary.csv` + `results/ZEROSHOT.md`.

Hasil (verbatim):

| algo | n_gnb | status | mean throughput_mbps | mean embb_p5_mbps | n seeds |
|---|---|---|---|---|---|
| `central-dqn` | 10 | CANNOT_RUN | - | - | 0 |
| `central-dqn` | 20 | CANNOT_RUN | - | - | 0 |
| `central-ppo` | 10 | CANNOT_RUN | - | - | 0 |
| `central-ppo` | 20 | CANNOT_RUN | - | - | 0 |
| `gnn-madqn_gat` | 10 | OK | 53.6283 | 0.772859 | 5 |
| `gnn-madqn_gat` | 20 | OK | 98.2603 | 0.559591 | 5 |
| `gnn-madqn_sage` | 10 | OK | 52.9201 | 1.322236 | 5 |
| `gnn-madqn_sage` | 20 | OK | 96.7307 | 0.804922 | 5 |
| `gnn-mappo_gat` | 10 | OK | 52.5869 | 1.075150 | 5 |
| `gnn-mappo_gat` | 20 | OK | 96.2575 | 0.807309 | 5 |
| `gnn-mappo_sage` | 10 | OK | 53.3520 | 1.304158 | 5 |
| `gnn-mappo_sage` | 20 | OK | 97.5888 | 0.821591 | 5 |
| `idqn` | 10 | OK | 52.9210 | 1.036578 | 5 |
| `idqn` | 20 | OK | 96.6237 | 0.679456 | 5 |
| `ippo` | 10 | OK | 53.5427 | 0.000015 | 5 |
| `ippo` | 20 | OK | 98.1832 | 0.000015 | 5 |

`central-*` gagal dengan error persis `mat1 and mat2 shapes cannot be multiplied (1x80 and 40x128)` di n=10 dan `(1x160 and 40x128)` di n=20 — **ini hasil yang diharapkan, bukan bug**: `obs_dim` di-bake saat training (`n_gnb*8`).

**Temuan penting yang tidak nyaman:** pada `embb_p5_mbps` (metrik cell-edge), backbone **SAGE transfer paling baik**, dan `idqn` (tanpa GNN sama sekali) **mengungguli `gnn-madqn_gat`** di kedua titik. Klaim "GNN unggul karena transfer topologi" tidak didukung data ini apa adanya.

### 4.3 Mekanisme attention vs interferensi → `results/ATTENTION.md`

`gnn/gat_backbone.py` dimodifikasi: `GATBackbone.forward()` menerima `return_attention: bool = False`, meneruskan `return_attention_weights=True` ke kedua layer `GATv2Conv`. Jalur default tidak berubah.

**`scripts/attention_analysis.py`** (baru). Dua hal, keduanya wajib menurut panduan §4.3:
1. **Korelasi** bobot attention (layer 2) vs path-loss dB mentah (skala env-native, bukan nilai ter-skala /100 yang masuk model). Pemasangan edge dilakukan lewat lookup `(src,dst)` — bukan posisi indeks — karena `GATv2Conv` menyisipkan self-loop.
2. **Ablation kausal**: context manager `uniform_attention()` yang me-nol-kan parameter `att` kedua layer sehingga softmax degenerate ke 1/degree, lalu jalankan episode berpasangan (normal vs uniform) pada seed sama.

Hasil:
- Pearson r = **-0.0107**, Spearman r = **0.0040**. Effect size ~0. p-value kecil (1e-11) **bukan bukti efek nyata** — 400.000 edge-observation itu berasal dari hanya 10 checkpoint × 10 episode, autokorelasi dalam-episode menggelembungkan signifikansi semu. Caveat ini sudah ditulis di dalam `ATTENTION.md`.
- **9 dari 10 checkpoint: selisih KPI normal vs uniform = 0,000000 persis.** Attention secara kausal inert.
- Satu pengecualian (`gnn-madqn_gat` seed 44) justru **berlawanan arah**: uniform attention *menyelamatkan* seed yang kolaps (0,000039 → 0,771623 Mbps) — pola attention terpelajarnya **merugikan**, bukan membantu.

**Implikasi:** narasi "GAT belajar memperhatikan interferer kuat" tidak didukung. Ini temuan negatif yang harus dilaporkan, bukan disembunyikan.

---

## 5. Fase 2 — SEBAGIAN SELESAI

### 5.1 Config — SELESAI

`configs/experiment_config.yaml`, dua nilai berubah, masing-masing dengan komentar inline berisi aritmetikanya:

| Parameter | Dari | Ke | Alasan |
|---|---|---|---|
| `buffer.urllc_max_bits` | 40.960 | **131.072** | DBP = `lambda_arrival × packet_size_bits × deadline_s` = 25000 × 256 × 0,010 = **64.000 bit**. Nilai lama = 0,64× DBP → buffer penuh sebelum deadline sempat mengikat, jadi violation didominasi overflow, bukan deadline. Nilai baru = 2× DBP + margin burst |
| `cmdp.delta` | 0,12 | **0,05** | 0,12 dikalibrasi terhadap policy **statis** (`calibrate_load.py`); gate lolos padahal policy terlatih jalan jauh di bawahnya → constraint tidak pernah mengikat saat training. 0,05 = titik awal; **§5.2 yang menetapkan final** |

`traffic.urllc.lambda_arrival` masih **25000.0** — belum disentuh, menunggu hasil §5.2.

### 5.3 Lokalisasi observasi — SELESAI, TEST HIJAU

`envs/network_slicing_env.py`:
- Field baru `self._prev_alloc_lag2` (init `None` di `__init__`, di-nol di `reset()`, di-update di `step()` **tepat sebelum** `self._prev_alloc = urllc_fracs...`).
- `_get_obs()` kolom ke-7 diganti: dulu `neighbor_urllc_frac_mean` (agregat lintas-gNB — membocorkan state global ke agen yang nominalnya independen), sekarang `self._prev_alloc_lag2` (murni lokal).
- **Lebar observasi tetap 8 kolom** → tidak ada shape yang berubah → tidak ada checkpoint/test yang rusak. Ini opsi "paling murah" yang direkomendasikan panduan §5.3.
- Docstring kelas diperbarui + koreksi klaim `area_size` yang usang (paragraf "CORRECTION" merujuk §1.6/§6.2).

Yang **tidak** perlu diubah setelah dicek: `tests/test_env.py` (assertion `(n_gnb, 8)` / `(8,)` di baris 26, 34, 52, 79, 250 tetap valid), `training/train_baselines.py` (angka 8 hardcoded tetap benar), `agents/mlp_agent.py` (di-grep, tidak ada referensi `neighbor`).

**Verifikasi:** `pytest tests/` → **80/80 pass**. `scripts/test_treatment_identity.py` → **PASS di 3 floor-mode** (`none`, `static`, `dynamic`). Kriteria selesai §5.3 untuk bagian ini terpenuhi.

### 5.2 Rekalibrasi policy terlatih — ⏸️ KODE SIAP, BELUM DIJALANKAN

**`scripts/calibrate_trained.py`** (baru, sudah di-upload ke remote via SFTP).

Alur: latih `ippo` (baseline termurah — dipilih karena murah, bukan karena hasilnya) via subprocess ke `training/train_baselines.py`, lalu eval via `scripts/evaluate_checkpoints.py`, baca `sla_violation_pct`, bandingkan ke `cmdp.delta`, cetak PASS kalau violation ∈ [5%, 12%] atau FAIL + saran nilai `lambda_arrival` baru.

Dua keputusan desain yang **jangan diubah**:
- `CALIB_TAG = "_calib"` → run name `ippo_calib_seed42`, **tidak menimpa** checkpoint wave utama `ippo_seed42` yang sudah diarsipkan. Tanpa ini, default tag kosong akan menimpanya.
- **Sengaja bukan loop multi-ronde otomatis.** Tiap ronde = satu training penuh, dan config yang di-tune itu masuk version control — memutuskan `lambda_arrival` berikutnya harus langkah manusia yang sadar.

Ada flag `--skip-train` untuk re-eval checkpoint yang sudah ada tanpa training ulang.

---

## 6. LANGKAH BERIKUTNYA — yang harus dikerjakan orang, bukan agent

### Langkah 1 (blocking): jalankan kalibrasi

Jalankan **langsung di terminal PC remote** (bukan lewat SSH agent — supaya tetap hidup kalau session Claude ditutup), dan **hanya saat GPU luang** (per 2026-08-04 ada training lain jalan di GPU yang sama):

```
cd "C:\Users\Adaptive Network\Documents\Lung Cancer\gnn-marl-network-slicing"
.venv\Scripts\activate
python scripts\calibrate_trained.py --seed 42
```

Biaya: ~1 training `ippo` 1M step (~0,7–1 jam saat v2; mungkin lebih lama dengan buffer/delta baru) + eval 30 episode otomatis.

### Langkah 2: baca output, putuskan

| Output | Tindakan |
|---|---|
| **PASS** (violation ∈ [5%, 12%]) | Bekukan `lambda_arrival` + `delta`, commit config, Fase 2 selesai → lanjut gate Fase 3 |
| **FAIL, violation < 5%** | Constraint tidak mengikat. Naikkan `traffic.urllc.lambda_arrival` sesuai saran script, jalankan lagi |
| **FAIL, violation > 12%** | Overshoot. Turunkan `lambda_arrival` atau naikkan `delta`, jalankan lagi |

Gate panduan §5.2: kalau violation terlatih masih >5pp di bawah δ setelah 5.1+5.2, **naikkan load lagi atau tambah gNB sebelum lanjut Fase 3.** Jangan lanjut dengan constraint yang tidak mengikat — itu mengulang persis kesalahan v3.

### Langkah 3: gate Fase 3 (setelah Fase 2 PASS)

Dua gate, keduanya sebelum wave dilepas:
1. **Gate SIR (§6.2).** Jalankan `scripts/diag_channel3.py` sekali di `n_gnb=20 / area_size=1000`. Pastikan fraksi good-SIR sebanding dengan 92–98% di n_gnb=5. Kalau meleset jauh, `area_size` perlu disetel ulang — sebelum wave.
2. **Gate biaya (§6.3).** Jalankan 1 run pendek (~5.000 step) di n_gnb=20, ukur SPS, ekstrapolasi. Bottleneck-nya loop Python `for i in range(n)` di `env.step()` (baris ~237, 245, 256, 289, 318), skalanya linear terhadap `n_gnb` — **bukan GPU**. Kalau ekstrapolasi >6×, **berhenti dan vektorkan loop dulu** — lebih murah daripada membakar dua minggu ekstra.

### Langkah 4: prareg (§6.4) — WAJIB, gerbang terakhir

Commit `docs/preregistration-wave2.md` **sebelum run pertama**. Isi minimum: hipotesis, metrik utama (`embb_p5_mbps` + collapse rate, dinyatakan a priori), `lambda_arrival`/`delta` beku dari §5.2 beserta protokolnya, aritmetika buffer, jumlah seed (10) + perhitungan power dari §3.4 termasuk pengakuan bahwa CI-nya tetap lebar, grid `n_gnb` {5, 20} + keputusan kerapatan konstan §6.2, kriteria falsifikasi.

### Langkah 5: wave Fase 3

Grid: 4 algo (`gnn-madqn_sage`, `gnn-mappo_gat`, `idqn`, `central-ppo`) × 10 seed (42–51) × 2 topologi (n_gnb 5 dan 20) = **80 run**, estimasi ~1.000–1.400 GPU-jam ≈ **31 hari** wall-clock.

```bash
# n_gnb=5, floor=dynamic, 10 seed
python scripts/run_wave.py --seeds 42,43,44,45,46,47,48,49,50,51 \
    --algos gnn-madqn_sage,gnn-mappo_gat,idqn,central-ppo --max-parallel 6

# n_gnb=20, area_size=1000 (kerapatan konstan)
python scripts/run_wave.py --seeds 42,43,44,45,46,47,48,49,50,51 \
    --algos gnn-madqn_sage,gnn-mappo_gat,idqn,central-ppo \
    --n-gnb 20 --area-size 1000 --max-parallel 6
```

Flag `--n-gnb`/`--area-size` **sudah ada** (dikerjakan di Fase 1 §4.1). Biarkan tag default menurun otomatis — jangan oper `--tag` manual.

4 algoritma tidak diuji di Fase 3 (`gnn-madqn_gat`, `gnn-mappo_sage`, `central-dqn`, `ippo`) — **sebutkan eksplisit sebagai batasan di paper**, jangan diam-diam.

### Langkah 6: pasca-wave

```bash
python scripts/evaluate_checkpoints.py
python scripts/rliable_report.py --out results/RLIABLE_wave2.md
python scripts/stability_report.py --out results/STABILITY_wave2.md
python scripts/analyze_results.py
python scripts/make_paper_figures.py
```

Lalu tulis `docs/journey/08_hasil-wave2.md` mengikuti format file 07: insiden operasional, statistik, penilaian terhadap kriteria prareg **apa adanya**, batasan.

---

## 7. Inventaris file — semua BELUM DI-COMMIT

HEAD saat ini: `a085157` — "Hasil training & evaluasi v3 lengkap (80 run) + uji identitas perlakuan" (2026-08-03).

### File baru (untracked) hasil Rev 2

| File | Fase |
|---|---|
| `docs/rev2-implementation-plan.md` | panduan induk |
| `docs/HANDOVER.md` | dokumen ini |
| `docs/Rev 2.md` | review/kritik sumber yang memicu redesain |
| `scripts/stability_report.py` | 0 |
| `scripts/zeroshot_eval.py` | 1 |
| `scripts/attention_analysis.py` | 1 |
| `scripts/calibrate_trained.py` | 2 |
| `results/STABILITY.md` | 0 |
| `results/ZEROSHOT.md`, `results/eval_zeroshot_summary.csv` | 1 |
| `results/ATTENTION.md`, `results/attention_summary.csv` | 1 |

### File termodifikasi (belum di-commit)

| File | Perubahan |
|---|---|
| `configs/experiment_config.yaml` | §5.1 — buffer + delta |
| `envs/network_slicing_env.py` | §5.3 — obs lokal + koreksi docstring |
| `gnn/gat_backbone.py` | §4.3 — `return_attention` |
| `scripts/run_wave.py` | §4.1 — override config + `--n-gnb`/`--area-size` |
| `scripts/evaluate_checkpoints.py` | §4.1 — `CANNOT_RUN` tidak crash batch |
| `scripts/rliable_report.py` | §3.3 — P(improvement) + performance profile |
| `results/RLIABLE*.md` (3 file) | regenerasi dengan metrik baru |

### Sampah yang jangan ikut di-stage

Banyak `desktop.ini` (artefak OneDrive/Windows) muncul di `git status`, plus folder `JOURNEY/` (mirror lokal dari `docs/journey/` — **berisi kredensial SSH plaintext di `HANDOFF.md`, jangan pernah di-commit**) dan dua notebook duplikat (`scripts/colab_training (1).ipynb`, `scripts/colab_training old.ipynb`).

**Selalu stage file spesifik setelah cek `git status` — jangan `git add -A`.**

---

## 8. Alur kerja remote

Semua pekerjaan yang butuh torch/gymnasium/rliable **harus** jalan di PC lab — mesin lokal (D:) tidak punya venv-nya. `scripts/rliable_report.py` melakukan `import rliable` di top-level, jadi langsung gagal di lokal.

- Host, user, password, dan path proyek remote: lihat `JOURNEY/HANDOFF.md` §"Akses remote" (sengaja tidak diulang di sini karena dokumen ini bisa ikut ter-commit ke repo publik).
- Selalu pakai `.venv\Scripts\python.exe` di remote.
- Driver paramiko di scratchpad session: `ssh_run.py`, `ssh_sftp_put.py`, `ssh_sftp_get.py`, `ssh_cat.py`.
- **Transfer selama sesi Rev 2 memakai SFTP mentah, bukan git.** Konsekuensinya: remote masih di commit `cf3af5a` sementara file-file terbaru sudah ada di sana lewat SFTP. Jangan `git checkout`/`git pull` di remote tanpa cek — bisa menimpa file yang belum ter-commit di mana pun.
- **Training GPU dijalankan sendiri oleh user di terminal remote**, bukan lewat SSH agent — supaya tetap jalan setelah session ditutup. Cek dulu GPU tidak sedang dipakai training lain.

---

## 9. Stack terukur

Python 3.11.9 · PyTorch 2.4.1+cu124 · PyTorch Geometric 2.8.0 · Gymnasium 1.1.0 · NumPy 1.26.4 · Pandas 2.2.3 (**diturunkan** dari 3.0.3, 2026-08-03) · rliable 1.2.0 · arch 7.2.0 (**8.0.0 tidak kompatibel** dengan rliable 1.2.0) · SciPy 1.13.0 · Matplotlib 3.11.0. GPU RTX 3060, 24-core CPU, Windows 11.

---

## 10. Catatan validitas yang harus dibawa ke paper

1. **v1 dan v2 INVALID** sebagai skenario UMa (bug penempatan UE). Arsip historis saja, bukan baseline kuantitatif.
2. **Setelah Fase 2, v3 juga tidak lagi sebanding langsung** dengan wave 2 — env berubah (buffer, obs). v3 = regime A, wave 2 = regime B, laporkan berdampingan, jangan campur.
3. **Attention GAT tidak terbukti bermekanisme** (Fase 1 §4.3). Jangan tulis klaim mekanisme attention tanpa data pendukung baru.
4. **`idqn` mengungguli `gnn-madqn_gat`** pada metrik cell-edge di zero-shot. Data ini sudah ada dan harus dilaporkan.
5. **CI collapse-rate akan tetap lebar** bahkan di 10 seed. Nyatakan sebagai batasan anggaran, bukan disembunyikan.
6. **Tabel referensi §3.2 panduan salah** (2/5 vs 1/5 aktual). Yang benar output script.
7. `cmdp.delta` diturunkan dari 0,12 ke 0,05 karena constraint lama **tidak pernah mengikat** pada policy terlatih — v3 secara efektif adalah MDP tak-terkendala yang menyamar sebagai CMDP. Ini kelemahan v3 yang harus diakui.

---

## 11. Aturan kerja (session-standing)

- **Git:** jangan `git add -A`. Selalu cek `git status`, stage file spesifik. Commit hanya kalau diminta.
- **Push:** user yang push sendiri. Agent tidak pernah push.
- **Identitas commit:** `-c user.name=Habb -c user.email=ihabhasanainakmal0409@gmail.com`.
- **Angka hasil:** selalu salin dari file report yang di-generate, jangan ketik ulang dari ingatan.
- **Jangan paksa hasil cocok dengan tabel referensi.** Kalau beda, investigasi ke data mentah lalu laporkan apa adanya (aturan §2 panduan).
- **Training GPU:** jalankan user sendiri di terminal remote, saat GPU luang.
