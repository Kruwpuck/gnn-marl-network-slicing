[← Hasil evaluasi v3](07_hasil-evaluasi-v3.md) | [Index](00_INDEX.md) | [Wave v4 →](09_wave-v4-dan-cacat-pembacaan.md)

# 08 — Rekalibrasi Titik Operasi & Gate A

**Periode:** 2026-08-04 → 2026-08-08
**Commit:** `04f4773` (HANDOVER Rev 2) → `528a726` (Gate A lolos seluruhnya)
**Artefak:** `runs/2026-08-05-run01/ledger.md`, `handoff/goal1.md`, `results/STABILITY.md`, `results/ZEROSHOT.md`, `results/ATTENTION.md`, `configs/experiment_config.yaml`

---

## Ringkasan eksekutif

1. **Diagnosis v3 yang salah dikoreksi.** File 07 menutup dengan "task terlalu mudah, constraint tidak pernah mengikat". Penyebab sebenarnya terukur di sini dan lebih sempit: `lambda_lr = 0.01` dikalikan gap ~0,01 memberi 1e-4 per update, jadi dalam 500 update dual cuma bergerak +0,03. **Dual tidak bisa mengikat *by construction*, berapa pun bebannya** — bukan soal beban traffic.
2. **Enam ronde kalibrasi, satu di antaranya lolos secara angka tapi hampa.** Ronde 3 lolos 4/4 sementara `lambda` naik 18× tanpa menggerakkan satu pun perilaku policy. Itu dilaporkan sebagai BLOCKED, bukan sebagai kelulusan.
3. **Rentang violation yang bisa disetir policy di titik operasi v3 cuma 1,6 pp.** Tidak ada arsitektur yang bisa berbeda kalau tidak ada policy yang bisa menyetir constraint-nya — itu, bukan dinamika dual, penjelasan 35/40 `COMPARABLE` di v3.
4. **Protokol pembacaan P3 dibekukan 2026-08-08, sebelum ada satu pun hasil v4.** Pemicunya satu checkpoint yang terbaca 94,06% (lebih buruk dari alokasi nol) di bawah argmax sementara kurva training-nya sehat: **yang kolaps bukan policy melainkan pembacaannya.**
5. **Gate A akhirnya lolos seluruhnya di titik operasi beku**, dan bentuk A1 diubah jadi uji kesetaraan — dengan uji falsifikasi dijalankan lebih dulu untuk membuktikan gate barunya masih bisa menolak.

---

## 1. Rev 2 Fase 0–1 — yang bisa dikerjakan tanpa menyentuh environment

Dua fase pertama Rev 2 sengaja tidak menyentuh `envs/`, jadi hasilnya tetap sah apa pun yang terjadi sesudahnya.

### 1.1 Collapse rate level-seed, dan tabel referensi yang keliru

`scripts/stability_report.py` (baru) menghitung collapse rate **level-seed** atas `embb_p5_mbps` dengan ambang 0,01 Mbps, plus Wilson 95% CI, worst-seed mean, dan CVaR@20%.

Panduan Rev 2 menyatakan `gnn-madqn_gat` seharusnya **2/5** seed kolaps; script menghasilkan **1/5**. Ditelusuri ke data mentah `gnn-madqn_gat_seed44_eval.csv`: seed 44 kolaps **intermiten** — 27 dari 30 episode nyaris nol (~0,00004 Mbps), 3 episode normal (~1,9 Mbps), sehingga mean-nya 0,193 Mbps, di atas ambang.

**Kesimpulannya tabel referensi panduannya yang keliru, bukan script-nya**, dan script tidak "diperbaiki" supaya cocok. Preseden ini dipakai berulang sepanjang jalur berikutnya: yang di-generate mengalahkan yang ditulis tangan.

Implikasi untuk paper: collapse level-seed pun menyembunyikan struktur — ada rezim intermiten, dan itu batasan metriknya.

### 1.2 Analisis power — kenapa 5 seed tidak cukup

| N seed | rate ~0,20 | rate ~0,30 |
|---|---|---|
| 5 (v3) | lebar CI 0,59 | 0,65 |
| 10 | 0,45 | 0,50 |
| 20 | 0,34 | 0,37 |
| 30 | 0,28 | 0,31 |

Dinyatakan jujur sejak awal: 10 seed memperbaiki nyata tapi tidak menyempit di bawah 0,35. Angka ini yang kemudian jadi dasar gate validitas C4 di file 09.

### 1.3 Zero-shot dan atensi pada checkpoint v3 — dua temuan negatif

Zero-shot (`results/ZEROSHOT.md`): `central-*` gagal dengan `mat1 and mat2 shapes cannot be multiplied (1x80 and 40x128)` — **hasil yang diharapkan, bukan bug**, karena `obs_dim` di-bake saat training. Yang tidak nyaman: pada `embb_p5_mbps`, backbone **SAGE transfer paling baik**, dan `idqn` (nol GNN) **mengungguli `gnn-madqn_gat`** di n=10 maupun n=20.

Atensi (`results/ATTENTION.md`): Pearson r = **-0,0107**, Spearman r = **0,0040**. p-value kecil (1e-11) **bukan bukti efek** — 400.000 edge-observation itu dari 10 checkpoint × 10 episode, dan autokorelasi dalam-episode menggelembungkan signifikansi. Ablasi kausal: **9 dari 10 checkpoint memberi selisih KPI normal-vs-uniform 0,000000 persis.** Satu pengecualian berlawanan arah — pada `gnn-madqn_gat` seed 44, atensi seragam justru *menyelamatkan* seed yang kolaps (0,000039 → 0,771623 Mbps).

Narasi "GAT belajar memperhatikan interferer kuat" tidak didukung. Dilaporkan, bukan disembunyikan.

### 1.4 Observasi dilokalkan

Kolom ke-7 observasi diganti dari `neighbor_urllc_frac_mean` — agregat lintas-gNB yang membocorkan state global ke agen yang nominalnya independen — jadi `_prev_alloc_lag2` yang murni lokal. Lebar observasi **tetap 8 kolom**, jadi nol checkpoint dan nol test rusak. `pytest` 80/80, uji identitas treatment PASS di ketiga floor-mode.

---

## 2. Gate A — enam ronde, dan satu kelulusan hampa

Gate A menguji apakah constraint CMDP benar-benar mengikat sebelum wave dilepas. Kronologinya di `runs/2026-08-05-run01/ledger.md`.

| Ronde | Perubahan | Hasil |
|---|---|---|
| 1 | titik operasi 2026-08-04 (`delta=0,05`) | A1 PASS 4,60%, A3 PASS, **A2 FAIL** `lam_ss` 1,033, **A4 FAIL** 2,74 pp |
| 2 | `lambda_lr` 0,01→2,0; `dual_update_every` 2000→5000 | A1/A3/A4 PASS, **A2 FAIL** `lam_ss` 3,192 |
| 3 | `lambda_lr` 2,0→14,0 | **4/4 secara angka** — lalu BLOCKED |
| 4 | titik operasi O1, `delta=0,15` | A1/A2a/A3/A4 PASS, **A2b FAIL** 0,02 pp |
| 5 | `delta` 0,15→0,085 | **A1 FAIL** 12,03%, **A2b FAIL** 0,52 pp |
| 6 | kontrol A2b diperbaiki + A1 didefinisikan ulang | **LOLOS SELURUHNYA** |

### 2.1 Ronde 3: lolos secara angka, hampa secara perilaku

Membandingkan 20% langkah terakhir ronde 1/2/3: `lambda` 1,03 → 3,19 → **18,59** (naik 18×), sementara violation 6,15 → 5,98 → 5,91% (turun 0,24 pp, **di bawah std window 1,61 pp**), eMBB 7,82/7,82/7,85, timely 29,82/29,87/29,87.

Dual naik 18 kali lipat tanpa mengubah satu pun perilaku. Agent berhenti dan menulis `handoff/STUCK.md`.

Koreksi diagnosis datang dari manusia: **`lambda` bukan inert, constraint-nya infeasible.** `delta` berada di bawah lantai yang bisa dicapai ruang aksi, dan pada constraint infeasible `lambda → ∞` adalah perilaku Lagrangian yang benar. Simetri dengan v3 langsung terbaca: v3 punya `delta=0,12` yang feasible tapi longgar sehingga `lambda → 0` hampa; ronde 3 punya `delta=0,05` yang infeasible sehingga `lambda → ∞` hampa. **Dua arah kegagalan yang sama.**

Akibatnya A2 dipecah: **A2a** (feasibility, syarat perlu, diukur tanpa training) + **A2b** (sensitivitas, yang sebenarnya dimaksud). Baseline referensi dipindah `central-ppo` → `ippo`, karena `central-ppo` menyiarkan satu tier PRB ke lima gNB sehingga titik operasi ikut ditentukan ruang aksi tersempit.

### 2.2 Tiga cacat pengukuran yang ditemukan dan dibuang sendiri

Ketiganya ditemukan sebelum angkanya dipakai:

1. **Probe lantai ronde 1 dibuang** — melaporkan lantai per-gNB 4,78% **di atas** lantai uniform 3,51%, mustahil karena ruang aksi per-gNB memuat seluruh vektor seragam. Descent overfit ke 10 episodenya sendiri.
2. **Varians antar-episode jauh lebih besar dari yang diasumsikan** — vektor seragam tier 8 terukur 5,99% pada 10 episode dan 3,42% pada 40 episode. Konsekuensinya seluruh angka kalibrasi wajib pakai jumlah episode tetap dan besar.
3. **Probe kedua mengukur sistem yang salah** — dijalankan `floor_mode=none` padahal wave jalan `dynamic`, dan hasilnya mustahil (lantai tanpa-floor lebih tinggi dari violation policy terlatih). Gate A2a wajib mengukur sistem **sebagaimana dikonfigurasi**.

Satu klaim juga dibatalkan sendiri: "derau antar-populasi-seed 2,0 pp" (dipakai di `STUCK.md`) diuji tuntas dengan 400 episode per populasi — seed 0+ memberi 14,18% ± 0,68, seed 10000+ memberi 14,87% ± 0,70. Selisih 0,69 pp = **0,7 sigma, tidak signifikan.** Angka lamanya derau 150-episode, bukan properti populasi.

### 2.3 Temuan akar: tidak ada policy yang bisa menyetir constraint

Grid tier penuh 0..10, 150 episode, dua mode floor. Di `floor=dynamic`, **seluruh rentang violation yang bisa dicapai policy cuma 3,66–5,26% — lebar 1,6 pp** — dan `delta=0,05` duduk di tengahnya.

Itu penjelasan v3 yang sebenarnya, dan ia menggantikan diagnosis `lambda`: kalau seluruh ruang aksi cuma memindahkan constraint sejauh 1,6 pp sementara derau pengukurannya sebanding, **tidak ada arsitektur yang bisa terbedakan**, sebagus apa pun ia.

---

## 3. Titik operasi baru — dan aturannya ditulis sebelum datanya masuk

Empat opsi diajukan ke manusia; O1 dipilih. Yang penting bukan pilihannya melainkan urutannya: **kriteria pemilihan floor primer (K1–K5) dan aturan penetapan `delta` ditulis ke ledger sebelum pengukuran yang memberi makan keduanya dijalankan**, supaya tidak bisa dituduh memilih hasil. Nol dari kriteria itu menyebut nama algoritma.

| Parameter | v3 | Titik operasi beku (O1) |
|---|---|---|
| `traffic.urllc.lambda_arrival` | 25.000 | **60.000** |
| `buffer.urllc_max_bits` | 40.960 | **307.200** (aturan 2× DBP tidak berubah, hanya bebannya yang bergeser) |
| `cmdp.delta` | 0,12 | **0,085** |
| `cmdp.dual_update_every` | 2.000 | **12.500** |
| `floor.mode` | dynamic | **none** |

Di titik operasi baru rentang violation yang bisa dijangkau jadi 21,8% → 11,9% (**9,9 pp**, ~6× std window), drop tetap 100% deadline-driven (overflow 0,00% di tiap tier), dan eMBB merespons alokasi (7,84 → 4,39 → 0,00 Mbps). Tegangan objective-vs-constraint jadi nyata.

`delta` sempat 0,15 sebelum jadi 0,085. **Aturannya tidak berubah** (lantai + 3,0 pp, dibulatkan ke 0,005); yang salah jangkarnya: 0,15 memakai lantai alokasi **statis** 12,22%, padahal `goal1.md` mewajibkan kalibrasi terhadap policy **terlatih**, yang lantainya 5,49%.

Satu konsekuensi wajib dilaporkan, bukan disembunyikan: pada `delta` sebesar ini, config tetap menuliskan `slices.urllc.reliability: 0.999`, padahal environment ini **tidak bisa** memberi 99,9% pada alokasi mana pun. SLA nominal dan titik operasi yang bisa dijangkau memang berbeda, dan itu properti task.

---

## 4. Protokol pembacaan P3 — cacat pertama dari tujuh

Run kontrol dengan `lambda` dipatok 30 terbaca **94,06%** di bawah evaluasi greedy — lebih buruk dari alokasi nol. Tapi kurva trainingnya sehat sepanjang 1M langkah: violation 10,65 → 5,03 per 100K, eMBB tetap 8,2–8,4, `ep_reward` −881 → −72.

> **Yang kolaps bukan policy, melainkan pembacaannya.**

Kurva respons `lambda` lengkap, `ippo` 1M langkah per titik, 150 episode held-out, greedy lawan stokastik:

| `lambda` | greedy | stokastik | selisih |
|---|---|---|---|
| 0 | 12,57% | 10,57% | −2,00 pp |
| 1 | 12,55% | 10,30% | −2,25 pp |
| 5 | 10,83% | 8,87% | −1,96 pp |
| 12 | 9,65% | 6,69% | −2,96 pp |
| 30 | **94,06%** | **5,49%** | **−88,56 pp** |

Sebabnya terukur, bukan ditebak: argmax cuma membawa `p_max` 0,167–0,331 dari massa aksi. Manusia menyetujui **P3 — stokastik primer untuk PPO, greedy tetap dilaporkan berdampingan tapi tidak pernah menggerbangi.** Dibekukan 2026-08-08, **sebelum satu pun hasil per-algoritma v4 ada.**

Satu hipotesis tandingan diuji dan dibatalkan di tempat: "gradien mati karena clipping reward saat `lambda` besar" tidak didukung, karena pada `lambda=30` kurva trainingnya justru sehat sampai akhir. Kolapsnya murni degenerasi argmax saat evaluasi, bukan hilangnya sinyal belajar.

Satu cacat reproduktibilitas ikut ketahuan: pembacaan stokastik — yang baru saja jadi angka primer — menarik sampel aksi dari RNG global torch yang tidak pernah di-seed, sehingga checkpoint yang sama membaca 8,71% lalu 8,65% pada dua pemanggilan berturut. Diperbaiki dengan seed per-episode sebelum wave dilepas.

---

## 5. A1 diubah bentuknya — dengan uji falsifikasi lebih dulu

Ronde 5 memperlihatkan celahnya: training 20% terakhir memberi violation 8,32% dengan `lambda` 1,87 melawan `delta` 8,50% — **dual setimbang tepat di sasaran, untuk pertama kalinya sejak v3.** Held-out stokastik 8,71% (celah dari training cuma +0,39 pp). Held-out greedy 12,03% (celah +3,32 pp). Jadi seluruh kegagalan A1 adalah artefak argmax.

Sesudah eval di-seed ulang, A1 tetap gagal — dan gagal di **kedua** konvensi, sehingga pilihan `t` lawan 1,96 tidak menentukan apa pun (mean 9,12%, `SE_seed` 0,19 pp, meleset 0,62 pp lawan 0,39 dan 0,54).

Manusia memutuskan mengubah **definisi** A1, bukan menggeser pitanya, dengan alasan yang dicatat verbatim: toleransi kesetaraan diskalakan pada **derau proses** (std window), bukan presisi estimasi mean (SE), karena offset tunak pengendali tidak menyusut dengan jumlah seed — kriteria berbasis SE **mustahil dilewati secara asimtotik** terlepas dari performa sistem, dan pita `|x − δ| ≤ k·SE` malah memberi hadiah pada instrumen yang lebih berderau.

**Gate baru diuji falsifikasi lebih dulu**, karena gate yang meloloskan semua rezim bukan gate. Tiga titik operasi cacat yang sudah diketahui, protokol identik:

| Rezim | Hasil |
|---|---|
| v3 (`delta=0,12`, longgar, `λ→0`) | meleset 6,81 pp lawan Δ 2,70 pp — **DITOLAK** |
| ronde 4 (`delta=0,15`, longgar, `λ→0`) | meleset 4,43 pp lawan Δ 1,40 pp — **DITOLAK** |
| ronde 3 (`delta=0,05`, infeasible, `λ→∞`) | CI lebih lebar dari pitanya — **DITOLAK** lewat klausa presisi |

Catatan jujur yang ikut ditulis: ronde 3 ditolak karena presisi, bukan lokasi — selisih mean-nya 0,86 pp sebenarnya di dalam Δ. Itu bukan lubang, karena rezim infeasible memang wilayah kerja A2a.

**Gate A lolos seluruhnya 2026-08-08T15:00** di titik operasi beku: A1 CI `[8,58 , 9,66]` termuat dalam pita `[6,88 , 10,12]` (n=5, mean 9,12%, `SE_seed` 0,19 pp); A2a lantai 5,49% margin 3,01 pp lawan std 1,62 pp; A2b 2,13 pp lawan 1,62 pp; A3 `inf:1`; A4 1,62 pp.

---

## Data & artefak

| Isi | File |
|---|---|
| Kronologi enam ronde, tiap keputusan bertanda waktu | `runs/2026-08-05-run01/ledger.md` (entri 2026-08-05T22:26 → 2026-08-08T15:00) |
| Definisi A1–A4, amandemen, uji falsifikasi | `handoff/goal1.md` §Gate A |
| Collapse rate v3 + analisis power | `results/STABILITY.md`, dikutip verbatim di `docs/HANDOVER.md` §3 |
| Zero-shot v3 | `results/ZEROSHOT.md`, `docs/HANDOVER.md` §4.2 |
| Atensi v3 + ablasi kausal | `results/ATTENTION.md`, `docs/HANDOVER.md` §4.3 |
| Titik operasi beku + aritmetika tiap nilai | `configs/experiment_config.yaml` (komentar inline blok `cmdp`, `traffic`, `buffer`, `floor`) |
| Bukti pembekuan sebelum wave | `results/GATE_C.md` baris C6 |
| Kepercayaan policy (`p_max`, entropi, agree) | `results/policy_confidence.csv` |

> Satu selisih dokumen dicatat apa adanya: pita A1 ditulis `[6,88 , 10,12]` di ledger 2026-08-08T15:00 dan di tabel Gate G4 `handoff/goal1.md`, tapi `[6,95 , 10,05]` di paragraf uji falsifikasi dokumen yang sama. Verdict-nya sama di kedua versi — CI `[8,58 , 9,66]` termuat di keduanya — jadi tidak ada yang bergerak, tapi selisihnya tidak disembunyikan.

[← Hasil evaluasi v3](07_hasil-evaluasi-v3.md) | [Index](00_INDEX.md) | [Wave v4 →](09_wave-v4-dan-cacat-pembacaan.md)
