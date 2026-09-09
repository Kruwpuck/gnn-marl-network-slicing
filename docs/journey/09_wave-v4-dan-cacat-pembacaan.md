[← Rekalibrasi titik operasi](08_rekalibrasi-titik-operasi.md) | [Index](00_INDEX.md) | [Diagnostik Fase 0 →](10_diagnostik-fase-0.md)

# 09 — Wave v4, Tiga Cacat Pembacaan, dan Dua Gate yang Gagal

**Periode:** 2026-08-09 → 2026-08-23
**Commit:** `44ad631` (data wave v4) → `a78b0a6` (rapikan repo)
**Wave:** 40 job (8 algoritma × 5 seed 42–46), `floor.mode=none`, tag `_v4`, **40/40 OK dalam 147,22 jam** device-time, selesai 2026-08-14 17:38

---

## Ringkasan eksekutif

1. **Wave v4 selesai bersih**, tapi angkanya sempat salah baca tiga kali — dan ketiga kesalahannya menghasilkan **baris yang terlihat valid sempurna**.
2. **Cacat terbesar: seluruh pembacaan "stokastik" keluarga DQN sebenarnya aksi acak seragam.** `epsilon` bukan bagian dari `state_dict`, jadi 20/20 checkpoint DQN dimuat pada nilai awal YAML `epsilon = 1.0`. Terukur, bukan disimpulkan: kesepakatan aksi antara pembacaan greedy dan "stokastik" 0,097–0,100 lawan 1/11 = 0,091 untuk tebakan murni acak.
3. **Kontrafaktual yang paling tajam datang dari cacat pertama:** di bawah argmax, `gnn-mappo_gat` terbaca 16,77 lawan `ippo` 68,06 — "proposed WORSE, CI terpisah". Kesimpulan itu **sepenuhnya artefak pembacaan**.
4. **Dua gate gagal dan ambangnya tidak diamandemen.** B3 gagal (rentang `urllc_delay_p99` 1,0140 ms lawan ambang 2 ms) dan C4 gagal untuk keluarga DQN. Yang dipersempit **cakupan klaimnya**, bukan ambangnya.
5. **Perluasan ke 20 seed merugikan narasi sendiri, dan tetap dijalankan serta dilaporkan** — collapse rate `gnn-mappo_gat` naik ke 14/20 dan `gnn-mappo_sage` ke 19/20.

---

## 1. Hasil wave v4

Verdict rliable di bawah pembacaan primer per keluarga (`results/RLIABLE_v4_primary.md`), 40 perbandingan CI: **23 `COMPARABLE`, 8 `proposed BETTER`, 9 `proposed WORSE`.**

Pola v3 bertahan: mayoritas comparable. Bedanya, kali ini task-nya **tidak** terlalu mudah — titik operasi baru memberi rentang violation 9,9 pp yang bisa disetir (file 08 §3), jadi "comparable" di sini berarti algoritmanya memang setara, bukan instrumen yang buta.

Akuntansi anggaran ditetapkan sebagai properti pengukuran, bukan hasil: ±450 GPU-jam dibaca sebagai **device-jam** (jam okupansi GPU), bukan penjumlahan jam tiap job — menjumlahkan job-jam menghitung ganda satu GPU yang dipakai enam job paralel. Wave v4 = **429,47 job-jam tetapi 147,22 device-jam**.

---

## 2. Tiga cacat pembacaan dalam satu wave

Ketiganya independen, dan ketiganya menghasilkan keluaran yang terlihat sah. Itu polanya, dan pola itulah yang kemudian dinaikkan jadi kontribusi metodologis paper.

### 2.1 Cacat #1 — argmax pada PPO

Sudah dikenali sebelum wave (file 08 §4) dan itulah kenapa P3 dibekukan lebih dulu. Kontrafaktualnya tetap dicatat karena ia yang paling tajam:

> Tanpa perbaikan ini, laporan akan menyimpulkan GNN kalah telak — `gnn-mappo_gat` 16,77 lawan `ippo` 68,06 dengan CI terpisah — kesimpulan yang sepenuhnya artefak.

Satu cacat turunan ikut ketahuan sehari setelah wave: `scripts/rliable_report.py` hanya membaca glob `*_eval.csv` (greedy) dan **tidak pernah** `*_eval_stoch.csv`, padahal P3 sudah menetapkan stokastik primer sejak 2026-08-08. Protokol yang benar tidak berguna kalau skrip pelapornya membaca file yang salah.

### 2.2 Cacat #2 dan #3 — ε = 1,0 pada seluruh keluarga DQN

Ditemukan 2026-08-16 saat memverifikasi grid zero-shot. Akarnya satu:

`epsilon` bukan bagian dari `state_dict` dan `_save_model()` tidak menyimpannya, jadi ke-20 checkpoint DQN v4 tidak membawa epsilon. `load_agent` lalu membangun agen baru yang duduk di nilai awal YAML `epsilon = 1.0`, dan `DQNAgent.act` mengembalikan aksi acak setiap kali `not greedy and rand < epsilon` — yaitu **setiap langkah**.

Dua laporan terkontaminasi, dengan bentuk kerusakan berbeda:

- **Zero-shot** — `central-dqn` yang secara struktural `CANNOT_RUN` di topologi non-5 justru terbaca sebagai baris **OK**, karena jalur aksi acak tidak pernah memanggil jaringannya.
- **Collapse rate** — seluruh `embb_p5` = 0 terbaca sebagai kolaps cell-edge, padahal itu artefak aksi acak.

**Perbaikannya bukan sekadar bug kode melainkan pertanyaan protokol**, dan protokol P3 sudah dibekukan pra-registrasi. Agent mengusulkan "argmax ADALAH policy DQN, epsilon murni eksplorasi" sebagai penetapan; **usulan itu ditolak manusia**, dan yang diminta adalah konvensi plus aturan keputusan yang dideklarasikan **sebelum** pengukuran.

Diagnostiknya lalu dijalankan: 20 checkpoint DQN × 150 episode pada `eps = 0,05` (lantai eksplorasi yang benar-benar dialami policy di akhir training), dibandingkan sd antar-episode terhadap pembacaan greedy pada jumlah episode sama.

| Keluarga | `sd_greedy / sd_(eps=0,05)` | Ambang | Verdict |
|---|---|---|---|
| DQN (`central-dqn`, `idqn`, `gnn-madqn_*`) | **1,01–1,10** | 2,0 | tidak degenerate → **argmax sah sebagai primer** |
| PPO (kasus degenerate yang sudah dikenal) | **3,17 dan 3,62** | 2,0 | degenerate → stokastik primer (P3) |

Jadi readout primer **berbeda per keluarga, dan keduanya diukur, bukan diasumsikan**: sampling untuk PPO, argmax untuk DQN.

### 2.3 Koreksi hasil utama yang mengikutinya

Collapse rate keluarga DQN di bawah pembacaan yang sah bergerak jauh:

| algo | dulu (ε=1,0) | sah (argmax) |
|---|---|---|
| `central-dqn` | 1/5 | **0/5** |
| `gnn-madqn_gat` | 5/5 | **2/5** |
| `gnn-madqn_sage` | 5/5 | **0/5** |
| `idqn` | 5/5 | **2/5** |

Empat baris hasil berubah karena satu cacat pembacaan. Nol angka lama yang tampak mencurigakan saat pertama dilaporkan.

### 2.4 Karantina, bukan penghapusan

Atas keputusan manusia, **40 file readout ε=1,0 dipindah ke `results/quarantine_eps1.0/`** — 20 dari `results/eval` (keluarga DQN di n_gnb=5) dan 20 `central-dqn_*_eval_stoch` dari grid zero-shot — bukan dihapus, karena cacatnya sendiri adalah bahan metodologi paper. README direktori itu membuka dengan larangan eksplisit: nol isinya boleh dipakai di laporan, tabel, atau klaim mana pun.

### 2.5 Instrumen yang mencegah pengulangannya

`scripts/readout_audit.py` (baru) → `results/READOUT_PROVENANCE.md`: tiap laporan yang membawa angka collapse **wajib** menyatakan protokol pembacaannya, dan audit **exit non-zero** kalau ada yang tidak berlabel. Cakupan **15 laporan, 0 tanpa label**.

Alasannya ditulis di file itu sendiri: pada ketiga cacat di atas, **nol angka yang tampak tidak masuk akal** — yang hilang adalah pernyataan protokolnya.

---

## 3. Gate B — B3 gagal, ambangnya tidak digeser

Dihitung ulang lewat `scripts/gate_b_report.py` di bawah pembacaan primer. Sebelumnya Gate B dihitung manual dan satu-satunya jejaknya sebaris di ledger, sehingga tidak bisa diturunkan ulang saat pembacaannya berubah — skrip barunya lebih dulu diverifikasi mereproduksi angka lama sebelum dipakai.

| # | KPI | min | max | rentang | ambang | verdict |
|---|---|---|---|---|---|---|
| B1 | `timely_throughput_mbps` | 59,2646 (`idqn`) | 68,9401 (`gnn-mappo_sage`) | 16,3259 % | ≥ 5% | **LOLOS** |
| B2 | `sla_satisfaction_pct` | 79,1375 (`idqn`) | 91,1192 (`gnn-mappo_sage`) | 11,9818 pp | ≥ 5 pp | **LOLOS** |
| B3 | `urllc_delay_p99` | 7,9827 (`central-ppo`) | 8,9967 (`gnn-mappo_gat`) | **1,0140 ms** | ≥ 2 ms | **GAGAL** |
| B4 | KPI tersaturasi | – | – | 0 dari 5 | ≤ 1 dari 5 | **LOLOS** |

Keputusan manusia dicatat verbatim: **terima apa adanya, ambang tidak diamandemen.** Bedanya dengan amandemen A1 dinyatakan eksplisit — A1 terbukti mustahil secara matematis (SE menyusut √n, offset pengendali tidak), sementara **B3 tidak mustahil, ia hanya gagal terukur.** Mengubahnya sekarang berarti memindahkan tiang gawang setelah melihat hasil.

**Mekanismenya dijelaskan, verdict-nya tidak diubah.** Hipotesis manusia — deadline-drop menyensor distribusi delay sehingga ekornya terpotong — terkonfirmasi dan lebih tajam dari dugaan: deadline 10,0 ms dengan slot 1,0 ms berarti delay paket terkirim hanya bisa mengambil **11 nilai** ({0, 1, …, 10} ms), dan nol nilai di atas 10 ms pernah bisa teramati — paket yang lewat umur di-pop sebelum dilayani sehingga dihitung sebagai drop, bukan sebagai pengiriman lambat. p99 seluruh algoritma karena itu tertarik ke nilai yang mirip **oleh konstruksi instrumennya**.

**Seed freeze ditegakkan lewat kode, bukan ingatan.** Begitu seed 47–61 mendarat, `results/eval` berisi 20 seed untuk empat algoritma PPO dan 5 untuk sisanya; menjalankan `gate_b_report.py` apa adanya akan diam-diam menghasilkan angka n-campuran. **9000 baris eval dari seed di luar himpunan pra-registrasi dikeluarkan**, dan aturannya dikunci di `tests/test_gate_b_seed_freeze.py`.

---

## 4. Gate C — dua baris tidak lolos penuh

| # | Kriteria | Verdict |
|---|---|---|
| C1 | Uji identitas treatment, ≥3 seed × semua floor mode | **LOLOS** — 9/9, `(f_min, lam, delta, violation_rate, reward)` identik bit-per-bit |
| C2 | Nol hyperparameter per-algoritma, semua dari satu YAML | **PARSIAL** |
| C3 | DQN (200K) dan PPO (1M) tidak pernah digabung | **LOLOS**, dengan satu kasus ditandai |
| C4 | Seed ≥ 20 untuk KPI cell-edge bimodal | **LOLOS (PPO) / GAGAL (DQN)** |
| C5 | Evaluasi held-out (seed ≥ 10000) disjoint dari seed training | **LOLOS** |
| C6 | Titik operasi beku dan ter-commit sebelum wave | **LOLOS** |

**C2 PARSIAL, dan alasannya tidak dibulatkan ke atas.** Klausa mengikatnya terpenuhi — agen dikonstruksi tanpa override, dan tiap job menerima daftar argumen yang identik. Tapi hyperparameter optimiser (`lr`, `gamma`, `clip_eps`) saat wave berjalan masih default Python di `agents/*.py`, bukan entri YAML. Diremediasi 2026-08-16 **sesudah** wave: config dapat blok `agent:`, tiap nilainya bit-identik dengan default yang digantikan (`tests/test_hparams_identity.py`), dan satu episode eval held-out mereproduksi baris pra-perubahan bit-per-bit. **Verdict untuk wave sebagaimana ia dijalankan tetap PARSIAL** — biner yang menghasilkan checkpoint itu tidak membaca YAML-nya.

**C3 ditandai jujur:** rentang Gate B (B1–B4) memang didefinisikan lintas kedelapan algoritma, jadi angka gate pra-registrasinya melintasi keluarga. Itu properti task, bukan klaim satu algoritma mengalahkan yang lain, dan pemecahan per-keluarga membuktikan nol verdict bergantung pada penggabungan itu (DQN: B1 8,95%, B2 6,30 pp, B3 0,19 ms; PPO: 7,27%, 5,98 pp, 1,01 ms).

---

## 5. Perluasan seed — aturan biaya lebih dulu, hasilnya belakangan

**Aturan dideklarasikan sebelum satu job pun jalan, atas dasar biaya keluarga bukan hasil:** keluarga yang lebih murah per seed dinaikkan ke 20 seed, keempat algoritmanya sekaligus. Diukur dari `elapsed_sec` terakhir 40 CSV training v4:

| Keluarga | total job-jam | per job | per seed |
|---|---|---|---|
| DQN (200K langkah) | 387,84 | 19,39 | 77,57 |
| PPO (1M langkah) | 41,63 | 2,08 | **8,32** |

PPO **9,3× lebih murah per seed — kebalikan dari dugaan** bahwa 200K langkah pasti lebih murah dari 1M. Aturannya diterapkan apa adanya, jadi yang diperluas keluarga PPO.

60 run tuntas tanpa job mati, 120 file eval, nol anomali. Hasilnya:

| algo | seed kolaps | rate | 95% Wilson CI |
|---|---|---|---|
| `central-ppo` | 3/20 | 0,15 | [0,05 , 0,36] |
| `gnn-mappo_gat` | **14/20** | 0,70 | [0,48 , 0,85] |
| `gnn-mappo_sage` | **19/20** | 0,95 | [0,76 , 0,99] |
| `ippo` | 20/20 | 1,00 | [0,84 , 1,00] |
| `mlp-knn-ppo` | 5/5 | 1,00 | [0,57 , 1,00] |

**Hasilnya merugikan narasi sendiri, persis seperti yang dinyatakan mungkin di muka**, dan tetap dilaporkan penuh. C4 karena itu dilaporkan **per keluarga** — LOLOS untuk PPO di n=20, GAGAL untuk DQN yang masih 5 seed — bukan sebagai satu verdict global, karena itu yang didukung datanya.

Satu efek samping statistik ikut dicetak, bukan didiamkan: pada n=20 dengan α=0,2, CVaR level-seed mengambil 4 seed terburuk sehingga **berdegenerasi jadi worst-seed mean** dan nol informasi tambahan. Dicetak apa adanya karena degenerasi itu sendiri konsekuensi dari C4 yang gagal di 5 seed.

---

## 6. Zero-shot v4, dan status yang diputuskan dari arsitektur

Grid penuh: 9 algoritma × 5 seed × n_gnb {10, 20} × 2 arm topologi × 2 readout, 150 episode, tanpa retrain.

Dua keputusan protokol yang membedakannya dari versi v3:

1. **`CANNOT_RUN` diputuskan dari arsitektur, bukan dari isi disk.** `central-*` punya `obs_dim = n_gnb × 8` yang di-bake saat training, jadi statusnya struktural — dan tiap barisnya membawa alasannya sendiri (`obs_dim mismatch: trained 40, required 80`), supaya **"tidak bisa dijalankan" tidak pernah tertukar dengan "tidak dicoba"**. Ini perbaikan langsung dari cacat #2: membaca filesystem untuk menjawab pertanyaan ini persis yang meloloskan 150 baris "hasil" dari jaringan yang tidak pernah dipanggil.
2. **Kolom yang dibaca throughput per-gNB, bukan agregat** — agregat naik ~4× dari n=5 ke n=20 apa pun yang dilakukan policy.

Kedua arm dilaporkan penuh: `fixed-area` (menaikkan n_gnb sekaligus menaikkan kekuatan kopling — confounded, tapi dipertahankan karena laporan v3 memakainya) dan `const-density` (area diskalakan √(n/5) supaya kerapatannya cocok dengan training).

---

## 7. Anchor sitasi melenceng — nol verdict bergerak

Enam kutipan `path:line` di kolom Evidence `results/GATE_C.md` sudah bergeser 1–5 baris karena file sumbernya disunting setelah tabelnya ditulis (`train_baselines.py:81,154` → `:84,159,163`, dan seterusnya). Dikoreksi 2026-08-23. **Tiap fakta yang didukungnya dibaca ulang pada baris yang benar dan masih berlaku**, jadi C2 tetap PARSIAL dan C5 tetap LOLOS.

Ini kemudian jadi cacat #4 dalam daftar instrumen (file 11): `citation_audit.py --update` tidak boleh dipercaya untuk **memperbaiki** anchor, hanya untuk melaporkan bahwa anchor bergeser.

---

## 8. Status akhir babak ini

`STATE.json` menutup dengan kalimat yang sengaja tidak dibulatkan:

> **BUKAN done.** `goal1.md` menyatakan selesai hanya kalau SEMUA gate lolos; B3 GAGAL, C4 GAGAL untuk keluarga DQN, C2 PARSIAL. Tidak ada ambang yang disentuh; yang dipersempit adalah cakupan klaim.

Cacat pembacaan dinaikkan jadi **hasil utama** paper atas keputusan manusia, dan urutan `handoff/paper_structure.md` dibalik: metodologi jadi bab 1, tesis performa turun ke bab 2. Alasannya ditulis eksplisit — bukan karena temuan performanya lemah, melainkan karena **tiga instansi independen dalam satu wave adalah pola**, bukan kecelakaan.

---

## Data & artefak

| Isi | File |
|---|---|
| IQM + bootstrap CI, pembacaan primer per keluarga | `results/RLIABLE_v4_primary.md` |
| Collapse rate + Wilson CI + CVaR, n=20 untuk PPO | `results/STABILITY_v4_primary.md` |
| Gate B dihitung ulang + pemecahan per-keluarga (C3) | `results/GATE_B_v4_primary.md` |
| Checklist validitas C1–C6 beserta buktinya | `results/GATE_C.md` |
| Mekanisme sensor delay (penjelas B3) | `results/B3_DELAY_CENSORING.md` |
| Perbandingan greedy vs non-greedy per algoritma | `results/READOUT_COMPARISON.md` |
| Provenance pembacaan, 15 laporan 0 tanpa label | `results/READOUT_PROVENANCE.md` |
| Zero-shot 10/20 gNB, dua arm, status struktural | `results/ZEROSHOT_v4_primary.md` |
| Data ε=1,0 yang dikarantina + larangan pemakaian | `results/quarantine_eps1.0/README.md` |
| Kronologi bertanda waktu seluruh babak ini | `runs/2026-08-05-run01/ledger.md` (2026-08-14 → 2026-08-23) |
| Status resmi + alasan "bukan done" | `STATE.json` |

[← Rekalibrasi titik operasi](08_rekalibrasi-titik-operasi.md) | [Index](00_INDEX.md) | [Diagnostik Fase 0 →](10_diagnostik-fase-0.md)
