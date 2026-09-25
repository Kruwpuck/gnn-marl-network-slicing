[← v5 Diblokir dan Wave v6](11_v5-diblokir-dan-wave-v6.md) | [Index](00_INDEX.md)

# 12 — Atensi v6: Korelasi Ada, Kausalitas Tidak

**Periode:** 2026-09-01 → 2026-09-24
**Sumber angka:** `results/ATTENTION_V6.md`, `results/ATTENTION_v4_greedy.md`, `results/DIAG_GNN_RELIANCE_V6.md`, `results/DIAG_INPUT_SEP_V6.md`
**Status:** kewajiban pasca-wave PREREG-V6 §5 tuntas — D2b, D6, dan analisis atensi sudah dijalankan di checkpoint v6

> File ini menutup pertanyaan mekanisme yang dibuka file 11: apakah kolom edge kedua membuat atensi benar-benar dipakai. Jawabannya tidak, dan jalan menuju jawaban itu memuat satu koreksi pembanding yang sempat membalik arah bacaan.

---

## Ringkasan eksekutif

1. **Ablasi atensi inert di v6 — dan ternyata sudah inert juga di v4.** Bukan regresi yang dibawa arm baru; keadaannya memang begitu sejak awal.
2. **Keselarasan atensi–interferensi justru melemah** setelah kolom kopling interferensi ditambahkan, diukur di tingkat yang laporannya sendiri sebut sebagai tempat klaim mekanisme hidup.
3. **Arm dua-kolom tidak lebih terdampak dari satu-kolom**, dan dua arm dua-kolom saling bertentangan arah.
4. **D2b sepakat**, dengan besaran dua orde lebih kecil lagi.
5. **D6 menunjuk tempatnya**: representasi antar-node sudah nyaris rata di `conv2`, sebelum kepala policy membacanya.

---

## 1. Kenapa korelasi saja tidak cukup

Dua keadaan berbeda menghasilkan rho negatif yang sama persis:

- **(a)** atensi belajar menimbang interferer kuat, dan policy memanfaatkan penimbangan itu.
- **(b)** atensi berkorelasi dengan path loss semata karena `edge_attr` memang path loss, tapi keluarannya tidak mempengaruhi keputusan.

Yang memisahkan keduanya hanya ablasi kausal: paksa atensi seragam (nol-kan parameter `att` kedua layer `GATv2Conv`, softmax runtuh ke 1/degree), lalu ukur apakah KPI bergerak. Tanpa itu, korelasi cuma hiasan — dan laporannya sendiri menyebut ablasi ini `(mandatory)`.

KPI-nya tiga, bukan `embb_p5_mbps` saja. Di titik operasi ini sebagian besar checkpoint sudah duduk di lantai cell-edge, dan KPI yang terpaku dekat nol tidak bisa mendegradasi sebesar apa pun ablasinya. `timely_throughput_mbps` dan `sla_satisfaction_pct` masih punya ruang, jadi di situlah efek nyata harus muncul.

---

## 2. Koreksi pembanding — yang sempat membalik arah bacaan

Pembacaan pertama memakai `results/ATTENTION.md` sebagai pembanding v4 dan menyimpulkan korelasi **berkembang dari nol jadi terarah**. Itu salah.

`ATTENTION.md` adalah laporan era **pra-v4**, dijalankan pada titik operasi lama sebelum rekalibrasi (`delta` 0,12, buffer 40.960). Membandingkannya dengan v6 membandingkan dua task berbeda.

Pembanding yang setara adalah `ATTENTION_v4_greedy.md`: readout `greedy` yang sama, checkpoint `gnn-*_gat_v4_seed*.pt`, titik operasi beku yang sama. Dengan pembanding itu arahnya terbalik:

| | v4 `gat` | v6 (3 arm) |
|---|---|---|
| Pearson (pooled) | −0,0991 | −0,1548 |
| Spearman (pooled) | −0,2462 | −0,2044 |
| **per-node mean rho** | **−0,5146** | **−0,2089** |
| **per-node median rho** | **−0,8000** | **−0,4000** |
| **fraksi rho < 0** | **0,765** | **0,611** |
| n edge | 400.000 | 3.000.000 |
| node-step | 100.000 | 750.000 |

Keselarasan atensi–interferensi di v4 sudah kuat, dan v6 **lebih lemah**. Pooled Pearson memang naik sedikit, tapi laporan itu sendiri menyatakan pooled bukan tingkat yang benar: ia mencampur edge dari node dengan skala atensi dan himpunan tetangga berbeda, sementara softmax dinormalisasi per node penerima. Di tingkat per-node, ketiga ukurannya turun.

> Menambahkan kolom kopling interferensi tidak membuat atensi lebih selaras dengan interferensi fisik. Ia membuatnya kurang selaras.

### Ini instansi kedelapan pola cacat instrumen — dan yang pertama bukan di kode

Kesalahan ini dicatat sebagai **cacat #8** di `PLAN-06` §5, bukan diperlakukan sebagai kekeliruan sekali lewat. Nomornya delapan, bukan tujuh: daftar itu sudah memuat tujuh instansi sebelum babak ini, tiga di antaranya lahir di file 11.

Yang membuatnya berbeda dari tujuh pendahulunya: **tidak ada kode yang salah.** `ATTENTION.md` dan `ATTENTION_v4_greedy.md` dua-duanya laporan sah yang di-generate dengan benar, dan nol angka di dalamnya keliru. Yang salah adalah pemilihan mana yang jadi pembanding — dan nama file keduanya tidak menyatakan titik operasi mana yang mereka ukur.

Akibatnya persis bentuk yang sama dengan tujuh yang lain: **keluaran yang terlihat valid padahal kesimpulannya terbalik.** Temuan yang sempat terbit berbunyi "rho berkembang dari nol jadi terarah, jadi kolom edge membuat atensi lebih selaras" — kebalikan dari yang dinyatakan data.

Aturan yang lahir darinya, dicatat di `PLAN-06` §5: sebuah laporan hanya boleh dibandingkan dengan laporan yang **menyatakan readout dan titik operasi yang sama di header-nya**, dan nama kedua file dikutip saat perbandingan ditulis. Itu sebabnya file ini menyebut `ATTENTION_v4_greedy.md` secara eksplisit, bukan cuma bilang "v4".

---

## 3. Ablasi kausal — inert di v6, dan inert juga di v4

Checkpoint mati dibuang lebih dulu: 15 dari 75 di v6 (dan 2 dari 10 di v4) punya `timely_throughput_mbps` di bawah 5 Mbps, jauh dari ~60–70 milik checkpoint sehat. Mereka tidak bisa terdegradasi, jadi memasukkannya akan menggelembungkan hitungan inert.

| | v4 `gat` | v6 (3 arm) |
|---|---|---|
| checkpoint hidup | 8 dari 10 | 60 dari 75 |
| berubah **persis nol** | 1/8 | 29/60 |
| mean Δ `timely_throughput_mbps` | −0,999 | +0,194 |
| sd | 5,124 | 3,105 |
| **t = mean/(sd/√n)** | **−0,55** | **+0,48** |
| median Δ (ketiga KPI) | +0,000000 | +0,000000 |

Tidak ada satu pun yang lolos derau. Nilai |t| terbesar di seluruh tabel v6, per-arm maupun gabungan, adalah **1,94** (`gnn-mappo_gatedge`); sisanya di bawah 1. Dari 31 checkpoint v6 yang bergerak sama sekali: **18 rugi, 13 untung** — nyaris lempar koin.

Temuan yang paling penting di sini bukan angka v6-nya, melainkan kolom v4-nya: **ablasi sudah inert sebelum arm baru ada.** Jadi ini bukan sesuatu yang dirusak PLAN-03, dan memperbaiki arm tidak akan menyentuhnya.

### Arm dua-kolom tidak lebih terdampak

| kelompok | n hidup | inert | mean | sd | t | rugi | untung |
|---|---|---|---|---|---|---|---|
| dua-kolom (`gatedge`, `gatres-edge`) | 40 | 18/40 | +0,358 | 2,721 | 0,83 | 13 | 9 |
| satu-kolom (`gatres`) | 20 | 11/20 | −0,133 | 3,736 | −0,16 | 5 | 4 |

Arah selisihnya kebetulan sesuai dugaan, besarannya tenggelam di sd. Dan dua arm dua-kolom **saling bertentangan**: `gnn-mappo_gatedge` mean +0,83 dengan 7 rugi lawan 1 untung, sementara `gnn-mappo_gatres-edge` mean −0,16 dengan 1 rugi lawan 4 untung. Kalau kolom kedua yang bekerja, keduanya harus sepakat.

### Koreksi istilah yang harus dibawa

Pengelompokan ini **bukan** "punya fitur edge lawan tidak punya". Keempat arm GATv2 membaca `edge_attr` (`gnn/__init__.py`):

| arm | `edge_dim` | isi |
|---|---|---|
| `gat` | 1 | path loss |
| `gatres` | 1 | path loss, plus residual |
| `gatedge` | 2 | path loss + kopling interferensi |
| `gatres-edge` | 2 | keduanya, plus residual |

Kontrasnya satu kolom lawan dua kolom. Merumuskannya sebagai ada-lawan-tiada membuat uji "penurunan hanya di arm ber-fitur-edge" tidak punya kelompok kontrol — dan uji itu tidak pernah bisa dijalankan sebagaimana diucapkan.

---

## 4. D2b sepakat, dan besarannya dua orde lebih kecil

D2b mengacak `edge_attr` di dalam tiap node penerima. Kalau atensi tidak dipakai, D2b harus ikut null — dan memang.

| kelompok | t (`timely_throughput_mbps`) | mean besaran efek |
|---|---|---|
| dua-kolom | 0,52 | 0,005–0,02 Mbps |
| satu-kolom | 1,39 | 0,005–0,02 Mbps |

Bandingkan dengan ablasi atensi yang besaran efeknya 1,0–1,4 Mbps. Mengacak informasi edge mengubah keputusan **lebih sedikit lagi** daripada meratakan atensi. Dua diagnostik menunjuk kesimpulan sama; tidak ada pertentangan yang perlu dijelaskan.

Satu batas topologi ikut dicatat laporannya sendiri: `build_interference_graph` memancarkan setiap pasangan gNB terurut, jadi grafnya lengkap dan tiap node penerima punya himpunan tetangga identik. Mempermutasi label sumber di dalam satu grup tujuan karena itu no-op. Yang dirusak D2b adalah **pemasangan edge-ke-atribut**, jadi pada graf lengkap ia menguji sensitivitas terhadap *informasi* edge, bukan terhadap *topologi*.

---

## 5. D6 menunjuk tempat kebocorannya

`DIAG_INPUT_SEP_V6.md` mengukur seberapa jauh node saling terbedakan di tiap tahap forward pass. `rel_spread` bernilai 0 berarti node-nya identik.

| backbone | input | `conv1_pre` | `conv1_act` | `conv2` |
|---|---|---|---|---|
| `gatedge` | 0,1590 | 0,0092 | 0,0120 | **0,0003** |
| `gatres` | 0,3705 | 0,0014 | 0,0338 | **0,0059** |
| `gatres-edge` | 0,3305 | 0,0021 | 0,0310 | **0,0070** |

Observasi masuk sudah membedakan node (0,16–0,37), lalu runtuh dua sampai tiga orde di keluaran `conv2`. Apa pun yang ditimbang atensi di hulu, yang sampai ke kepala policy sudah nyaris rata.

Itu menyatukan ketiga diagnostik jadi satu kalimat:

> Informasi edge sampai ke bobot atensi — korelasi per-node membuktikannya. Ia tidak sampai ke keputusan, karena representasi antar-node sudah dihomogenkan sebelum kepala policy membacanya. Titik perbaikannya ada di antara keluaran `conv2` dan kepala policy, bukan di fitur edge-nya.

Residual memang menahan sebagian keruntuhan itu — `gatres` dan `gatres-edge` menutup di 0,0059 dan 0,0070 lawan 0,0003 milik `gatedge`, dua puluh kali lebih tinggi. Tapi menahan separasi ternyata tidak cukup untuk membuat atensi terpakai.

---

## 6. Batasan yang harus ikut dibaca

- **Checkpoint mati.** 15 dari 75 checkpoint v6 duduk jauh di bawah rezim sehat dan tidak mungkin mendegradasi. Mereka sudah dikeluarkan dari seluruh angka di file ini, tapi keberadaannya sendiri menunjukkan sebagian besar populasi tidak cocok untuk uji mekanisme apa pun.
- **Keluarga DQN cuma 5 seed**, jadi Gate C4 tetap gagal di sana dan tidak ada klaim karakterisasi yang dibuat untuknya.
- **Readout `greedy`**, yang laporannya tandai sebagai `reported, never gates`. Ia dipakai di sini karena pembanding v4-nya juga greedy; ia bukan pembacaan primer yang menggerbangi klaim komparatif.
- **Nol uji hipotesis formal.** Nilai t di file ini rasio deskriptif, bukan uji dengan tingkat kesalahan yang dikendalikan. Semuanya jauh di bawah ambang konvensional mana pun, jadi tidak ada keputusan yang bergantung pada pilihan uji.

---

## 7. Apa yang boleh dan tidak boleh ditulis dari ini

**Boleh:** atensi GATv2 pada arsitektur dan titik operasi ini tidak dipakai policy secara kausal, terbukti lewat ablasi di v4 maupun v6. Menambah kolom kopling interferensi tidak mengubahnya, dan justru melemahkan keselarasan atensi–interferensi per-node.

**Tidak boleh:** menyebut rho negatif sebagai bukti mekanisme. Arah rho itu informasi tentang apa yang dipelajari bobot atensi, bukan tentang apa yang dipakai policy — dan ablasi yang memisahkan keduanya menjawab tidak.

PREREG-V6 §6 sudah mendaftar keadaan ini di muka sebagai hasil sah: *"representasi membaik tapi KPI tidak → hasil sah, dan itu justru gerbang PLAN-04 §0c"*. Di sini bahkan representasinya pun tidak membaik.

### Konsekuensi mengikat untuk PLAN-06

Klaim **"GNN belajar koordinasi yang bermakna secara fisik" tidak didukung, dan tidak boleh ditulis.** Kriterianya ditetapkan PLAN-06 §2 sendiri sebelum datanya ada — *"KPI turun signifikan → atensi dipakai (kausal); KPI stabil → atensi hanya dekorasi"* — dan KPI-nya stabil.

Yang boleh ditulis sebagai gantinya, dan ini pernyataan yang lebih sempit sekaligus lebih kuat karena ia bertahan terhadap ablasi:

> Atensi berkorelasi dengan struktur interferensi, tetapi policy tidak memanfaatkan korelasi itu.

Bacaan itu tidak berdiri sendiri. Ia konsisten dengan tiga temuan yang sudah lebih dulu tegak: perilaku **lockstep** antar-agen, **over-smoothing** yang dikonfirmasi D3 dan D6, dan KPI yang berulang kali `COMPARABLE` terhadap baseline non-GNN. Keempatnya menggambarkan mekanisme yang sama dari empat sudut — informasi tetangga masuk, lalu dihomogenkan sebelum sempat membedakan keputusan.

Catatan hasil ini sudah dituliskan balik ke `docs/revisi/PLAN-06-MECHANISM-EVIDENCE.md` §2 supaya larangan klaimnya hidup di dokumen yang mengatur penulisan paper, bukan cuma di file journey ini.

---

## 8. Uji lanjutan — prediksi ditulis sebelum dijalankan

Bagian ini di-commit **sebelum** ujinya dijalankan. Tanpa itu uji ini tidak punya daya falsifikasi: prediksi yang ditulis setelah melihat hasil selalu cocok.

**Pertanyaannya:** kenapa ablasi inert. §5 menyarankan sebabnya keruntuhan separasi di `conv2` — apa pun yang ditimbang atensi di `conv1` tidak lagi punya jalan keluar ke kepala policy. Itu hipotesis penjelasan, dan ia bisa diuji dengan mengablasi satu layer pada satu waktu alih-alih keduanya sekaligus.

**Yang diperkirakan:** kedua kondisi inert. Kalau keruntuhan `conv2` benar yang memutus jalurnya, meratakan atensi `conv1` saja tidak akan mengubah apa pun karena `conv2` tetap menghomogenkan keluarannya, dan meratakan `conv2` saja juga tidak karena yang ia terima sudah nyaris seragam.

**Yang akan memfalsifikasi**, dinyatakan lengkap supaya tidak ada ruang menafsirkan belakangan:

| Hasil | Bacaan yang mengikat |
|---|---|
| `conv1` inert **dan** `conv2` inert | Hipotesis didukung |
| `conv1` berpengaruh, `conv2` tidak | **Hipotesis jatuh.** Kalau atensi `conv1` sampai ke keputusan, keruntuhan `conv2` bukan penjelasannya |
| `conv2` berpengaruh, `conv1` tidak | Atensi terpakai, tapi hanya di layer terakhir. Verdict null §3 perlu **dipersempit**, bukan dicabut |
| Keduanya berpengaruh padahal gabungannya tidak | Efeknya saling meniadakan; perlu penyelidikan terpisah sebelum klaim apa pun |

**Yang tidak berubah apa pun hasilnya.** Ketiga temuan di §2 dan §3 tidak bergantung pada uji ini. Pembanding tetap salah file, keselarasan per-node tetap memburuk, dan ablasi gabungan tetap inert sejak v4. Uji per-layer menjelaskan sebab; ia tidak bisa membatalkan sebab-akibat yang sudah terukur.

**Protokolnya:** flag `--ablate-layers {both,conv1,conv2}` pada `scripts/attention_analysis.py`, default `both` yang harus terbukti **bit-identik** dengan run `ATTENTION_V6.md` pada checkpoint yang sama sebelum kondisi baru dijalankan. Kalau tidak identik, ada yang berubah selain penambahan flag, dan itu dikejar dulu.

---

## 9. Hasil uji per-layer — prediksi terpenuhi

Dijalankan 2026-09-24, 75 checkpoint per kondisi, protokol identik §3. Angka di bawah `timely_throughput_mbps`, checkpoint hidup saja.

| kondisi | n | inert | mean | sd | t | rugi | untung |
|---|---|---|---|---|---|---|---|
| `both` (§3) | 60 | 29/60 | +0,194 | 3,105 | 0,48 | 18 | 13 |
| `conv1` saja | 60 | 29/60 | +0,174 | 2,952 | 0,46 | 18 | 13 |
| `conv2` saja | 60 | 31/60 | +0,081 | 0,364 | 1,73 | 17 | 12 |

`sla_satisfaction_pct` bergerak sejalan: t 0,42 / 0,39 / 1,71.

**Prediksi §8 terpenuhi, tapi kedua kondisi tidak sama bentuknya.** Nol cabang falsifikasi terpicu. Bedanya harus dinyatakan, bukan diratakan jadi satu kata:

- **`conv2` inert.** 31/60 berubah persis nol, dan yang bergerak pun bergerak seragam kecil (sd 0,364).
- **`conv1` tidak berpengaruh secara konsisten** — bukan inert. 31 dari 60 checkpoint punya efek non-nol, sd 2,95 Mbps, ke dua arah. Ada efek; yang tidak ada adalah polanya.

Pembedaan ini menentukan langkah lanjutan, jadi bukan soal gaya. "Inert" berarti tidak ada yang bisa dikejar. "Tidak konsisten" berarti ada sesuatu yang bergerak dan pertanyaannya apakah ia berpola — dan §9.1 menjawab: tidak.

Tabel prediksi §8 memakai kata "inert" untuk kedua kondisi. **Kata itu sengaja dibiarkan apa adanya**; ia pra-registrasi yang di-commit sebelum ujinya jalan, dan menyuntingnya sekarang berarti menulis ulang prediksi setelah melihat hasil. Cabang falsifikasinya berbunyi "`conv1` berpengaruh", dan efek tersebar dua arah tanpa pola bukan itu — jadi prediksinya bertahan, dan koreksi katanya hidup di sini, bukan di sana.

### 9.1 Sebaran 31 checkpoint conv1 non-inert — tersebar, bukan mengumpul

Kalau efek besar mengumpul di satu arm atau di satu status seed, "tidak konsisten" akan jadi kesimpulan sementara yang menunggu penjelasan. Ia tidak mengumpul.

| arm | non-inert | mean abs d | max abs d |
|---|---|---|---|
| `gnn-madqn_gatedge` | 4/5 | 0,793 | 1,603 |
| `gnn-madqn_gatres` | 5/5 | 1,680 | 3,401 |
| `gnn-madqn_gatres-edge` | 5/5 | 0,868 | 2,478 |
| `gnn-mappo_gatedge` | 8/16 | 2,408 | 6,349 |
| `gnn-mappo_gatres` | 4/15 | 6,521 | 13,920 |
| `gnn-mappo_gatres-edge` | 5/14 | 4,331 | 9,395 |

- **Keenam arm terwakili.** Tidak ada arm yang bersih, tidak ada arm yang menyetir.
- **Konsentrasi rendah.** Checkpoint terbesar menyumbang 16,8% dari total besaran efek, tiga teratas 36,2%, sepuluh teratas 76,0%. Efek yang berpola akan jauh lebih terpusat dari ini.
- **Dua arah.** Terbesar −13,920 (`gnn-mappo_gatres` seed 45), kedua +9,395 (`gnn-mappo_gatres-edge` seed 53). Ablasi sama seringnya menolong dan merugikan.
- **Status kolaps tidak memisahkan.** Seed kolaps 5/12 non-inert, tidak kolaps 26/48 — 42% lawan 54%, beda yang tenggelam di n sekecil ini.

Satu pola nyata memang muncul, dan bentuknya bukan "conv1 berpengaruh" melainkan dua ragam ketidakkonsistenan: **DQN hampir selalu bergerak sedikit** (14/15 non-inert, mean abs d 0,79–1,68), **PPO jarang bergerak tapi besar** (17/45 non-inert, mean abs d 2,41–6,52). Itu sejalan dengan mekanisme argmax lawan sampling, bukan dengan atensi yang terpakai.

**Kesimpulan akhir: tidak konsisten.** Bukan "ada efek yang belum terdeteksi".

### Dua hal yang lebih tajam dari sekadar "conv1 lemah, conv2 inert"

**`conv1` sendirian mereproduksi ablasi penuh.** Mean +0,174 lawan +0,194, jumlah inert sama persis 29/60, pembagian tanda sama persis 18 rugi lawan 13 untung. Artinya seluruh efek ablasi penuh — yang memang kecil — berasal dari `conv1`, dan `conv2` nyaris tidak menyumbang apa-apa.

**Efek `conv2` satu orde lebih kecil, dan sebaran perubahannya runtuh.** Mean 0,081 lawan 0,194, tapi yang lebih menunjuk adalah sd: **0,364 lawan 3,105**, delapan setengah kali lebih sempit. Meratakan `conv2` menghasilkan perubahan yang kecil *dan seragam kecil*; meratakan `conv1` sesekali melempar satu-dua checkpoint jauh.

Itu persis yang diramalkan hipotesis. `conv2` menerima masukan yang sudah nyaris seragam (§5: `rel_spread` 0,0003–0,0070), jadi menimbang ulang masukan seragam memang tidak bisa mengubah banyak. `conv1` masih menerima masukan yang terbedakan (0,16–0,37), jadi meratakannya sesekali cukup untuk membalik argmax — dan ekor gemuk itulah yang membuat sd-nya delapan kali lebih lebar.

### Jebakan bacaan yang harus dinyatakan

Nilai t terbesar di seluruh tabel ini justru milik `conv2` (1,73). Itu **bukan** tanda atensi `conv2` terpakai. t naik karena penyebutnya mengecil, bukan karena pembilangnya membesar: mean 0,081 Mbps di atas basis ~65 Mbps adalah **0,12%**. Membacanya sebagai "conv2 paling berpengaruh" akan membalik arti datanya sendiri.

Nol kondisi mendekati ambang konvensional mana pun; yang tertinggi 1,75 di tingkat per-kelompok.

### Verifikasi yang menyertai

- **Gerbang bit-identik lolos sebelum kondisi baru dijalankan.** `--ablate-layers both` pada 3 checkpoint menghasilkan 27 angka yang cocok persis sampai 6 desimal dengan baris yang sama di `ATTENTION_V6.md`. Jadi flag baru tidak mengubah apa pun selain memilih layer.
- **Aritmetika edge mengonfirmasi `--episodes 10`**, yang sebelumnya hanya turunan: run verifikasi melaporkan 120.000 edge = 3 ckpt × 10 ep × 200 langkah × 20 edge.
- **Statistik korelasi identik di ketiga laporan** (Pearson −0,1548, Spearman −0,2044, mean rho −0,2089, 3.000.000 edge). Jalur penangkapan atensi memang tidak boleh tersentuh ablasi, dan angka yang sama persis membuktikannya.

---

## Data & artefak

| Isi | File |
|---|---|
| Korelasi + ablasi atensi v6, 75 checkpoint | `results/ATTENTION_V6.md` |
| Ablasi `conv1` saja | `results/ATTENTION_V6_conv1.md` |
| Ablasi `conv2` saja | `results/ATTENTION_V6_conv2.md` |
| Pembanding v4 setara (readout greedy, arm `gat`) | `results/ATTENTION_v4_greedy.md` |
| Laporan era pra-v4, **bukan** pembanding sah | `results/ATTENTION.md` |
| D2a/D2b/D3 pada checkpoint v6 | `results/DIAG_GNN_RELIANCE_V6.md` |
| D6 separabilitas per tahap | `results/DIAG_INPUT_SEP_V6.md` |
| Definisi arm dan `edge_dim`-nya | `gnn/__init__.py` |
| Kewajiban pasca-wave dan daftar hasil yang tetap dilaporkan | `docs/revisi/PREREG-V6.md` §5, §6 |

[← v5 Diblokir dan Wave v6](11_v5-diblokir-dan-wave-v6.md) | [Index](00_INDEX.md)
