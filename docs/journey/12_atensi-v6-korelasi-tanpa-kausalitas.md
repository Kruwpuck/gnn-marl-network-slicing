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

## Data & artefak

| Isi | File |
|---|---|
| Korelasi + ablasi atensi v6, 75 checkpoint | `results/ATTENTION_V6.md` |
| Pembanding v4 setara (readout greedy, arm `gat`) | `results/ATTENTION_v4_greedy.md` |
| Laporan era pra-v4, **bukan** pembanding sah | `results/ATTENTION.md` |
| D2a/D2b/D3 pada checkpoint v6 | `results/DIAG_GNN_RELIANCE_V6.md` |
| D6 separabilitas per tahap | `results/DIAG_INPUT_SEP_V6.md` |
| Definisi arm dan `edge_dim`-nya | `gnn/__init__.py` |
| Kewajiban pasca-wave dan daftar hasil yang tetap dilaporkan | `docs/revisi/PREREG-V6.md` §5, §6 |

[← v5 Diblokir dan Wave v6](11_v5-diblokir-dan-wave-v6.md) | [Index](00_INDEX.md)
