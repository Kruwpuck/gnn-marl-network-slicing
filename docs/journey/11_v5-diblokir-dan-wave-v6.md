[← Diagnostik Fase 0](10_diagnostik-fase-0.md) | [Index](00_INDEX.md)

# 11 — v5 Diblokir jadi Temuan, v6 Jalan Duluan

**Periode:** 2026-08-25 → 2026-08-31
**Commit:** `b48dd79` (constraint resilient, terblokir) → `187b349` (handover pasca-wave PPO)
**Status saat file ini ditulis:** wave PPO v6 selesai, wave DQN v6 **sedang berjalan**, **nol KPI v6 dibaca**

> **File ini belum memuat hasil.** Ia menutup di keadaan operasional, bukan di verdict. Angka KPI v6 belum satu pun dibaca, jadi nol klaim dibuat tentang ketiga arm arsitektur baru. Hasilnya jadi file journey berikutnya.

---

## Ringkasan eksekutif

1. **Kalibrasi `f_min` tidak menghasilkan kandidat, dan skripnya menolak memilih satu.** Itu dilaporkan sebagai temuan tentang titik operasi, bukan sebagai kalibrasi yang gagal dijalankan — dan `calibrate_fmin.py` tidak ditambal.
2. **Dua godaan ditolak dan dicatat**, keduanya akan menghasilkan `f_min` yang terlihat wajar: aturan peringkat yang menetapkan constraint dari distribusi metode *proposed* sendiri, dan penggantian keluarga referensi.
3. **Urutan estafet ditukar.** Rencana fitur edge (v6) tidak bergantung pada keluaran rencana constraint (v5), jadi v6 jalan duluan sementara v5 menunggu keputusan manusia.
4. **Arm pembanding `gat` tidak dilatih ulang** — digerbangi uji identitas numerik bit-per-bit, bukan pembacaan riwayat commit. Hemat ~166 job-jam PPO.
5. **Tiga cacat instrumen lagi**, dan yang ketujuh **arahnya terbalik**: pengukuran yang melaporkan sakit padahal sehat, yang langsung memicu tindakan destruktif — sebuah wave sehat dimatikan dan 4,2 jam latihan hilang.

---

## 1. `f_min` tidak punya nilai yang sekaligus feasible dan mengikat

Rencana v5 menambahkan constraint laju minimum per-gNB untuk melindungi cell-edge. Sebelum wave-nya bisa jalan, `f_min` harus dikalibrasi. Dua populasi diukur:

| Populasi | Hasil |
|---|---|
| Alokasi statis, 9 fraksi | pelanggaran terendah **0,0964** di frac 0,9 — di atas `delta = 0,085`. **Nol fraksi feasible** |
| Checkpoint v4 terlatih | referensi `ippo` melanggar (**0,1013**) **dan** kolaps di **5/5** seed (`embb_p5` 0,0000) |

Yang statis memang diperkirakan gagal, karena `delta` dikalibrasi terhadap policy **terlatih** (file 08 §3). Yang kedua yang jadi masalah.

### Kenapa ini temuan, bukan kegagalan implementasi

Baris kedua bukan `f_min` yang salah pilih — **dua constraint berebut PRB yang sama.** Bandwidth-nya tetap: melindungi cell-edge eMBB berarti mengambil PRB dari URLLC, jadi menambah constraint kedua membuat yang pertama lebih sulit dipenuhi. Gate A ronde 6 lolos justru **karena** constraint resilient belum aktif; kelolosannya bukan bukti keduanya bisa hidup bersama di titik operasi yang sama.

Jadi temuannya dinyatakan begini:

> Pada titik operasi v4, tidak ada `f_min` yang sekaligus **feasible** (shortfall konvergen) dan **mengikat** (μ steady-state > 0) bagi baseline referensi non-GNN.

Bentuknya identik dengan temuan `delta` di file 08, cuma di sumbu berbeda: di sana rentang yang bisa dijangkau terlalu sempit untuk memisahkan apa pun; di sini ruang feasible-nya kosong. Keduanya temuan tentang **task**, bukan kecelakaan kalibrasi.

Laporan yang di-generate menutup sendiri dengan kalimat yang sama: *"That is a finding about the operating point, not a number to soften: report it and do not pick an `f_min` anyway."*

### Dua godaan yang ditolak

**Godaan 1 — aturan peringkat.** "eMBB rata-rata tertinggi di antara yang feasible", dijalankan atas seluruh checkpoint, memilih `gnn-mappo` (eMBB 10,064 Mbps, violation 0,0448) — yaitu **menetapkan `f_min` dari distribusi yang bisa dicapai metode proposed sendiri**, menanamkan keunggulannya ke dalam constraint yang nanti dipakai mengujinya. Larangan rencananya melarang persis itu. Pembatasan ke keluarga referensi non-GNN sekarang **dipaksakan di kode** (`REFERENCE_ALGOS`), bukan diserahkan ke aturan peringkat. Kandidat `gnn-mappo` juga bersandar pada **1 dari 10** run; batas minimum 3 run non-kolaps kini dinyatakan di muka supaya nol persentil pernah berdiri di atas satu seed.

**Godaan 2 — mengganti keluarga referensi** supaya ada yang lolos. Ditolak dengan alasan yang sama: memilih referensi setelah melihat mana yang memberi jawaban yang enak adalah memindahkan tiang gawang.

---

## 2. Urutan ditukar, dan v6 pra-registrasi lebih dulu

Rencana fitur edge tidak bergantung pada keluaran rencana constraint, jadi urutannya ditukar (keputusan manusia 2026-08-26). v5 menunggu keputusan manusia atas titik operasi; v6 jalan.

**Pra-registrasi v6 ditulis sebelum satu pun arm dilatih**, dan isinya sengaja memuat hal-hal yang tidak menguntungkan:

- **Prediksi null dinyatakan di muka** — `gatedge` sendirian diperkirakan menunjukkan efek terbatas, karena informasi tambahannya tetap dihomogenkan `conv1`.
- **Daftar apa yang dilaporkan apa pun hasilnya:** `gatedge` null, representasi membaik tapi KPI datar, KPI membaik tapi zero-shot rusak, dan "v6 tidak memperbaiki apa pun" — keempatnya hasil sah. Seed yang kolaps tidak dibuang. Hasil v4 tetap dilaporkan penuh.
- **Bacaan yang dilarang:** `gatedge` yang `COMPARABLE` **tidak boleh** dibaca sebagai "fitur edge tidak berguna". Bacaan yang sah adalah "efeknya tidak punya jalan keluar", dan yang memisahkan kedua bacaan itu adalah arm gabungan.

### Empat arm, dipilih dengan alasan mekanistik

| Arm | `edge_dim` | residual | Isi |
|---|---|---|---|
| `gat` | 1 | tidak | v4, **pembanding** — checkpoint lama, bukan run baru |
| `gatres` | 1 | ya | residual menjangkau **input** pada kedua layer, α = 0,1 |
| `gatedge` | 2 | tidak | kolom edge kedua = `interference_coupling` |
| `gatres-edge` | 2 | ya | keduanya |

Arm gabungan ikut **bukan demi kelengkapan faktorial**, melainkan karena D6 (file 10 §6) memperkirakan kedua mekanismenya **berurutan**: residual menyelamatkan separasi, lalu fitur edge punya sesuatu untuk disumbangkan pada separasi yang selamat. Kalau benar, efek gabungannya superaditif — dan itu tidak bisa terbaca dari dua arm terisolasi saja.

### Kenapa kolom edge kedua menambah informasi meski turunannya sudah ada di node

Keberatan yang paling wajar: SINR sudah ada di observasi node, dan kopling interferensi salah satu bahan penyusunnya. Jawabannya bukan besarannya melainkan **bentuknya**:

> SINR pada observasi node adalah **agregat seluruh interferer** — satu penjumlahan yang membuang identitas penyumbangnya. Kolom edge memberi **dekomposisi per-tetangga** dari agregat yang sama. Agen tahu "SINR saya jelek", tapi tanpa fitur edge tidak tahu "gara-gara tetangga mana".

Koordinasi yang mensyaratkan "kurangi alokasi terhadap tetangga tertentu" memang tidak punya dasar informasi di observasi node — dan itulah yang message passing seharusnya sediakan.

---

## 3. Pembanding `gat` dipakai ulang — digerbangi uji numerik

Melatih ulang `gat` akan memakan ~166 job-jam PPO plus arm DQN-nya. Keputusannya: **pakai checkpoint v4 apa adanya** (20 seed PPO, 5 seed DQN).

Syaratnya satu — dinamika environment harus identik antara saat v4 dilatih dan sekarang. Riwayat commit memberi tujuh alasan untuk mengira nol perubahan, tapi **gerbangnya dinyatakan sebagai uji, bukan sebagai pembacaan kode**, dan gerbangnya ditulis sebelum ujinya dijalankan: beda sekecil apa pun berarti `gat` dilatih ulang sebagai arm keempat, dan selisihnya dilaporkan sebagai temuan.

Ujinya: pohon di `c73de09` (keadaan kode saat wave v4) melawan HEAD, config sama, **200 langkah dengan barisan aksi yang sama**. Hasil: `obs` (202×5×8), `reward`, `embb_thr_bps`, kolom path loss `edge_attr`, `edge_index`, dan himpunan kunci `info` **identik bit-per-bit**. Uji lolos, jadi jalur latih-ulang tidak terpakai.

**Konsekuensi yang harus dijaga sejak titik ini:** checkpoint `_v4` berstatus **pembanding aktif, bukan arsip** — beda status dari `results/v1_uncoupled/` dan `v2_scalarized/`. Ia tidak boleh ditimpa, dipindah, atau dibersihkan, dan integritasnya diperiksa sebelum dan sesudah tiap pekerjaan.

Satu sifat struktural ikut menutup satu kelas kesalahan: environment **selalu memancarkan kedua kolom edge**, dan yang membedakan arm cuma `edge_dim` backbone yang meng-slice kolom pertama saja kalau nilainya 1. Jadi nol config per-arm, jadi **nol cara memasangkan config yang salah dengan checkpoint saat evaluasi**.

---

## 4. Tiga cacat instrumen lagi — dan yang ketujuh arahnya terbalik

Daftar instrumen yang melaporkan sehat padahal salah kini punya **tujuh** instansi. Tiga di antaranya lahir di babak ini.

### Cacat #5 — arm baru menyamar jadi pembandingnya sendiri

`parse_run_name` memakai alternasi regex yang **tidak terpanjang-dulu**, jadi `gnn-mappo_gatres_seed42` terpecah salah: `gatres` cocok dengan grup **tag**, algonya terbaca `gnn-mappo` telanjang, dan ketiga arm baru runtuh jadi satu baris. Ia **tidak gagal** — laporannya tetap terbit.

### Cacat #6 — arm baru tidak punya bagian sama sekali

`MATCHED_BASELINES` mengeraskan empat nama algo v4. Tanpa entri baru, laporan rliable tetap diproduksi dan tetap **exit 0**, hanya saja ketiga arm v6 **tidak punya section apa pun** — wave yang baru saja dijalankan tidak muncul di laporannya sendiri.

Keduanya ditangkap oleh **smoke wave** (3 arm × 2 seed × 50k langkah) yang dijalankan sebelum wave penuh. Empat dari lima cek pipa lolos apa adanya; yang kelima inilah yang menyelamatkan 75 job.

### Cacat #7 — melaporkan sakit padahal sehat

Ini yang paling mahal, dan satu-satunya dari tujuh yang arahnya terbalik.

**Yang dikira terjadi (2026-08-26):** satu perintah `run_wave.py` melahirkan dua proses berargumen identik dengan start terpaut 9 milidetik, masing-masing menurunkan enam trainer — dibaca sebagai dua wave berjalan bersamaan. Wave dihentikan.

**Yang sebenarnya terjadi, diukur ulang 2026-08-27 dari satu peluncuran tunggal:** `.venv\Scripts\python.exe` adalah **stub peluncur**, bukan salinan interpreter — 274.712 byte dengan hash berbeda dari `Python311\python.exe` yang 103.192 byte. Tiap pemanggilan venv karena itu **selalu** dua proses, dan yang kedua adalah **anak** dari yang pertama (`ParentProcessId` cocok), dengan si anak yang benar-benar menjalankan skripnya.

Satu induk + 6 trainer = 7 pasang = **14 proses** — persis sidik jari yang dibaca sebagai dua wave. Selisih 9 ms itu jarak stub memanggil anaknya; "4 proses per backbone" adalah 2 job × 2 proses, bukan 2 penulis per job.

**Biayanya langsung: wave sehat dimatikan, 4,2 jam latihan hilang, dan sebuah cacat orkestrasi tercatat di ledger padahal tidak pernah terjadi.**

Kontrafaktualnya justru sudah terjadi: artefak wave itu **utuh** — satu inversi langkah per CSV, tepat di batas resume; dua belas checkpoint dimuat penuh. Pemeriksaan artefak itu sudah menjawab benar sejak awal. Yang keliru adalah **menomorduakan bukti artefak di bawah hitungan proses.**

> **Pelajarannya, dan ia berlaku di luar insiden ini: hitungan proses bukan instrumen integritas data.** Yang mengikat ada tiga — nilai langkah duplikat, inversi langkah di luar batas resume, dan checkpoint yang gagal dimuat — dan ketiganya bersih sejak awal.

Koreksinya ditulis sebagai **entri ledger baru**, bukan dengan menyunting entri lama, dan ia membatalkan sebagian entri 2026-08-26 beserta sebagian commit yang mengikutinya.

### Kunci tetap dipasang, tapi statusnya jujur

`acquire_wave_lock` (`scripts/run_wave.py`) membuat lockfile per-tag dengan `os.O_EXCL`. Yang dijaganya adalah mode gagal yang **belum pernah teramati**: dua peluncuran sungguhan atas satu tag, misalnya dari dua terminal. `O_EXCL` dipilih karena atomik — pemeriksaan cek-lalu-buat akan meloloskan dua proses yang berangkat berdekatan.

Dua batasnya dinyatakan di muka, bukan ditemukan orang lain nanti: **kunci mati harus dihapus manual** (tidak ada deteksi pid-hidup; pesan penolakannya menyebut pid dan argv pemegangnya supaya menghapusnya jadi keputusan, bukan tebakan), dan **lingkupnya per-tag** (wave PPO dan DQN bertag sama tidak bisa jalan serentak).

---

## 5. Keadaan per 2026-08-31

| | |
|---|---|
| Wave PPO v6 | **SELESAI** — `Wave done in 71.27h. all jobs ok`, 60 job (3 arm × 20 seed 42–61), nol gagal |
| Jalan | 2026-08-27 12:03:04 → 2026-08-30 11:19:30 |
| Checkpoint final `_v6` | 60 |
| Wave DQN v6 | **BERJALAN** — dilepas 2026-08-31 14:17, 15 job (3 arm × 5 seed 42–46) |
| Evaluasi & laporan v6 | **BELUM SATU PUN** |
| KPI v6 yang sudah dibaca | **nol** |

Dua cacat instrumen tambahan diperbaiki sebelum diagnostik v6 dijalankan, keduanya konsekuensi dari arm ber-`edge_dim` 2:

- `strip_neighbours` (D2a) mengeraskan `edge_attr` selebar 1 kolom, sehingga `GATv2Conv` yang dibangun untuk 2 kolom menolaknya — **crash**, jadi ketahuan.
- Korelasi atensi meratakan `edge_attr` (E,2) jadi 2E nilai lalu memasangkannya ke E edge lewat `zip` yang **memotong diam-diam** — tiap edge mengambil path loss milik tetangganya, dan korelasinya tetap tercetak sebagai angka yang wajar. **Tidak crash, dan itu yang lebih berbahaya.**

Keduanya dikunci di `tests/test_diag_edge_dim.py`.

Yang wajib dijalankan sesudah wave DQN selesai, sebelum klaim apa pun: evaluasi keenam arm, laporan rliable yang menjangkau dua tag (`_v6` untuk arm, `_v4` untuk pembanding), **D2b diulang** pada checkpoint v6, dan **D6 diulang**. D2b penting karena uji `allclose` di `tests/test_gnn_v6.py` cuma membuktikan `edge_attr` mengubah keluaran **backbone** — itu bukan bukti policy memakainya, dan pembandingnya 0/25 dari v4 (file 10 §3).

---

## 6. Utang terbuka

- **`resilient.f_min_mbps` dan titik operasi** menunggu keputusan manusia. Kriteria a priori untuk menggeser titik operasi sudah dikunci, tapi belum dieksekusi. Kegagalan kalibrasinya **temuan**, dan tidak boleh ditambal.
- **Analisis atensi v6** boleh diulang sesudah kolom kedua ada, dengan ablasi kausal — dan sekarang instrumennya sudah benar untuk 2 kolom.
- **Terminologi paper:** varian `gat` sebenarnya GATv2 (file 10 §1, P1). Belum dikoreksi di draf.

---

## Data & artefak

| Isi | File |
|---|---|
| Temuan `f_min`, dua godaan yang ditolak | `docs/revisi/PREREG-V5.md` §0 |
| Tabel kalibrasi + kalimat "no candidate" | `results/CALIBRATE_FMIN.md` |
| Hipotesis per-arm, prediksi null, daftar hasil yang tetap dilaporkan | `docs/revisi/PREREG-V6.md` §1, §6 |
| Uji identitas numerik yang menggerbangi pemakaian ulang `gat` | `docs/revisi/PREREG-V6.md` §2, §7 |
| Tujuh instansi cacat instrumen + kontrafaktualnya | `docs/revisi/PLAN-06-MECHANISM-EVIDENCE.md` §5 |
| Keadaan operasional pasca-wave + perintah berikutnya | `docs/revisi/HANDOVER-V6.md` |
| Log wave PPO v6 | `results/logs/stdout/wave_v6_ppo.out` |
| Log wave DQN v6 (berjalan) | `results/logs/stdout/wave_v6_dqn.out` |
| Kronologi bertanda waktu, termasuk koreksi insiden | `runs/2026-08-05-run01/ledger.md` (2026-08-25 → 2026-08-27) |
| Regresi lebar `edge_attr` untuk kedua diagnostik | `tests/test_diag_edge_dim.py` |

[← Diagnostik Fase 0](10_diagnostik-fase-0.md) | [Index](00_INDEX.md)
